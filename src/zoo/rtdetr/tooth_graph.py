"""
ToothGraphEnhance: tooth-anatomy-guided cross-scale graph enhancement
======================================================================
GraphFPN-inspired module specialised for dental panoramic X-rays.

Design:
  * Nodes    = spatial patches of P2 and P3 (patch_size x patch_size blocks).
  * Edges    = within-scale 8-neighbourhood (grid) + cross-scale
               (each P3 patch <-> the 4 P2 patches it spatially covers).
  * Edge weights = product of the two nodes' tooth-mask coverage (soft gate):
               message passing is strong inside teeth (where lesions live) and
               weak off-tooth — the dental prior shapes WHERE features
               propagate, without hard masking or removing anything.
  * Message  = one normalised aggregation layer, applied as a RESIDUAL delta
               (out = feat + up(delta)). Nothing is removed, so the module can
               only add feature signal and cannot reduce other objects.
  * Cross-scale propagation lets fine P2 detail reinforce coarse P3 context
               and vice-versa — the GraphFPN "hierarchical" idea — which is
               exactly the signal small lesions are missing.

Interface: `GraphHybridEncoder` runs it on the FPN/PAN outputs P2/P3, reading
the tooth mask from the targets (train + eval, like the density prior).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rtdetr import RTDETR
from .rtdetr_criterion import SetCriterion
from .hybrid_encoder import HybridEncoder
from src.core import register

__all__ = ['ToothGraphEnhance', 'GraphHybridEncoder', 'GraphRTDETR', 'GraphCriterion']


class ToothGraphEnhance(nn.Module):
    def __init__(self, channels=256, patch_size=4, mask_eps=0.1,
                 cross_scale=True):
        super().__init__()
        self.patch_size = patch_size
        self.mask_eps = mask_eps
        self.cross_scale = cross_scale
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        # GAT-style learned source importance: each node's weight as a message
        # SOURCE is learned, so the graph propagates from informative (lesion)
        # nodes instead of blind-averaging all neighbours.
        self.src_lin = nn.Linear(channels, 1)
        # Node-level lesion-prediction head (aux supervision): supervised on the
        # P2 node grid to predict "this patch contains a lesion", which shapes
        # the node features (and thus the messages) to be lesion-aware. This
        # replicates the gpf_p2 baseline's DRFL train-time aux recipe through
        # the graph module.
        self.lesion_head = nn.Conv2d(channels, 1, 1)
        self._graph_loss = None
        self.register_buffer('_ones3', torch.ones(1, 1, 3, 3))

    # ---- helpers -------------------------------------------------------
    def _neigh_sum(self, x):
        """Sum over the 3x3 spatial neighbourhood, per-channel."""
        B, C, H, W = x.shape
        w = self._ones3.expand(C, -1, -1, -1)
        return F.conv2d(x, w, padding=1, groups=C)

    def _mask_grid(self, mask, h, w, B, device):
        """Tooth mask [B,1,H0,W0] -> soft node-grid mask [B,1,h,w] in [eps,1].

        Adaptive avg-pool = fraction of each patch covered by teeth (soft,
        no hard threshold, so off-tooth edges decay smoothly).
        """
        if mask is None:
            return torch.ones(B, 1, h, w, device=device)
        m = F.adaptive_avg_pool2d(mask, (h, w))
        return self.mask_eps + (1.0 - self.mask_eps) * m

    def _refine(self, x):
        """Channel-wise MLP + LayerNorm on a [B,C,h,w] node grid."""
        x = x.permute(0, 2, 3, 1)          # [B,h,w,C]
        x = self.norm(self.proj(x))
        return x.permute(0, 3, 1, 2)

    def _build_lesion_labels(self, targets, h, w, B, device):
        """Map lesion boxes (cxcywh normalized, train format) to node labels."""
        labels = torch.zeros(B, 1, h, w, device=device)
        if targets is None:
            return labels
        for b, t in enumerate(targets):
            boxes = t.get('boxes')
            if boxes is None or len(boxes) == 0:
                continue
            boxes = boxes.float()
            cx, cy, bw, bh = boxes.unbind(-1)
            x0 = ((cx - bw / 2) * w).clamp(0, w - 1)
            x1 = ((cx + bw / 2) * w).clamp(0, w)
            y0 = ((cy - bh / 2) * h).clamp(0, h - 1)
            y1 = ((cy + bh / 2) * h).clamp(0, h)
            for i in range(len(boxes)):
                xi0, xi1 = int(x0[i]), max(int(x1[i]), int(x0[i]) + 1)
                yi0, yi1 = int(y0[i]), max(int(y1[i]), int(y0[i]) + 1)
                labels[b, 0, yi0:yi1, xi0:xi1] = 1.0
        return labels

    def _focal_bce(self, logits, labels, alpha=0.25, gamma=2.0, fg_w=50.0):
        """Focal BCE for sparse node-level lesion prediction.

        Positive nodes are strongly amplified (1 + fg_w*label) — the mean over
        ~1600 nodes dilutes the ~10 positive nodes, so without amplification the
        aux loss stays ~0.01 and its gradient to the shared P2 features is
        drowned by the detection loss. Amplification restores the DRFL-like
        magnitude that actually shapes features.
        """
        p = torch.sigmoid(logits)
        p_t = p * labels + (1 - p) * (1 - labels)
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        fg = 1.0 + fg_w * labels
        return (fg * alpha_t * (1 - p_t).pow(gamma) * ce).mean()

    def _within_scale_msg(self, nodes, m):
        """Tooth-gated, source-attentive 8-neighbour aggregation.

        msg_i = m_i * (sum_j w_j h_j) / (sum_j w_j + eps),
        w_j = sigmoid(src_lin(h_j)) * m_j   (learned source importance).
        The graph learns which neighbours are informative (lesion) sources to
        propagate from, instead of blind-averaging all neighbours.
        """
        src = torch.sigmoid(self.src_lin(nodes.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        w = src * m
        num = self._neigh_sum(w * nodes)          # sum_j w_j h_j
        den = self._neigh_sum(w)                  # sum_j w_j
        return m * num / (den + 1e-6)

    # ---- main ----------------------------------------------------------
    def forward(self, feats, mask=None, targets=None):
        """feats = [P2 (B,C,H2,W2), P3 (B,C,H3,W3)], mask = [B,1,H0,W0] or None.
        Returns the two enhanced feature maps (same shapes, residual-added).
        At train with targets, also computes the node-level lesion aux loss
        (stored in self._graph_loss)."""
        self._graph_loss = None
        p2, p3 = feats[0], feats[1]
        B, C, H2, W2 = p2.shape
        _, _, H3, W3 = p3.shape
        ps = self.patch_size
        h2, w2 = H2 // ps, W2 // ps
        h3, w3 = H3 // ps, W3 // ps

        # patch-pool to node grids
        n2 = F.avg_pool2d(p2, ps)                # [B,C,h2,w2]
        n3 = F.avg_pool2d(p3, ps)                # [B,C,h3,w3]
        m2 = self._mask_grid(mask, h2, w2, B, p2.device)
        m3 = self._mask_grid(mask, h3, w3, B, p3.device)

        # --- Change 1: node-level lesion aux supervision (train only) ---
        if self.training and targets is not None:
            labels = self._build_lesion_labels(targets, h2, w2, B, p2.device)
            logits = self.lesion_head(n2)
            self._graph_loss = self._focal_bce(logits, labels)

        msg2 = self._within_scale_msg(n2, m2)
        msg3 = self._within_scale_msg(n3, m3)

        if self.cross_scale:
            # P3 parent -> P2 children (nearest upsample, tooth-gated)
            parent_up = F.interpolate(n3, size=(h2, w2), mode='nearest')
            m3_up = F.interpolate(m3, size=(h2, w2), mode='nearest')
            msg2 = msg2 + m2 * m3_up * parent_up
            # P2 children -> P3 parent (mask-weighted child average)
            m2pool = F.avg_pool2d(m2, 2)
            msg3 = msg3 + F.avg_pool2d(m2 * n2, 2) / (m2pool + 1e-6)

        # residual deltas, scattered back to spatial resolution.
        # Gate the FINAL delta by the node tooth value: LayerNorm normalises the
        # message vector to unit scale, so without this the tooth gating would
        # only affect direction, not magnitude. With it, enhancement is strong
        # on teeth and weak off-tooth (off-tooth residual ~ identity).
        d2 = m2 * self._refine(msg2)
        d3 = m3 * self._refine(msg3)
        out_p2 = p2 + F.interpolate(d2, size=(H2, W2), mode='nearest')
        out_p3 = p3 + F.interpolate(d3, size=(H3, W3), mode='nearest')
        return out_p2, out_p3


@register
class GraphHybridEncoder(HybridEncoder):
    """HybridEncoder + tooth-anatomy-guided cross-scale graph enhancement on
    the FPN/PAN P2 and P3 outputs. Standard dense AIFI, no sparse attention,
    no density head. The graph reads the tooth mask from targets (train+eval)."""
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 graph_enabled=True,
                 graph_patch_size=4,
                 graph_mask_eps=0.1,
                 graph_cross_scale=True):
        super().__init__(
            in_channels=in_channels, feat_strides=feat_strides,
            hidden_dim=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, enc_act=enc_act, use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers, pe_temperature=pe_temperature,
            expansion=expansion, depth_mult=depth_mult, act=act,
            eval_spatial_size=eval_spatial_size)
        self.graph_enabled = graph_enabled
        if graph_enabled:
            self.tooth_graph = ToothGraphEnhance(
                channels=hidden_dim, patch_size=graph_patch_size,
                mask_eps=graph_mask_eps, cross_scale=graph_cross_scale)

    @staticmethod
    def _get_mask(targets, device, B):
        if targets is None:
            return None
        masks = [t.get('tooth_mask', None) if isinstance(t, dict) else None
                 for t in targets]
        if all(m is not None for m in masks):
            tm = torch.stack([m.float() for m in masks], 0).unsqueeze(1)
            return tm.to(device)
        return None

    def forward(self, feats, images=None, targets=None):
        outs = super().forward(feats)   # standard dense FPN/PAN
        self._graph_loss = None
        if self.graph_enabled:
            mask = self._get_mask(targets, outs[0].device, outs[0].shape[0])
            outs[0], outs[1] = self.tooth_graph([outs[0], outs[1]], mask, targets)
            self._graph_loss = getattr(self.tooth_graph, '_graph_loss', None)
        return outs


def _choice(seq):
    """random.choice works on any sized sequence (list or np.array)."""
    import random
    return random.choice(seq)


@register
class GraphRTDETR(RTDETR):
    """RTDETR that forwards the graph module's node-level lesion aux loss to
    the output dict so the criterion can add it to the total loss."""
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = _choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        images = x.clone()  # keep raw images for expert priors
        x = self.backbone(x)
        if self.training:
            enc_targets = targets
        elif targets is not None and any(
                isinstance(t, dict) and 'tooth_mask' in t for t in targets):
            enc_targets = [{k: t[k] for k in ('tooth_mask', 'tooth_idmap')
                            if k in t} for t in targets]
        else:
            enc_targets = None
        try:
            x = self.encoder(x, images=images, targets=enc_targets)
        except TypeError:
            x = self.encoder(x)

        x = self.decoder(x, targets if self.training else None)

        if isinstance(x, dict):
            if (self.training and hasattr(self.encoder, '_graph_loss')
                    and self.encoder._graph_loss is not None):
                x['graph_lesion_loss'] = self.encoder._graph_loss
        return x


@register
class GraphCriterion(SetCriterion):
    """SetCriterion + graph module's node-level lesion aux loss."""
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, matcher, weight_dict, losses,
                 alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80,
                 graph_lesion_weight=1.0):
        super().__init__(matcher, weight_dict, losses,
                         alpha=alpha, gamma=gamma,
                         eos_coef=eos_coef, num_classes=num_classes)
        self.graph_lesion_weight = graph_lesion_weight

    def forward(self, outputs, targets, **kwargs):
        losses = super().forward(outputs, targets, **kwargs)
        gl = outputs.get('graph_lesion_loss')
        if gl is not None:
            losses['graph_lesion_loss'] = gl * self.graph_lesion_weight
        return losses
