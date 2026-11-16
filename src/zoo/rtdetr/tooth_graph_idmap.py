"""
ToothGraphIdmap: toothgraph-weight3.0 + the exact per-tooth idmap
================================================================================
The base ToothGraphEnhance reads the UNION tooth mask, which only says "is this
node inside any tooth". The per-tooth idmap additionally says WHICH tooth. This
module uses that extra bit of anatomy: message passing propagates STRONGLY
within a tooth (same-tooth edges, weight 1) and WEAKLY across tooth boundaries
(cross-tooth edges, weight `cross_tooth_w`). The dental prior thus acts at the
patch level per tooth, instead of a single binary in/out gate.

This is a clean ablation: everything else is identical to the toothgraph-weight3.0
baseline (ToothGraphEnhance message, lesion aux, cross-scale, GraphCriterion).
The union mask controls node coverage at both train and eval. When a per-tooth
idmap is available, it only modulates same-tooth and cross-tooth edge weights.
The base tooth_graph.py is untouched.
"""
import torch
import torch.nn.functional as F

from .tooth_graph import ToothGraphEnhance, GraphHybridEncoder, GraphRTDETR, GraphCriterion
from src.core import register

__all__ = ['ToothGraphIdmapEnhance', 'GraphIdmapHybridEncoder']


class ToothGraphIdmapEnhance(ToothGraphEnhance):
    def __init__(self, channels=256, patch_size=4, mask_eps=0.1,
                 cross_scale=True, cross_tooth_w=0.5):
        super().__init__(channels, patch_size, mask_eps, cross_scale)
        self.cross_tooth_w = cross_tooth_w

    # ---- idmap helpers --------------------------------------------------
    @staticmethod
    def _shift(x, dy, dx):
        """Shift a [B,C,h,w] tensor by (dy,dx), zero-filled at the edges."""
        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        sy0, sy1 = max(0, -dy), min(H, H - dy)
        sx0, sx1 = max(0, -dx), min(W, W - dx)
        out[:, :, sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = x[:, :, sy0:sy1, sx0:sx1]
        return out

    def _tooth_id_grid(self, targets, h, w, B, device):
        """Per-node tooth ID [B,1,h,w] (0 = off-tooth) from the aligned idmap,
        nearest-downsampled. None if the field is missing (fall back to the
        union-mask behaviour of the base module)."""
        if targets is None:
            return None
        ids = []
        for t in targets:
            m = t.get('tooth_idmap') if isinstance(t, dict) else None
            if m is None:
                return None
            ids.append(m.float()[None])
        m = torch.stack(ids, 0).to(device)
        return F.interpolate(m, size=(h, w), mode='nearest').long()   # [B,1,h,w]

    def _within_scale_msg(self, nodes, m, tid=None):
        """Base message + same-tooth edge weighting. Each of the 8 neighbours
        contributes  sigmoid(src_lin) * m * tooth_factor  where tooth_factor = 1
        if the neighbour is in the SAME tooth, else cross_tooth_w."""
        src = torch.sigmoid(self.src_lin(nodes.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        w = src * m
        if tid is None:
            num = self._neigh_sum(w * nodes)
            den = self._neigh_sum(w)
            return m * num / (den + 1e-6)
        tf = tid.float()
        num = torch.zeros_like(nodes)
        den = torch.zeros_like(m)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                sh_t = self._shift(tf, dy, dx)
                factor = torch.where(sh_t == tf, torch.ones_like(tf),
                                     torch.full_like(tf, self.cross_tooth_w))
                sh_w = self._shift(w, dy, dx)
                num = num + factor * sh_w * self._shift(nodes, dy, dx)
                den = den + factor * sh_w
        return m * num / (den + 1e-6)

    def forward(self, feats, mask=None, targets=None):
        """Use the union mask for coverage and optional tooth IDs for edges."""
        self._graph_loss = None
        p2, p3 = feats[0], feats[1]
        B, C, H2, W2 = p2.shape
        _, _, H3, W3 = p3.shape
        ps = self.patch_size
        h2, w2 = H2 // ps, W2 // ps
        h3, w3 = H3 // ps, W3 // ps

        n2 = F.avg_pool2d(p2, ps)                # [B,C,h2,w2]
        n3 = F.avg_pool2d(p3, ps)
        tid2 = self._tooth_id_grid(targets, h2, w2, B, p2.device)
        tid3 = self._tooth_id_grid(targets, h3, w3, B, p3.device)

        m2 = self._mask_grid(mask, h2, w2, B, p2.device)
        m3 = self._mask_grid(mask, h3, w3, B, p3.device)

        # node-level lesion aux (train only), unchanged from base
        if self.training and targets is not None:
            labels = self._build_lesion_labels(targets, h2, w2, B, p2.device)
            logits = self.lesion_head(n2)
            self._graph_loss = self._focal_bce(logits, labels)

        msg2 = self._within_scale_msg(n2, m2, tid2)
        msg3 = self._within_scale_msg(n3, m3, tid3)

        if self.cross_scale:
            parent_up = F.interpolate(n3, size=(h2, w2), mode='nearest')
            m3_up = F.interpolate(m3, size=(h2, w2), mode='nearest')
            msg2 = msg2 + m2 * m3_up * parent_up
            m2pool = F.avg_pool2d(m2, 2)
            msg3 = msg3 + F.avg_pool2d(m2 * n2, 2) / (m2pool + 1e-6)

        d2 = m2 * self._refine(msg2)
        d3 = m3 * self._refine(msg3)
        out_p2 = p2 + F.interpolate(d2, size=(H2, W2), mode='nearest')
        out_p3 = p3 + F.interpolate(d3, size=(H3, W3), mode='nearest')
        return out_p2, out_p3


@register
class GraphIdmapHybridEncoder(GraphHybridEncoder):
    """GraphHybridEncoder whose graph is the per-tooth-idmap-aware
    ToothGraphIdmapEnhance. The union mask controls node coverage; an idmap,
    when supplied, changes only the within-scale edge weights."""

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
                 graph_cross_scale=True,
                 graph_cross_tooth_w=0.5):
        super().__init__(
            in_channels=in_channels, feat_strides=feat_strides,
            hidden_dim=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, enc_act=enc_act, use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers, pe_temperature=pe_temperature,
            expansion=expansion, depth_mult=depth_mult, act=act,
            eval_spatial_size=eval_spatial_size, graph_enabled=graph_enabled,
            graph_patch_size=graph_patch_size, graph_mask_eps=graph_mask_eps,
            graph_cross_scale=graph_cross_scale)
        if graph_enabled:
            self.tooth_graph = ToothGraphIdmapEnhance(
                channels=hidden_dim, patch_size=graph_patch_size,
                mask_eps=graph_mask_eps, cross_scale=graph_cross_scale,
                cross_tooth_w=graph_cross_tooth_w)
