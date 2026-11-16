"""
AnatomyDensityGraph (density-contributes-features): toothgraph_idmap base +
a density extractor whose FEATURES are fused into P2 at train AND eval.
================================================================================
Previous attempt (density gate) failed because the gate was TRAIN-ONLY: the
decoder learned on gated features at train but saw ungated features at eval ->
train/eval mismatch -> APs stalled. This version removes the gate and the
consistency loss entirely. Instead the density EXTRACTOR's intermediate
multi-scale features (dilated depthwise 1/3/5, tooth-mask gated) are fused
directly into the graph-enhanced P2:

    out_p2 = p2 + graph_delta                 (toothgraph base)
    feat, density = DensityExtractor(out_p2)  (multi-scale lesion features)
    out_p2 = out_p2 + fuse_proj(feat)         <-- NO training gate -> CONSISTENT
                                                  train AND eval

Auxiliary loss modes:
  legacy: original pre-graph focal supervision and DRFL-inspired regression.
  refined_balanced: Graph-refined Node Loss on n2 + d2 and Instance-balanced
    Density Loss with per-instance local means and a separate background mean.
The legacy mode remains the default for reproducibility. New configurations
explicitly select refined_balanced. Gaussian targets and inference stay unchanged.
Density recall weighting is adapted from Dome-DETR (Hu et al., ACM MM 2025);
node supervision retains the focal modulation of Lin et al. (ICCV 2017).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tooth_graph_idmap import ToothGraphIdmapEnhance, GraphIdmapHybridEncoder
from .defe_loss import GaussHeatmapGenerator
from src.core import register

__all__ = ['DensityExtractor', 'AnatomyDensityGraph', 'AnatomyDensityHybridEncoder']


class DensityExtractor(nn.Module):
    """Multi-scale lesion-feature extractor, tooth-mask gated (dilation 1/3/5).
    Returns the intermediate features (fused into P2 at train AND eval) and the
    density output (auxiliary supervision, train only). The output remains learnable
    outside the predicted tooth mask."""

    def __init__(self, in_ch=256, mid_ch=64, mask_dilation=2):
        super().__init__()
        self.mask_dilation = mask_dilation
        self.reduce = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1), nn.SiLU(inplace=True))
        self.b1 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, dilation=1, groups=mid_ch)
        self.b2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=3, dilation=3, groups=mid_ch)
        self.b3 = nn.Conv2d(mid_ch, mid_ch, 3, padding=5, dilation=5, groups=mid_ch)
        self.fuse = nn.Sequential(nn.Conv2d(mid_ch * 3, mid_ch, 1), nn.SiLU(inplace=True))
        self.gate_conv = nn.Conv2d(mid_ch, 1, 3, padding=1)
        self.residual = nn.Conv2d(in_ch, mid_ch, 1)
        self.proj = nn.Conv2d(mid_ch, 1, 1)

    def _tooth_mask(self, targets, H, W, B, device):
        masks = []
        if targets is not None:
            for t in targets:
                tm = t.get('tooth_mask', None) if isinstance(t, dict) else None
                masks.append(tm)
        if masks and all(m is not None for m in masks):
            tm = torch.stack([m.float() for m in masks], 0).unsqueeze(1).to(device)
            tm = F.interpolate(tm, size=(H, W), mode='nearest')
        else:
            tm = torch.ones(B, 1, H, W, device=device)
        if self.mask_dilation > 0:
            k = 2 * self.mask_dilation + 1
            tm = (F.max_pool2d((tm > 0).float(), kernel_size=k, stride=1,
                               padding=self.mask_dilation) > 0).float()
        return tm

    def forward(self, x, targets):
        B, C, H, W = x.shape
        tm = self._tooth_mask(targets, H, W, B, x.device)
        h = self.reduce(x)
        h = self.fuse(torch.cat([self.b1(h), self.b2(h), self.b3(h)], 1))
        g = torch.sigmoid(self.gate_conv(h)) * tm
        feat = h * g + self.residual(x)                        # [B,mid_ch,H,W]
        density = torch.sigmoid(self.proj(feat))               # [B,1,H,W]
        return feat, density


class AnatomyDensityGraph(ToothGraphIdmapEnhance):
    def __init__(self, channels=256, patch_size=4, mask_eps=0.1,
                 cross_scale=True, cross_tooth_w=0.5,
                 density_w=1.0, density_recall_penalty=0.3,
                 density_sigma_ratio=1.2, ramp_epochs=30, current_epoch=0,
                 aux_loss_mode='legacy', node_loss_weight=1.0,
                 node_negative_weight=1.0, density_background_weight=0.25):
        super().__init__(channels, patch_size, mask_eps, cross_scale,
                         cross_tooth_w)
        if aux_loss_mode not in ('legacy', 'refined_balanced'):
            raise ValueError(f"Unknown auxiliary loss mode: {aux_loss_mode}")
        if min(node_loss_weight, node_negative_weight, density_background_weight) < 0:
            raise ValueError("Auxiliary loss weights must be nonnegative")
        self.aux_loss_mode = aux_loss_mode
        self.node_loss_weight = node_loss_weight
        self.node_negative_weight = node_negative_weight
        self.density_background_weight = density_background_weight
        self._aux_loss_stats = None
        self.density_w = density_w
        self.density_recall_penalty = density_recall_penalty
        self.density_sigma_ratio = density_sigma_ratio
        self.ramp_epochs = max(1, ramp_epochs)
        self.current_epoch = current_epoch
        self.density_head = DensityExtractor(in_ch=channels, mid_ch=64,
                                             mask_dilation=2)
        self.fuse_proj = nn.Conv2d(64, channels, 1)   # extractor feat -> P2

    def _ramp(self):
        """AMR: density emphasis ramps 0 -> 1 over ramp_epochs."""
        return min(1.0, self.current_epoch / self.ramp_epochs)

    def _build_density_labels(self, targets, H, W, B, device):
        """Gaussian-blob density target (peak-normalised, capped at 1)."""
        gen = GaussHeatmapGenerator(img_size=(H, W),
                                    sigma_ratio=self.density_sigma_ratio,
                                    peak_normalize=True)
        labels = torch.zeros(B, 1, H, W, device=device)
        for b, t in enumerate(targets):
            boxes = t.get('boxes')
            if boxes is None or len(boxes) == 0:
                continue
            hm = gen(boxes.detach().cpu().float())
            labels[b] = hm.to(device)
        return labels

    def _density_loss(self, density, target):
        """DRFL: up-weight MISSED lesion regions."""
        diff = density - target
        under = (density < target).float()
        w = 1.0 + self.density_recall_penalty * target * under
        return (w * diff ** 2).mean()

    def _graph_refined_node_loss(self, logits, labels, alpha=0.25, gamma=2.0):
        """Graph-refined Node Loss: separate positive/negative focal means.

        The caller attaches this head to residual-updated fine-scale nodes.
        Empty classes contribute zero. Reduction is per image, then per batch.
        No fixed 51x positive multiplier is used in this mode.
        """
        z, y = logits.float(), labels.float()
        p = z.sigmoid()
        pt = p * y + (1 - p) * (1 - y)
        alpha_t = alpha * y + (1 - alpha) * (1 - y)
        ce = F.binary_cross_entropy_with_logits(z, y, reduction='none')
        focal = alpha_t * (1 - pt).pow(gamma) * ce
        positive, negative = y > 0.5, y <= 0.5
        dims = tuple(range(1, z.ndim))
        pos = (focal * positive).sum(dims) / positive.sum(dims).clamp_min(1)
        neg = (focal * negative).sum(dims) / negative.sum(dims).clamp_min(1)
        return (pos + self.node_negative_weight * neg).mean()

    def _instance_balanced_density_loss(self, density, target, targets):
        """Instance-balanced Density Loss on the existing merged Gaussian target.

        Each valid GT box has unit total spatial weight before averaging over
        instances in an image. Fractional pixel overlap preserves subpixel boxes.
        Overlapping boxes use the SAME merged target; they never impose competing
        instance-specific targets. Pixels outside all boxes form a separate mean.
        Background is regressed to the existing Gaussian target (including tails),
        not forced to zero where that target is nonzero. No predicted-mask clipping.
        """
        pred, truth = density.float(), target.float()
        diff = pred - truth
        error = (1 + self.density_recall_penalty * truth * (pred < truth)) * diff.square()
        batch, _, height, width = pred.shape
        per_image = []
        for b in range(batch):
            local_means = []
            occupied = torch.zeros((height, width), dtype=torch.bool, device=pred.device)
            boxes = targets[b].get('boxes')
            if boxes is not None:
                boxes = boxes.detach().to(device=pred.device, dtype=torch.float32)
                for cx, cy, bw, bh in boxes:
                    if not bool(torch.isfinite(torch.stack((cx, cy, bw, bh))).all()):
                        continue
                    if bw <= 0 or bh <= 0:
                        continue
                    x0, x1 = ((cx - bw / 2) * width).clamp(0, width), ((cx + bw / 2) * width).clamp(0, width)
                    y0, y1 = ((cy - bh / 2) * height).clamp(0, height), ((cy + bh / 2) * height).clamp(0, height)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    ix0, ix1 = int(x0.floor()), int(x1.ceil())
                    iy0, iy1 = int(y0.floor()), int(y1.ceil())
                    xs = torch.arange(ix0, ix1, device=pred.device, dtype=torch.float32)
                    ys = torch.arange(iy0, iy1, device=pred.device, dtype=torch.float32)
                    wx = (torch.minimum(xs + 1, x1) - torch.maximum(xs, x0)).clamp_min(0)
                    wy = (torch.minimum(ys + 1, y1) - torch.maximum(ys, y0)).clamp_min(0)
                    weight = wy[:, None] * wx[None, :]
                    local_means.append((error[b, 0, iy0:iy1, ix0:ix1] * weight).sum()
                                       / weight.sum().clamp_min(torch.finfo(torch.float32).tiny))
                    occupied[iy0:iy1, ix0:ix1] |= weight > 0
            foreground = torch.stack(local_means).mean() if local_means else error[b].sum() * 0
            outside = ~occupied
            background = (error[b, 0] * outside).sum() / outside.sum().clamp_min(1)
            per_image.append(foreground + self.density_background_weight * background)
        return torch.stack(per_image).mean()

    def forward(self, feats, mask=None, targets=None):
        self._graph_loss = None
        self._aux_loss_stats = None
        p2, p3 = feats[0], feats[1]
        B, C, H2, W2 = p2.shape
        _, _, H3, W3 = p3.shape
        ps = self.patch_size
        h2, w2 = H2 // ps, W2 // ps
        h3, w3 = H3 // ps, W3 // ps

        n2 = F.avg_pool2d(p2, ps)                  # [B,C,h2,w2]
        n3 = F.avg_pool2d(p3, ps)
        tid2 = self._tooth_id_grid(targets, h2, w2, B, p2.device)
        tid3 = self._tooth_id_grid(targets, h3, w3, B, p3.device)
        m2 = self._mask_grid(mask, h2, w2, B, p2.device)
        m3 = self._mask_grid(mask, h3, w3, B, p3.device)

        # node-level lesion aux (base behaviour)
        aux = None
        if self.training and targets is not None and self.aux_loss_mode == 'legacy':
            labels = self._build_lesion_labels(targets, h2, w2, B, p2.device)
            node_logits = self.lesion_head(n2)
            aux = self._focal_bce(node_logits, labels)

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
        if self.training and targets is not None and self.aux_loss_mode == 'refined_balanced':
            labels = self._build_lesion_labels(targets, h2, w2, B, p2.device)
            aux = self._graph_refined_node_loss(self.lesion_head(n2 + d2), labels)
        out_p2 = p2 + F.interpolate(d2, size=(H2, W2), mode='nearest')
        out_p3 = p3 + F.interpolate(d3, size=(H3, W3), mode='nearest')

        # ---- DensityExtractor contributes features (train AND eval, CONSISTENT) ----
        feat, density = self.density_head(out_p2, targets)
        out_p2 = out_p2 + self.fuse_proj(feat)

        # Training-only node/density supervision; feature fusion is unchanged.
        if self.training and aux is not None:
            dlabels = self._build_density_labels(targets, H2, W2, B, p2.device)
            ramp = self._ramp()
            if self.aux_loss_mode == 'legacy':
                d_loss = self._density_loss(density, dlabels)
                self._graph_loss = aux + self.density_w * ramp * d_loss
            else:
                d_loss = self._instance_balanced_density_loss(density, dlabels, targets)
                self._graph_loss = self.node_loss_weight * aux + self.density_w * ramp * d_loss
            self._aux_loss_stats = {'loss_node': aux.detach(), 'loss_density': d_loss.detach(),
                                    'density_ramp': ramp}
        return out_p2, out_p3


@register
class AnatomyDensityHybridEncoder(GraphIdmapHybridEncoder):
    """GraphIdmapHybridEncoder with the density-contributes-features graph."""

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
                 graph_cross_tooth_w=0.5,
                 graph_density_w=1.0,
                 graph_density_recall_penalty=0.3,
                 graph_density_sigma_ratio=1.2,
                 graph_ramp_epochs=30,
                 graph_aux_loss_mode='legacy',
                 graph_node_loss_weight=1.0,
                 graph_node_negative_weight=1.0,
                 graph_density_background_weight=0.25):
        super().__init__(
            in_channels=in_channels, feat_strides=feat_strides,
            hidden_dim=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, enc_act=enc_act, use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers, pe_temperature=pe_temperature,
            expansion=expansion, depth_mult=depth_mult, act=act,
            eval_spatial_size=eval_spatial_size, graph_enabled=graph_enabled,
            graph_patch_size=graph_patch_size, graph_mask_eps=graph_mask_eps,
            graph_cross_scale=graph_cross_scale,
            graph_cross_tooth_w=graph_cross_tooth_w)
        self.current_epoch = 0
        if graph_enabled:
            self.tooth_graph = AnatomyDensityGraph(
                channels=hidden_dim, patch_size=graph_patch_size,
                mask_eps=graph_mask_eps, cross_scale=graph_cross_scale,
                cross_tooth_w=graph_cross_tooth_w,
                density_w=graph_density_w,
                density_recall_penalty=graph_density_recall_penalty,
                density_sigma_ratio=graph_density_sigma_ratio,
                ramp_epochs=graph_ramp_epochs,
                aux_loss_mode=graph_aux_loss_mode,
                node_loss_weight=graph_node_loss_weight,
                node_negative_weight=graph_node_negative_weight,
                density_background_weight=graph_density_background_weight)

    def forward(self, feats, images=None, targets=None):
        self.tooth_graph.current_epoch = self.current_epoch
        return super().forward(feats, images=images, targets=targets)
