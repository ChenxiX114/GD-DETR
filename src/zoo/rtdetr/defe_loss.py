"""
DRFL: Density Regression Focal Loss
Adapted from Dome-DETR (ACM MM 2025)

The DRFL loss supervises the DeFE module's density map prediction.
It penalizes underestimation of object density more heavily than
overestimation — missing a lesion region is worse than including
some background.

Key formula:
    diff = density_pred - density_gt
    underestimation_mask = (density_pred < density_gt)
    weight = 1 + density_recall_penalty * density_gt * underestimation_mask
    loss = mean(weight * diff^2)
"""

import torch
import torch.nn as nn


class GaussHeatmapGenerator:
    """Generate ground-truth density maps from bounding box annotations.

    Places a 2D Gaussian at each object center, with sigma proportional
    to the object's width/height. Overlapping Gaussians are summed.

    Args:
        img_size: (H, W) of the image the bboxes are defined on
        sigma_ratio: controls Gaussian spread relative to bbox size
    """

    def __init__(self, img_size=(640, 640), sigma_ratio=1.2, peak_normalize=True):
        self.img_size = img_size
        self.sigma_ratio = sigma_ratio
        self.peak_normalize = peak_normalize

    def __call__(self, bboxes):
        """Generate heatmap from list of bboxes.

        Args:
            bboxes: [N, 4] tensor in (cx, cy, w, h) format, normalized [0, 1]

        Returns:
            heatmap: [1, H, W] tensor
        """
        H, W = self.img_size
        heatmap = torch.zeros((H, W), dtype=torch.float32)

        for box in bboxes:
            x_center, y_center, width, height = box
            x_center_px = int(x_center * W)
            y_center_px = int(y_center * H)
            w_px = max(int(width * W), 1)
            h_px = max(int(height * H), 1)

            sigma_x = max(w_px * self.sigma_ratio, 1.0)
            sigma_y = max(h_px * self.sigma_ratio, 1.0)

            kernel = self._gaussian_kernel(sigma_x, sigma_y)
            if kernel.numel() == 0:
                continue

            k_h, k_w = kernel.shape
            radius_x = k_w // 2
            radius_y = k_h // 2

            x_start = max(x_center_px - radius_x, 0)
            y_start = max(y_center_px - radius_y, 0)
            x_end = min(x_center_px + radius_x + 1, W)
            y_end = min(y_center_px + radius_y + 1, H)

            k_start_x = max(radius_x - (x_center_px - x_start), 0)
            k_start_y = max(radius_y - (y_center_px - y_start), 0)
            k_end_x = k_w - max((x_center_px + radius_x + 1) - x_end, 0)
            k_end_y = k_h - max((y_center_px + radius_y + 1) - y_end, 0)

            kernel_cropped = kernel[k_start_y:k_end_y, k_start_x:k_end_x]
            if kernel_cropped.numel() == 0:
                continue

            patch_h = y_end - y_start
            patch_w = x_end - x_start
            if kernel_cropped.shape != (patch_h, patch_w):
                kernel_cropped = kernel_cropped[:patch_h, :patch_w]

            heatmap[y_start:y_end, x_start:x_end] += kernel_cropped

        if self.peak_normalize:
            # per-instance peak normalization: each kernel is max-normalized to 1
            # (see _gaussian_kernel), so every lesion center targets ~1 regardless
            # of its size. No global-max re-scaling; overlapping lesions capped at 1.
            heatmap = heatmap.clamp(max=1.0)
        else:
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

        return heatmap.unsqueeze(0)

    def _gaussian_kernel(self, sigma_x, sigma_y):
        sigma_x = max(sigma_x, 0.1)
        sigma_y = max(sigma_y, 0.1)
        kernel_w = int(6 * sigma_x) + 1
        kernel_h = int(6 * sigma_y) + 1
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1

        x = torch.arange(kernel_w, dtype=torch.float32) - (kernel_w // 2)
        y = torch.arange(kernel_h, dtype=torch.float32) - (kernel_h // 2)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        kernel = torch.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))
        if self.peak_normalize:
            # peak = 1 at every lesion center (raw Gaussian): small lesions get a
            # full-strength target instead of being down-weighted by mass
            # normalization (which spreads a big lesion over more pixels).
            kernel = kernel / kernel.max().clamp(min=1e-6)
        else:
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel = kernel / kernel_sum
        return kernel


def compute_defe_losses(defe_info, targets, img_size=(640, 640),
                         density_map_weight=1.0, density_recall_penalty=0.3,
                         peak_normalize=True, defe_fg_weight=2.0):
    """Compute DeFE supervision losses from Dome-DETR.

    Args:
        defe_info: dict from DefeHybridEncoder._defe_info with keys:
                   'density'  — [B, 1, H, W] predicted density map
                   'reg_value' — [B, 1] predicted object count ratio
        targets: list of target dicts with 'boxes' key
        img_size: (H, W) of original images
        density_map_weight: scalar weight for density map loss
        density_recall_penalty: penalty factor for underestimation (default 0.3)

    Returns:
        losses: dict with 'defe_density_loss' and optional 'defe_reg_loss'
    """
    density_pred = defe_info['density']  # [B, 1, H_d, W_d]
    reg_value = defe_info.get('reg_value', None)
    B = density_pred.shape[0]

    heatmap_generator = GaussHeatmapGenerator(img_size=img_size,
                                              peak_normalize=peak_normalize)
    losses = {}

    # --- Density Map Loss (DRFL: asymmetric penalty for underestimation) ---
    gt_density_maps = []
    H, W = img_size
    for b in range(B):
        boxes = targets[b]['boxes'].clone()
        if boxes.shape[0] == 0:
            # Image with no objects: GT density is all zeros
            gt_density_maps.append(torch.zeros(1, H, W, dtype=torch.float32))
            continue
        # Convert whatever box format to cxcywh (normalized).
        # Training targets are cxcywh-normalized (dataloader ConvertBox).
        # Also handle xyxy absolute / xyxy normalized / xywh for robustness.
        b = boxes
        if b[:, 2].max() > 2.0:
            # absolute pixel coords: xyxy (x2>x1) or xywh (w small)
            if (b[:, 2] > b[:, 0]).float().mean() > 0.5:  # xyxy
                cx = (b[:, 0] + b[:, 2]) / 2
                cy = (b[:, 1] + b[:, 3]) / 2
                w = b[:, 2] - b[:, 0]
                h = b[:, 3] - b[:, 1]
            else:  # xywh
                cx = b[:, 0] + b[:, 2] / 2
                cy = b[:, 1] + b[:, 3] / 2
                w = b[:, 2]
                h = b[:, 3]
            cx, cy, w, h = cx / W, cy / H, w / W, h / H
        else:
            # normalized [0,1]: cxcywh (w,h small) or xyxy (x2>=x1)
            if (b[:, 2] >= b[:, 0]).float().mean() > 0.5:  # xyxy normalized
                cx = (b[:, 0] + b[:, 2]) / 2
                cy = (b[:, 1] + b[:, 3]) / 2
                w = b[:, 2] - b[:, 0]
                h = b[:, 3] - b[:, 1]
            else:  # cxcywh normalized (training)
                cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        boxes_cxcywh = torch.stack([cx, cy, w, h], dim=1)

        heatmap = heatmap_generator(boxes_cxcywh)
        gt_density_maps.append(heatmap)

    gt_density_map = torch.stack(gt_density_maps).to(density_pred.device)

    # Interpolate GT to match predicted density size
    if gt_density_map.shape[-2:] != density_pred.shape[-2:]:
        gt_density_map = torch.nn.functional.interpolate(
            gt_density_map, size=density_pred.shape[-2:],
            mode='bilinear', align_corners=False
        )

    # DRFL core: weighted MSE with asymmetric penalty
    diff = density_pred - gt_density_map
    underestimation_mask = (density_pred < gt_density_map).float()
    penalty_weight = 1.0 + density_recall_penalty * gt_density_map * underestimation_mask
    # Foreground emphasis: small lesions occupy a tiny pixel fraction, so a plain
    # mean over the whole image dilutes their gradient and the density branch
    # under-fires at them. Amplify lesion-pixel losses by (1 + defe_fg_weight*gt).
    fg_weight = 1.0 + defe_fg_weight * gt_density_map
    defe_density_loss = (fg_weight * penalty_weight * (diff ** 2)).mean() * density_map_weight
    losses['defe_density_loss'] = defe_density_loss

    # --- Regression Loss (object count prediction) ---
    if reg_value is not None:
        min_num = 0
        max_num = 100
        reg_targets = []
        for t in targets:
            num = max(min(t['labels'].shape[0], max_num), min_num)
            reg_targets.append((num - min_num) / (max_num - min_num))
        reg_targets = torch.tensor(reg_targets, dtype=torch.float32,
                                    device=reg_value.device).unsqueeze(1)
        diff_reg = reg_value - reg_targets
        penalty_weights_reg = torch.where(diff_reg < 0, 2.0, 1.0).to(diff_reg.device)
        defe_reg_loss = (penalty_weights_reg * (diff_reg ** 2)).mean()
        losses['defe_reg_loss'] = defe_reg_loss

    return losses


def compute_tooth_density_loss(defe_info, targets, img_size=(640, 640)):
    """Tooth-anchored density supervision (MIL style).

    The density map is free-form 2D; this anchors it to anatomical tooth
    units. For each tooth box (target['tooth_boxes_t'], cxcywh normalized,
    transformed with the image), pool the max density inside the tooth and
    supervise it with a binary label: high iff the tooth overlaps a lesion GT
    box. Differentiated: we supervise DENSITY STATISTICS per tooth, not tooth
    segmentation (DentalX) nor crop-and-redetect (DentMamba).

    Returns {'tooth_density_loss': scalar}.
    """
    import torch.nn.functional as F
    density = defe_info['density']                     # [B,1,H,W] in [0,1]
    B = density.shape[0]
    H, W = density.shape[-2:]

    def _to_xy(boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - w / 2) * W; x2 = (cx + w / 2) * W
        y1 = (cy - h / 2) * H; y2 = (cy + h / 2) * H
        return torch.stack([x1, y1, x2, y2], -1)

    losses = []
    for b in range(B):
        tb = targets[b].get('tooth_boxes_t', None)
        gb = targets[b].get('boxes', None)
        if tb is None or gb is None or len(tb) == 0 or len(gb) == 0:
            continue
        tb = tb.detach().float()
        gb = gb.detach().float()
        tb_xy = _to_xy(tb)
        tb_xy = torch.stack([
            tb_xy[:, 0].round().long().clamp(min=0),
            tb_xy[:, 1].round().long().clamp(min=0),
            (tb_xy[:, 2].round().long() + 1).clamp(max=W),
            (tb_xy[:, 3].round().long() + 1).clamp(max=H),
        ], -1)
        gb_xy = _to_xy(gb)
        dens = density[b, 0]                            # [H, W]
        vals = []
        for i in range(len(tb_xy)):
            x1, y1, x2, y2 = tb_xy[i]
            if x2 <= x1 or y2 <= y1:
                vals.append(dens.new_zeros(()))
            else:
                vals.append(dens[y1:y2, x1:x2].max())
        tooth_max = torch.stack(vals)                   # [T]
        # has_lesion: tooth box overlaps any GT box (grid coords)
        gx1, gy1, gx2, gy2 = gb_xy.unbind(-1)
        tx1, ty1, tx2, ty2 = tb_xy.float().unbind(-1)
        ix1 = torch.maximum(tx1[:, None], gx1[None, :])
        iy1 = torch.maximum(ty1[:, None], gy1[None, :])
        ix2 = torch.minimum(tx2[:, None], gx2[None, :])
        iy2 = torch.minimum(ty2[:, None], gy2[None, :])
        has_lesion = (((ix2 - ix1) > 0) & ((iy2 - iy1) > 0)).any(dim=1).float()
        if has_lesion.sum() == 0 or has_lesion.sum() == len(has_lesion):
            continue   # skip degenerate batches (all pos / all neg)
        # class-balanced weighting for the MIL BCE (capped so a rare positive
        # tooth cannot dominate the loss at init)
        w_pos = min(len(has_lesion) / max(has_lesion.sum(), 1), 10.0)
        w_neg = min(len(has_lesion) / max(len(has_lesion) - has_lesion.sum(), 1), 10.0)
        weight = torch.where(has_lesion > 0, w_pos, w_neg)
        losses.append(F.binary_cross_entropy(tooth_max, has_lesion,
                                             weight=weight))

    if losses:
        return {'tooth_density_loss': sum(losses) / len(losses)}
    return {'tooth_density_loss': density.new_zeros(())}
