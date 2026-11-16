"""
OralXrays9Idmap: CocoDetection subclass that also returns the REAL per-tooth
idmap (exact tooth boundaries, precomputed) as an aligned field.
================================================================================
The per-tooth graph previously derived tooth membership with a nearest-box-
center (Voronoi) approximation over the (overlapping) tooth boxes. This subclass
adds the ground-truth per-tooth label map (`tooth_idmaps_*.npy`, 256x256, value
k = tooth index k, 0 = background) as a `tv_tensors.Mask` field injected BEFORE
the transforms, so flip/resize/zoomout align it exactly like the union mask.

The base `CocoDetection` is untouched; this subclass is used only by configs
that opt in. NOTE: do not combine with tooth_crop_prob > 0 (a tooth crop shifts
coordinates and would break idmap alignment).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F_nn

from torchvision import tv_tensors

from src.core import register

from .coco_dataset import CocoDetection

__all__ = ['OralXrays9Idmap']


@register
class OralXrays9Idmap(CocoDetection):
    __inject__ = ['transforms']

    def __init__(self, img_folder, ann_file, transforms, return_masks,
                 remap_mscoco_category=False,
                 return_original=False, tooth_boxes_path=None,
                 orig_max_dim=1280,
                 tooth_crop_prob=0.0, tooth_crop_inflate_min=1.6,
                 tooth_crop_inflate_max=2.4,
                 return_tooth_boxes_t=False,
                 return_tooth_mask=False, tooth_masks_path=None,
                 return_tooth_idmap=False, tooth_idmaps_path=None):
        super().__init__(
            img_folder, ann_file, transforms, return_masks,
            remap_mscoco_category=remap_mscoco_category,
            return_original=return_original, tooth_boxes_path=tooth_boxes_path,
            orig_max_dim=orig_max_dim,
            tooth_crop_prob=tooth_crop_prob,
            tooth_crop_inflate_min=tooth_crop_inflate_min,
            tooth_crop_inflate_max=tooth_crop_inflate_max,
            return_tooth_boxes_t=return_tooth_boxes_t,
            return_tooth_mask=return_tooth_mask,
            tooth_masks_path=tooth_masks_path)
        self.return_tooth_idmap = return_tooth_idmap
        self._tooth_idmaps = None
        if tooth_idmaps_path and os.path.exists(tooth_idmaps_path):
            self._tooth_idmaps = np.load(tooth_idmaps_path,
                                         allow_pickle=True).item()

    def __getitem__(self, idx):
        # CocoDetection.__getitem__ + the tooth_idmap field (before transforms).
        # Copied so the base module stays untouched; only this subclass changes.
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        orig_tensor = None
        orig_scale = 1.0
        if self.return_original:
            from torchvision.transforms import functional as _F
            orig_tensor = _F.to_tensor(img)
            H0, W0 = orig_tensor.shape[-2:]
            if max(H0, W0) > self.orig_max_dim:
                s = self.orig_max_dim / max(H0, W0)
                orig_tensor = F_nn.interpolate(
                    orig_tensor[None], size=(int(H0 * s), int(W0 * s)),
                    mode='bilinear')[0]
                orig_scale = s

        if (self.tooth_crop_prob > 0 and not self.return_original
                and __import__('random').random() < self.tooth_crop_prob):
            img, target = self._random_tooth_crop(img, target)

        if 'boxes' in target:
            target['boxes'] = tv_tensors.BoundingBoxes(
                target['boxes'], format='XYXY', canvas_size=img.size[::-1])

        if (self.return_tooth_boxes_t and self._tooth_boxes is not None):
            _fn = self.coco.loadImgs(image_id)[0]['file_name']
            _tb = self._tooth_boxes.get(
                _fn, np.zeros((0, 4), dtype=np.float32)).astype(np.float32)
            target['tooth_boxes_t'] = tv_tensors.BoundingBoxes(
                torch.as_tensor(_tb), format='XYXY', canvas_size=img.size[::-1])

        if (self.return_tooth_mask and self._tooth_masks is not None):
            _fn = self.coco.loadImgs(image_id)[0]['file_name']
            _m = self._tooth_masks.get(_fn)
            H0, W0 = img.size[1], img.size[0]
            if _m is not None and _m.max() > 0:
                _mt = torch.as_tensor(_m, dtype=torch.float32)
                _mt = F_nn.interpolate(_mt[None, None], size=(H0, W0),
                                       mode='nearest')[0, 0]
                _mt = (_mt > 0).to(torch.uint8)
            else:
                _mt = torch.zeros(H0, W0, dtype=torch.uint8)
            target['tooth_mask'] = tv_tensors.Mask(_mt)

        # --- NEW: real per-tooth idmap, aligned through the transforms ---
        if (self.return_tooth_idmap and self._tooth_idmaps is not None):
            _fn = self.coco.loadImgs(image_id)[0]['file_name']
            _im = self._tooth_idmaps.get(_fn)
            H0, W0 = img.size[1], img.size[0]
            if _im is not None and _im.max() > 0:
                _it = torch.as_tensor(_im, dtype=torch.float32)
                _it = F_nn.interpolate(_it[None, None], size=(H0, W0),
                                       mode='nearest')[0, 0]
                _it = _it.to(torch.uint8)
            else:
                _it = torch.zeros(H0, W0, dtype=torch.uint8)
            target['tooth_idmap'] = tv_tensors.Mask(_it)

        if 'masks' in target:
            target['masks'] = tv_tensors.Mask(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        if orig_tensor is not None:
            target['orig_image'] = orig_tensor
        if self._tooth_boxes is not None:
            fn = self.coco.loadImgs(image_id)[0]['file_name']
            boxes = self._tooth_boxes.get(
                fn, np.zeros((0, 4), dtype=np.float32)).astype(np.float32)
            if orig_scale != 1.0:
                boxes = boxes * orig_scale
            target['tooth_boxes'] = boxes

        return img, target
