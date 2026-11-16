"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import os
import random

import numpy as np
import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn

from torchvision import tv_tensors

from pycocotools import mask as coco_mask

from src.core import register

__all__ = ['CocoDetection']


@register
class CocoDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks,
                 remap_mscoco_category=False,
                 return_original=False, tooth_boxes_path=None,
                 orig_max_dim=1280,
                 tooth_crop_prob=0.0, tooth_crop_inflate_min=1.6,
                 tooth_crop_inflate_max=2.4,
                 return_tooth_boxes_t=False,
                 return_tooth_mask=False, tooth_masks_path=None,
                 return_tooth_idmap=False, tooth_idmaps_path=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        # SPAZ opt-in: return the original (downscaled) image + tooth boxes
        self.return_original = return_original
        self.orig_max_dim = orig_max_dim
        # Tooth-crop scale augmentation (train-only, opt-in): with prob
        # tooth_crop_prob a random tooth region (overlapping a lesion GT) is
        # cropped+inflated so the detector learns small lesions at high res.
        # Default 0.0 -> existing behavior unchanged.
        self.tooth_crop_prob = tooth_crop_prob
        self.tooth_crop_inflate_min = tooth_crop_inflate_min
        self.tooth_crop_inflate_max = tooth_crop_inflate_max
        # Tooth-anchored density supervision: also return the tooth boxes as a
        # BoundingBoxes field that goes THROUGH the transforms (v2 aligns it
        # with the augmented image). Default off -> existing behavior unchanged.
        self.return_tooth_boxes_t = return_tooth_boxes_t
        # Tooth-mask prior: return the precomputed tooth mask as a Mask field
        # that goes through the transforms (more precise than boxes, no overlap).
        self.return_tooth_mask = return_tooth_mask
        self._tooth_masks = None
        if tooth_masks_path and os.path.exists(tooth_masks_path):
            self._tooth_masks = np.load(tooth_masks_path,
                                        allow_pickle=True).item()
        # Per-tooth instance prior: id 0 is background and ids 1..N identify
        # predicted tooth instances. It is transformed alongside tooth_mask.
        self.return_tooth_idmap = return_tooth_idmap
        self._tooth_idmaps = None
        if tooth_idmaps_path and os.path.exists(tooth_idmaps_path):
            self._tooth_idmaps = np.load(tooth_idmaps_path,
                                         allow_pickle=True).item()
        self._tooth_boxes = None
        if tooth_boxes_path and os.path.exists(tooth_boxes_path):
            self._tooth_boxes = np.load(tooth_boxes_path,
                                        allow_pickle=True).item()

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # SPAZ: capture the ORIGINAL (pre-transform) image, downscaled to
        # orig_max_dim for memory (the zoom crops from this; tooth boxes scaled
        # to match).
        orig_tensor = None
        orig_scale = 1.0
        if self.return_original:
            orig_tensor = F.to_tensor(img)          # [3, H0, W0] in [0,1]
            H0, W0 = orig_tensor.shape[-2:]
            if max(H0, W0) > self.orig_max_dim:
                s = self.orig_max_dim / max(H0, W0)
                orig_tensor = F_nn.interpolate(
                    orig_tensor[None], size=(int(H0 * s), int(W0 * s)),
                    mode='bilinear')[0]
                orig_scale = s

        # Tooth-crop scale augmentation (train-only, opt-in). Skipped when
        # return_original (SPAZ mode) keeps the full-original correspondence.
        if (self.tooth_crop_prob > 0 and not self.return_original
                and random.random() < self.tooth_crop_prob):
            img, target = self._random_tooth_crop(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = tv_tensors.BoundingBoxes(
                target['boxes'],
                format='XYXY',
                canvas_size=img.size[::-1])  # h, w

        # Tooth-anchored density supervision: tooth boxes as a BoundingBoxes
        # field that the v2 transforms flip/resize consistently with the image.
        if (self.return_tooth_boxes_t and self._tooth_boxes is not None):
            _fn = self.coco.loadImgs(image_id)[0]['file_name']
            _tb = self._tooth_boxes.get(
                _fn, np.zeros((0, 4), dtype=np.float32)).astype(np.float32)
            target['tooth_boxes_t'] = tv_tensors.BoundingBoxes(
                torch.as_tensor(_tb), format='XYXY',
                canvas_size=img.size[::-1])

        # Tooth-mask prior: precomputed union tooth mask, upsampled to the
        # original size and carried through the transforms as a Mask (precise,
        # no box-overlap artifacts).
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

        if (self.return_tooth_idmap and self._tooth_idmaps is not None):
            _fn = self.coco.loadImgs(image_id)[0]['file_name']
            _im = self._tooth_idmaps.get(_fn)
            H0, W0 = img.size[1], img.size[0]
            if _im is not None and _im.max() > 0:
                _it = torch.as_tensor(_im, dtype=torch.float32)
                _it = F_nn.interpolate(_it[None, None], size=(H0, W0),
                                       mode='nearest')[0, 0].to(torch.uint8)
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

    def _random_tooth_crop(self, img, target):
        """Train-only: crop + inflate a random tooth region that overlaps a
        lesion GT, resize later to 640 -> the detector sees the tooth (and its
        small lesions) at high resolution. GT boxes are shifted to crop coords;
        a crop ending with no lesion falls back to the full image."""
        if self._tooth_boxes is None or 'boxes' not in target:
            return img, target
        image_id = int(target['image_id'][0]) if torch.is_tensor(target['image_id']) \
            else int(target['image_id'])
        fn = self.coco.loadImgs(image_id)[0]['file_name']
        tb = self._tooth_boxes.get(fn)
        if tb is None or len(tb) == 0:
            return img, target
        gt = target['boxes'].numpy()
        if len(gt) == 0:
            return img, target
        tb = np.asarray(tb, dtype=np.float32)
        # candidate teeth = those overlapping at least one lesion GT box
        cand = []
        for t in tb:
            ox1 = np.maximum(t[0], gt[:, 0]); oy1 = np.maximum(t[1], gt[:, 1])
            ox2 = np.minimum(t[2], gt[:, 2]); oy2 = np.minimum(t[3], gt[:, 3])
            if ((ox2 - ox1 > 0) & (oy2 - oy1 > 0)).any():
                cand.append(t)
        if not cand:
            cand = list(tb)
        t = cand[random.randrange(len(cand))]
        cx = (t[0] + t[2]) / 2; cy = (t[1] + t[3]) / 2
        tw = max(t[2] - t[0], 1.0); th = max(t[3] - t[1], 1.0)
        f = random.uniform(self.tooth_crop_inflate_min, self.tooth_crop_inflate_max)
        cw = f * tw; ch = f * th
        W, H = img.size  # PIL (width, height)
        x0 = int(np.clip(cx - cw / 2, 0, W)); y0 = int(np.clip(cy - ch / 2, 0, H))
        x1 = int(np.clip(cx + cw / 2, 0, W)); y1 = int(np.clip(cy + ch / 2, 0, H))
        if x1 <= x0 or y1 <= y0:
            return img, target
        crop = img.crop((x0, y0, x1, y1))
        shifted = gt - np.array([x0, y0, x0, y0], dtype=np.float32)
        keep = ((shifted[:, 2] > 0) & (shifted[:, 3] > 0)
                & (shifted[:, 0] < (x1 - x0)) & (shifted[:, 1] < (y1 - y0)))
        if not keep.any():
            return img, target   # crop contains no lesion -> keep full image
        target['boxes'] = torch.as_tensor(shifted[keep].clip(min=0),
                                          dtype=torch.float32)
        target['labels'] = target['labels'][keep]
        if 'area' in target:
            target['area'] = target['area'][keep]
        if 'iscrowd' in target:
            target['iscrowd'] = target['iscrowd'][keep]
        if 'masks' in target and target['masks'] is not None \
                and len(target['masks']):
            target['masks'] = target['masks'][keep][:, y0:y1, x0:x1]
        target['size'] = torch.as_tensor([x1 - x0, y1 - y0])
        target['orig_size'] = torch.as_tensor([x1 - x0, y1 - y0])
        return crop, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s 


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
