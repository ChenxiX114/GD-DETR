"""
离线预计算牙齿 YOLO-seg 先验 (conf=0.5)
=====================================
把冻结牙齿分割器跑一遍指定图像目录，并为每张图存储：
  - tooth_boxes_*.npy: 牙框（原图坐标）
  - tooth_masks_*.npy: 全部牙齿的 union mask（256x256）
  - tooth_idmaps_*_u8.npy: 每颗牙的 instance id map（256x256；背景=0）

id map 和 union mask 从同一组实例掩码构造。实例按照 YOLO 输出顺序
编码为 1..N；若实例重叠，后一个实例覆盖前一个实例，确保 mask == idmap>0。
用法(CPU 后台跑, 不占 GPU):
    python tools/precompute_tooth_boxes.py \
        --img-dir <train images> --yolo <tooth_seg.pt> -o <save_dir>
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', required=True, help='Directory containing input images')
    parser.add_argument('--yolo', required=True, help='Path to a trained tooth instance-segmentation checkpoint')
    parser.add_argument('-o', '--output', required=True, help='Directory for the generated prior files')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--name', type=str, default='tooth_boxes_train.npy')
    args = parser.parse_args()
    if os.path.basename(args.name) != args.name or not args.name.startswith('tooth_boxes') or not args.name.endswith('.npy'):
        parser.error('--name must be a file name starting with tooth_boxes and ending with .npy')
    os.makedirs(args.output, exist_ok=True)

    from ultralytics import YOLO
    yolo = YOLO(args.yolo)

    imgs = sorted(glob.glob(os.path.join(args.img_dir, '*.jpg')))
    imgs += sorted(glob.glob(os.path.join(args.img_dir, '*.png')))
    print(f'{len(imgs)} 张训练图', flush=True)

    result = {}
    masks = {}
    idmaps = {}
    MASKS_SIZE = 256
    for i, p in enumerate(imgs):
        name = os.path.basename(p)
        img = cv2.imread(p)
        if img is None:
            continue
        r = yolo(img, imgsz=640, conf=args.conf, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None \
            else np.zeros((0, 4))
        result[name] = boxes.astype(np.float32)
        # Per-tooth instance ids and their union mask, both computed on the
        # same 256x256 grid so the two priors stay exactly aligned.
        m = np.zeros((1, 1), dtype=np.uint8)
        idmap = np.zeros((1, 1), dtype=np.uint8)
        if r.masks is not None and r.masks.data is not None:
            instances = (r.masks.data.cpu().numpy() > 0).astype(np.uint8)
            if len(instances) > np.iinfo(np.uint8).max:
                raise RuntimeError(f'{name}: {len(instances)} instances exceed uint8 idmap capacity')
            if len(instances) > 0:
                idmap = np.zeros((MASKS_SIZE, MASKS_SIZE), dtype=np.uint8)
                for tooth_id, instance in enumerate(instances, start=1):
                    instance = cv2.resize(instance, (MASKS_SIZE, MASKS_SIZE),
                                          interpolation=cv2.INTER_NEAREST)
                    idmap[instance > 0] = tooth_id
                # A very small instance can vanish after downsampling. Remap
                # surviving labels to consecutive ids (1..N), matching the
                # OralXrays9 idmap convention.
                present_ids = np.unique(idmap)
                present_ids = present_ids[present_ids > 0]
                remap = np.zeros(256, dtype=np.uint8)
                remap[present_ids] = np.arange(1, len(present_ids) + 1,
                                                dtype=np.uint8)
                idmap = remap[idmap]
                m = (idmap > 0).astype(np.uint8)
        masks[name] = m
        idmaps[name] = idmap
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(imgs)} 图, 平均牙数 {np.mean([len(v) for v in result.values()]):.1f}',
                  flush=True)

    out_path = os.path.join(args.output, args.name)
    np.save(out_path, np.array(result, dtype=object))
    masks_path = os.path.join(args.output,
                              args.name.replace('tooth_boxes', 'tooth_masks'))
    np.save(masks_path, np.array(masks, dtype=object))
    idmaps_name = args.name.replace('tooth_boxes', 'tooth_idmaps')
    stem, ext = os.path.splitext(idmaps_name)
    idmaps_path = os.path.join(args.output, f'{stem}_u8{ext}')
    np.save(idmaps_path, np.array(idmaps, dtype=object))
    print(f'saved {out_path}: {len(result)} 图')
    print(f'saved {masks_path}: {len(masks)} 图掩码 ({MASKS_SIZE}x{MASKS_SIZE})')
    print(f'saved {idmaps_path}: {len(idmaps)} 图 per-tooth idmap ({MASKS_SIZE}x{MASKS_SIZE}, uint8)')
    print(f'平均每图牙数: {np.mean([len(v) for v in result.values()]):.1f}')


if __name__ == '__main__':
    main()
