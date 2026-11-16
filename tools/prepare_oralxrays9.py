"""Create a deterministic train/val COCO dataset from YOLO image/label folders.

This creates a new split; it has not been verified to recover the paper's split.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

CLASSES = ['Apical Periodontitis', 'Decay', 'Wisdom Tooth', 'Missing Tooth',
           'Dental Filling', 'Root Canal Filling', 'Implant',
           'Porcelain Crown', 'Ceramic Bridge']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New or empty output directory')
    parser.add_argument('--train-count', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.images.is_dir() or not args.labels.is_dir():
        parser.error('image and label directories must exist')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if len({p.stem for p in images}) != len(images):
        parser.error('image basenames must be unique')
    missing = [p.name for p in images if not (args.labels / (p.stem + '.txt')).is_file()]
    if missing:
        parser.error(f'{len(missing)} images lack a label file; examples: {missing[:5]}')
    if not 0 < args.train_count < len(images):
        parser.error('train-count must be positive and smaller than the number of images')
    random.Random(args.seed).shuffle(images)
    groups = {'train': images[:args.train_count], 'val': images[args.train_count:]}
    for split, paths in groups.items():
        dst = args.output / 'images' / split
        dst.mkdir(parents=True, exist_ok=True)
        coco = {'images': [], 'annotations': [], 'categories': [
            {'id': i, 'name': name, 'supercategory': 'oral_anomaly'}
            for i, name in enumerate(CLASSES)]}
        for image_id, image in enumerate(paths, 1):
            with Image.open(image) as im:
                width, height = im.size
            # Normalize extensions for the prior-generation script.
            filename = image.stem + image.suffix.lower()
            if image.suffix.lower() == '.jpeg':
                filename = image.stem + '.jpg'
            shutil.copy2(image, dst / filename)
            coco['images'].append({'id': image_id, 'file_name': filename,
                                   'width': width, 'height': height})
            for line_no, line in enumerate((args.labels / (image.stem + '.txt')).read_text(encoding='utf-8').splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'{image.stem}.txt:{line_no}: expected class cx cy width height')
                cls = int(fields[0])
                cx, cy, bw, bh = map(float, fields[1:])
                if cls not in range(len(CLASSES)) or not all(0 <= x <= 1 for x in (cx, cy, bw, bh)) or bw == 0 or bh == 0:
                    raise ValueError(f'{image.stem}.txt:{line_no}: invalid class or normalized box')
                x, y, w, h = (cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height
                coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': image_id,
                    'category_id': cls, 'bbox': [x, y, w, h], 'area': w * h, 'iscrowd': 0})
        ann = args.output / 'annotations'
        ann.mkdir(parents=True, exist_ok=True)
        (ann / f'instances_{split}.json').write_text(json.dumps(coco), encoding='utf-8')
        print(f'{split}: {len(paths)} images, {len(coco["annotations"])} annotations')
    print('This is a new deterministic split, not a verified reconstruction of the paper partition.')


if __name__ == '__main__':
    main()
