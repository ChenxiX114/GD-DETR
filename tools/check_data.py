"""Check local OralXrays-9 image, annotation and precomputed-prior paths."""

import argparse
import json
from pathlib import Path

import numpy as np


def check_split(root, prior_dir, split):
    images_dir = root / 'images' / split
    annotation = root / 'annotations' / f'instances_{split}.json'
    if not annotation.is_file() or not images_dir.is_dir():
        raise FileNotFoundError(f'Missing annotation or image directory: {annotation}, {images_dir}')
    with annotation.open(encoding='utf-8') as stream:
        images = json.load(stream)['images']
    names = [item['file_name'] for item in images]
    if len(names) != len(set(names)):
        raise ValueError(f'{split}: duplicate image names in annotations')
    missing_images = [name for name in names if not (images_dir / name).is_file()]
    unsupported = [name for name in names if Path(name).name != name or Path(name).suffix not in ('.jpg', '.png')]
    if missing_images or unsupported:
        raise ValueError(f'{split}: {len(missing_images)} missing images, {len(unsupported)} names unsupported by prior generator; examples: {(missing_images + unsupported)[:5]}')
    print(f'{split}: {len(names)} annotated images found')
    required = set(names)
    for name in (f'tooth_boxes_{split}.npy', f'tooth_masks_{split}.npy', f'tooth_idmaps_{split}_u8.npy'):
        path = prior_dir / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing prior: {path}')
        # These files are trusted local dictionaries saved by the prior generator.
        prior = np.load(path, allow_pickle=True).item()
        missing = required - set(prior)
        if missing:
            raise ValueError(f'{path}: {len(missing)} image keys missing; examples: {sorted(missing)[:5]}')
        print(f'  {path}: {len(prior)} entries')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('dataset/OralXrays9'))
    parser.add_argument('--priors', type=Path, default=Path('data/tooth_priors'))
    args = parser.parse_args()
    for split in ('train', 'val'):
        check_split(args.dataset, args.priors, split)
    print('Paths and image-key coverage passed; mask geometry and experiment provenance are not checked.')


if __name__ == '__main__':
    main()
