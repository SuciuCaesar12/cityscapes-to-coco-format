import argparse
import json
import shutil
import numpy as np
from pathlib import Path


# Script used to take a subset of the cityscapes dataset in COCO format.
SPLITS = ['train', 'val', 'test']


def read_args():
    parser = argparse.ArgumentParser(description='Take a subset of the cityscapes dataset in COCO format.')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to take from train set.')
    parser.add_argument('--num_images_val', type=int, default=10, help='Number of images to take from val set.')
    parser.add_argument('--num_images_test', type=int, default=10, help='Number of images to take from test set.')
    parser.add_argument('--cityscapes_dir', type=str, required=True, help='Path to the Cityscapes dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    return parser.parse_args()


def main():
    args = read_args()
    INPUT_DIR = Path(args.cityscapes_dir)
    OUTPUT_DIR = Path(args.output_dir)
    N_SPLITS = [args.num_images, args.num_images_val, args.num_images_test]

    for split in SPLITS:
        (OUTPUT_DIR / 'annotations' / f'cityscapes_panoptic_{split}').mkdir(parents=True, exist_ok=True)

    for split, n_images in zip(SPLITS, N_SPLITS):
        with open(INPUT_DIR / 'annotations' / f'cityscapes_instances_{split}.json', 'r') as f:
            instances = json.load(f)
            images = instances['images']
            annotations = instances['annotations']

            indices = np.random.choice(len(images), n_images, replace=False)
            instance_images = [images[i] for i in indices]
            for image in instance_images:
                dst = OUTPUT_DIR / Path(image['file_name']).parent
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src=INPUT_DIR / image['file_name'], dst=dst)

            instance_annotations = []
            for image in instance_images:
                instance_annotations.extend([ann for ann in annotations if ann['image_id'] == image['id']])

            with open(OUTPUT_DIR / "annotations" / f'cityscapes_instances_{split}.json', 'w') as f:
                json.dump(
                    {'images': instance_images,
                     'annotations': instance_annotations,
                     'categories': instances['categories']
                     }, f)

        with open(INPUT_DIR / 'annotations' / f'cityscapes_panoptic_{split}.json', 'r') as f:
            panoptic = json.load(f)
            images = panoptic['images']
            annotations = panoptic['annotations']

            indices = np.random.choice(len(images), n_images, replace=False)
            panoptic_images = [images[i] for i in indices]
            for image in panoptic_images:
                dst = OUTPUT_DIR / Path(image['file_name']).parent
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src=Path(args.cityscapes_dir) / image['file_name'], dst=dst)

            panoptic_annotations = []
            for image in panoptic_images:
                ann = next(ann for ann in annotations if ann['image_id'] == image['id'])
                src = INPUT_DIR / "annotations" / f'cityscapes_panoptic_{split}' / ann['file_name']
                dst = OUTPUT_DIR / "annotations" / f'cityscapes_panoptic_{split}' / ann['file_name']
                shutil.copy2(src=src, dst=dst)
                panoptic_annotations.append(ann)

            with open(OUTPUT_DIR / "annotations" / f'cityscapes_panoptic_{split}.json', 'w') as f:
                json.dump(
                    {'images': panoptic_images,
                     'annotations': panoptic_annotations,
                     'categories': panoptic['categories']
                     }, f)


if __name__ == '__main__':
    main()
