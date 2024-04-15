import argparse
import labels as cityscapes_labels
import numpy as np
import json
import cv2
from pathlib import Path
from PIL import Image

# Script used to convert cityscapes dataset to COCO format for object detection and panoptic segmentation.
# Both stuff and thing classes are included.
# Can be used for DETR which requires bounding box annotations for both stuff and thing classes in COCO format
# for panoptic segmentation.
############################################################################################################
SPLITS = ['train', 'val', 'test']


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cityscapes_dir', type=str, required=True, help='Path to the Cityscapes dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    return parser.parse_args()


def get_bbox_and_area_from_mask(mask):
    ys, xs = np.nonzero(mask)
    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)
    area = np.sum(mask)
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)], float(area)


def get_bbox_area_from_poly(poly):
    x, y = poly[::2], poly[1::2]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)], float(area)


def main():
    args = read_args()
    ROOT = Path(args.cityscapes_dir)
    OUTPUT_DIR = Path(args.output_dir)
    imageId, annId = 0, 0

    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / ('cityscapes_panoptic_' + split)).mkdir(parents=True, exist_ok=True)
        instance_categories, panoptic_categories = [], []
        instance_annotations, panoptic_annotations = [], []
        instance_images, panoptic_images = [], []

        for label in cityscapes_labels.labels:
            if label.ignoreInEval:
                continue

            instance_categories.append({
                'id': label.trainId,
                'name': label.name,
                'supercategory': label.category,
                'hasInstances': 1 if label.hasInstances else 0,
            })
            panoptic_categories.append({
                'id': label.trainId,
                'name': label.name,
                'supercategory': label.category,
                'isthing': 1 if label.hasInstances else 0,
                'color': [float(c / 255.0) for c in label.color]
            })

        path_gtFine = ROOT / 'gtFine' / split
        files_instanceIds = list(path_gtFine.rglob('*_instanceIds.png'))
        print(f'Processing {len(files_instanceIds)} images in {split} set')

        for step, f_instanceIds in enumerate(files_instanceIds):
            img_instanceIds = Image.open(ROOT / f_instanceIds)

            city = f_instanceIds.parts[-2]
            image_file = f_instanceIds.name.replace('gtFine_instanceIds.png', 'leftImg8bit.png')
            image_dict = {
                'id': imageId,
                'width': img_instanceIds.width,
                'height': img_instanceIds.height,
                'file_name': str(Path('leftImg8bit') / split / city / image_file)
            }
            panoptic_images.append(image_dict)
            instance_images.append(image_dict)

            np_instanceIds = np.array(img_instanceIds)
            pan_format = np.zeros(shape=(np_instanceIds.shape[0], np_instanceIds.shape[1], 3), dtype=np.uint8)
            segmInfo = []

            for instance_id in np.unique(np_instanceIds):
                if instance_id < 1000:
                    semantic_id, is_crowd = instance_id, 1
                else:
                    semantic_id, is_crowd = instance_id // 1000, 0

                label_info = cityscapes_labels.id2label[semantic_id]
                if label_info.ignoreInEval:
                    continue

                mask = np_instanceIds == instance_id
                color = [instance_id % 256, instance_id // 256, instance_id // 256 // 256]
                pan_format[mask] = color

                contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]

                if len(polygons) == 0:
                    continue

                bbox, area = get_bbox_and_area_from_mask(mask)
                segmInfo.append({
                    'id': int(instance_id),
                    'category_id': label_info.trainId,
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': is_crowd
                })
                instance_annotations.append({
                    'id': annId,
                    'image_id': imageId,
                    'category_id': label_info.trainId,
                    'segmentation': polygons,
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': is_crowd
                })
                annId += 1

            pan_file_name = f_instanceIds.name.replace('_instanceIds.png', '_panoptic.png')
            panoptic_annotations.append({
                'image_id': imageId,
                'file_name': pan_file_name,
                'segments_info': segmInfo
            })
            Image.fromarray(pan_format).save(OUTPUT_DIR / ('cityscapes_panoptic_' + split) / pan_file_name)
            imageId += 1

            if step % 100 == 0:
                print(f'Processed {step} images')

        with open(OUTPUT_DIR / f'cityscapes_panoptic_{split}.json', 'w') as f:
            json.dump({
                'images': panoptic_images,
                'annotations': panoptic_annotations,
                'categories': panoptic_categories
            }, f)

        with open(OUTPUT_DIR / f'cityscapes_instances_{split}.json', 'w') as f:
            json.dump({
                'images': instance_images,
                'annotations': instance_annotations,
                'categories': instance_categories
            }, f)


if __name__ == '__main__':
    main()
