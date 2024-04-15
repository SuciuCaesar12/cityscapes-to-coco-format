# Cityscapes to COCO format conversion for Object Detection and Panoptic Segmentation
- Utility scripts to convert Cityscapes dataset in COCO format.
- Creates annotations for both COCO Object Detection and Panoptic Segmentation formats.
- Create Object Detection annotations for `stuff` classes as well.
- Take a look [here](https://cocodataset.org/#format-data) for more information on the COCO formats.

## How to use
- Download `leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip` from [Cityscapes](https://www.cityscapes-dataset.com/downloads/)
- You should have the following directory structure:
```
root/
├─ data/
│  ├─ gtFine/
│  ├─ leftImg8bit/
├─ cityscapes2coco.py
├─ subset.py
├─ labels.py
├─ README.md
```

- Run the following command to convert the dataset to COCO format:
```bash
python cityscapes2coco.py --cityscapes_dir data --output_dir data/annotations
```

- The `data` directory should now have the following structure:
```
data/
├─ gtFine/
├─ leftImg8bit/
├─ annotations/
│  ├─ cityscapes_panoptic_test/
│  ├─ cityscapes_panoptic_train/
│  ├─ cityscapes_panoptic_val/
│  ├─ cityscapes_panoptic_test.json
│  ├─ cityscapes_panoptic_train.json
│  ├─ cityscapes_panoptic_val.json
│  ├─ cityscapes_instances_test.json
│  ├─ cityscapes_instances_train.json
│  ├─ cityscapes_instances_val.json

```
- `cityscapes_panoptic_{train, val, test}/` contains the masks with segment ids in COCO format
- `cityscapes_instances_{train, val, test}.json` contains the object detection annotations in COCO format
- `cityscapes_panoptic_{train, val, test}.json` contains the panoptic segmentation annotations in COCO format

## How to create a subset
- Run the following command to create a subset of the dataset:
```bash
python subset.py --cityscapes_dir data --output_dir path/to/subset --num_images 100 --num_images_val 10 --num_images_test 10
```
- This script will create a dataset with the same structure but the images will be randomly sampled from the original dataset