# TXL-PBC Dataset

## Overview

The TXL-PBC Dataset is a comprehensive collection of re-annotated and integrated cell images from multiple cell datasets. It is specifically designed for evaluating various object detection models, especially those that use the YOLO format.
 TXL-PBC dataset is divided into a training set (train: 1008), a validation set (val: 288), and a test set (test: 144).

## Contents

- **images/**: Contains subfolders for training, testing, and validation images.
- **labels/**: Contains subfolders for the corresponding YOLO format annotation files.
- **data.yaml**: Configuration file for the YOLO dataset.

## Dataset Structure

The dataset is organized as follows:
TXL-PBC-Dataset/
├── images/
│ ├── train/
│ │ ├── img1.jpg
│ │ ├── img2.jpg
│ │ └── ...
│ ├── test/
│ │ ├── img1.jpg
│ │ ├── img2.jpg
│ │ └── ...
│ └── val/
│ ├── img1.jpg
│ ├── img2.jpg
│ └── ...
├── labels/
│ ├── train/
│ │ ├── img1.txt
│ │ ├── img2.txt
│ │ └── ...
│ ├── test/
│ │ ├── img1.txt
│ │ ├── img2.txt
│ │ └── ...
│ └── val/
│ ├── img1.txt
│ ├── img2.txt
│ └── ...
├── data.yaml
└── README.md

- `images/`: Contains train, test, and val subfolders with the respective images.
- `labels/`: Contains train, test, and val subfolders with the respective YOLO format annotation files.
- `data.yaml`: Contains dataset configuration for YOLO.
## Dataset Structure
This dataset is licensed under the MIT License.

