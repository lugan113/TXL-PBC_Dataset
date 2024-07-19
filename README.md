# TXL-PBC Dataset

## Overview

The TXL-PBC Dataset is a comprehensive collection of re-annotated and integrated cell images from multiple cell datasets. The main objective of this study is to perform sample reduction, re-labeling, and integration from [BCCD](https://github.com/Shenggan/BCCD_Dataset) and [BCD datasets](https://www.kaggle.com/datasets/adhoppin/blood-celldetection-datatset). Then, the original dataset is integrated with two new cell datasets, PBC dataset [Peripheral Blood Cells](https://pubmed.ncbi.nlm.nih.gov/32346559/) and Raabin-WBC dataset [Raabin White Blood Cells](https://raabindata.com/raabin-health-database/), to create a high-quality, sample balanced new dataset. We call it TXL-PBC dataset. We use the [Labelimg](https://github.com/HumanSignal/labelImg) tool Semi-automated labeling is performed using [YOLOv8n](https://github.com/ultralytics/ultralytics).to annotate all the datasets. It is specifically designed for evaluating various object detection models, especially those that use the YOLO format.


## Contents
 TXL-PBC dataset is divided into a training set (train: 1008), a validation set (val: 288), and a test set (test: 144).
You can see a example of the labeled cell image.
[TXL-PBC Dataset Example](example.png)
We have three kind of labels :

'RBC' (Red Blood Cell)
'WBC' (White Blood Cell)
'Platelets'

## Dataset Structure

The dataset is organized as follows:
```
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
│ ├── classes.txt
├── data.yaml
└── README.md
```


- images/: Contains train, test, and val subfolders with the respective images.
- labels/: Contains train, test, val, and classes subfolders with the respective YOLO format annotation files.
- data.yaml: Contains dataset configuration for YOLO.

## License
This dataset is licensed under the [MIT License](LICENSE).

## Citing TXL-PBC Dataset
If you're using this dataset, please cite: Lu Gan, Xi Li [TXL-PBC: a freely accessible labeled peripheral blood cell dataset](https://arxiv.org/abs/2407.13214) 	arXiv:2407.13214.


