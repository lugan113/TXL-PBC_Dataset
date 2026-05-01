# TXL-PBC: A Curated and Re-annotated Peripheral Blood Cell Dataset

## Overview

TXL-PBC is a curated and re-annotated peripheral blood cell dataset constructed by integrating four publicly available resources:  
- Blood Cell Count and Detection (BCCD)  
- Blood Cell Detection Dataset (BCDD)  
- Peripheral Blood Cells (PBC)  
- Raabin White Blood Cell (Raabin-WBC)  

The dataset contains 1,260 images and 18,143 bounding box annotations for three major blood cell types:  
- White blood cells (WBC)  
- Red blood cells (RBC)  
- Platelets (PC)  

All images are annotated in YOLO format and split into training, validation, and test sets.

## Directory Structure

```
TXL-PBC/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
├── BCCD_selection.xlsx
├── classes.txt
├── metadata_file.xlsx
└── annotation_protocol.pdf
```

- `images/`: Contains all images, split into train/val/test.
- `labels/`: YOLO-format annotation files, split into train/val/test.
- `data.yaml`: YOLO configuration file.
- `BCCD_selection.xlsx`: List of selected and excluded BCCD images.
- `metadata.csv`: Mapping of image filenames to source datasets.
- `annotation_protocol.pdf`: Manual annotation guidelines.

## Usage

- The dataset can be used directly with object detection frameworks such as YOLO.
- Recommended preprocessing: image normalization and data augmentation.
- Suitable for training, validation, and benchmarking of blood cell detection models.
- Can be combined with other datasets for cross-validation or transfer learning.



## License
This dataset is licensed under the [MIT License](LICENSE).

## Citing the TXL-PBC Dataset

If you use this dataset in your research, please cite our paper:

> Gan, L., Li, X. & Wang, X. A Curated and Re-annotated Peripheral Blood Cell Dataset Integrating Four Public Resources. *Sci Data* **12**, 1694 (2025).  
> https://doi.org/10.1038/s41597-025-05980-z

**BibTeX entry:**
```bibtex
@article{gan2025curated,
  title={A Curated and Re-annotated Peripheral Blood Cell Dataset Integrating Four Public Resources},
  author={Gan, Lu and Li, Xi and Wang, Xichun},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={1694},
  year={2025},
  publisher={Nature Publishing Group UK London}
}


