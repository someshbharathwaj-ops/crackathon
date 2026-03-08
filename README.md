
# Road Damage Detection using YOLOv8

This repository contains the implementation of an object detection system designed to identify different types of road damage using the YOLOv8 framework.  
The project was developed as part of the **Crackathon road damage detection challenge**.

The objective is to automatically detect and classify road surface defects from images. Automating this process helps improve road maintenance workflows by reducing the need for manual inspection.

---

## Problem Statement

Manual road inspection is slow, labor-intensive, and often inconsistent. The goal of this project is to build a computer vision system capable of detecting and classifying road damage directly from images.

The model predicts:

- The **location** of each damage instance (bounding box)
- The **type of damage** present

The supported damage classes are:

- Longitudinal Crack  
- Transverse Crack  
- Alligator Crack  
- Other Corruption  
- Pothole  

---

## Dataset

The dataset follows the **YOLO annotation format**.

Directory structure:

```

data/
├── images/
│   ├── train
│   ├── val
│   └── test
└── labels/
├── train
└── val

```

Each image has a corresponding `.txt` label file containing:

```

<class_id> <x_center> <y_center> <width> <height>

```

All coordinates are normalized relative to the image size.

---

## Model

The project uses **YOLOv8** from the Ultralytics framework.

Key characteristics:

- Single-stage object detection architecture
- Anchor-free detection head
- Pretrained backbone (transfer learning)
- Optimized for real-time detection tasks

Final training configuration:

| Parameter | Value |
|----------|------|
| Model | YOLOv8-L |
| Image Size | 1024 |
| Epochs | 150 |
| Batch Size | 4 |
| Optimizer | AdamW |
| Framework | Ultralytics |

---

## Repository Structure

```

crackathon/
│
├── configs/
│   └── rdd.yaml
│
├── data/
│   ├── images/
│   └── labels/
│
├── experiments/
│
├── submissions/
│
├── utils/
│
├── train.py
├── infer.py
└── README.md

```

---

## Training

The training script initializes the YOLO model, loads the dataset configuration, and trains the network using transfer learning.

Run training:

```

python train.py

```

The trained weights will be saved to:

```

/project_folder/RDD_V3_CHAMPION/weights/best.pt

```

---

## Inference

Inference generates prediction files for the test images in the required competition format.

Run inference:

```

python infer.py

```

Output prediction labels will be stored in:

```

submissions/labels/

```

Each output file contains:

```

<class_id> <x_center> <y_center> <width> <height> <confidence>

```

---

## Evaluation Metric

Model performance is evaluated using **Mean Average Precision (mAP)**.

Two metrics are commonly reported:

- **mAP@0.5** – detection accuracy with IoU threshold of 0.5
- **mAP@0.5:0.95** – stricter metric averaged over multiple IoU thresholds

The competition ranking is based on the **mAP score on a hidden test set**.

---

## Approach

The following strategies were used to improve detection performance:

- Transfer learning from pretrained YOLOv8 weights
- High-resolution training (1024px) for detecting small cracks
- Data augmentation (mosaic, mixup, copy-paste)
- Extended training schedule
- Balanced batch configuration for GPU memory limits

---

## Output Format

The submission consists of prediction files structured as follows:

```

predictions/
├── image_001.txt
├── image_002.txt
└── ...

```

Each line represents a detected object.

---

## Dependencies

Install required packages:

```

pip install ultralytics
pip install opencv-python
pip install numpy

```

---

## Notes

- Training was performed using **GPU acceleration on Kaggle**.
- The project focuses on **object detection**, not segmentation.
- The pipeline is designed to be reproducible and easily extensible.

---

## Author

Somesh Bharathwaj


