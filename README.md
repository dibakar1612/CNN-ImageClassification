# CNN Image Classification with PyTorch — Geometric Shapes

## Student & Course Information

| Item | Details |
|---|---|
| Student Name | Dibakar Sarker Akash |
| Student ID | 220125 |
| Registration No. | 2201024 |
| Session | 2022–2023 |
| Department | Computer Science and Engineering |
| Institution | Jashore University of Science and Technology (JUST) |
| Course | CSE-3208: Pattern Recognition & Machine Learning Lab |
| Assignment | Assignment 8 — CNN Shape Classification |
| Dataset Option | Option 5 — Geometric Shapes |
| Classes | Circles, Squares, Triangles |

## Project Overview

This project implements a complete PyTorch CNN pipeline for classifying Circles, Squares, and Triangles.

Repository: https://github.com/dibakar1612/CNN-ImageClassification

Notebook: `Final_Lab/CNN/220125_CNN.ipynb`

## Dataset

```text
training/
├── train/
│   ├── circles/
│   ├── squares/
│   └── triangles/
├── validation/
│   ├── circles/
│   ├── squares/
│   └── triangles/
└── test/
    ├── circles/
    ├── squares/
    └── triangles/
```

| Split | Circles | Squares | Triangles | Total |
|---|---:|---:|---:|---:|
| Training | 304 | 103 | 136 | 543 |
| Validation | 60 | 60 | 60 | 180 |
| Test | 30 | 30 | 30 | 90 |

The training split is class-imbalanced, so the current notebook uses `WeightedRandomSampler`.

## Preprocessing

Training uses 64×64 RGB images with tensor conversion and normalization (`mean=0.5`, `std=0.5`). Training additionally uses random resized crops, rotation, affine transforms, and brightness/contrast jitter. Validation and test use deterministic resizing, tensor conversion, and normalization.

## Current CNN Architecture

```text
Input: 3 × 64 × 64 RGB
        ↓
Conv2d(3 → 32, 3×3)
BatchNorm → ReLU → MaxPool
        ↓
Conv2d(32 → 64, 3×3)
BatchNorm → ReLU → MaxPool
        ↓
Conv2d(64 → 128, 3×3)
BatchNorm → ReLU → MaxPool
        ↓
AdaptiveAvgPool2d(1×1)
        ↓
Flatten
        ↓
Linear(128 → 64) → ReLU → Dropout(0.30)
        ↓
Linear(64 → 3)
        ↓
Circle / Square / Triangle
```

**Total parameters: 102,147**

## Training Configuration

- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW`
- Learning rate: `0.001`
- Weight decay: `1e-4`
- Epochs: `30`
- Batch size: `32`
- Class balancing: `WeightedRandomSampler`
- Scheduler: `ReduceLROnPlateau`
- Random seed: `42`

The trained state dictionary must be saved as:

```text
model/220125.pth
```

## Custom Smartphone Images

The project uses 10 real-world smartphone photographs:

- 3 circles
- 3 squares
- 4 triangles

They are stored under `dataset/` and are processed to the same 64×64 RGB format before inference.

## Evaluation Outputs

The notebook generates:

```text
results/
├── AvE.png
├── LvE.png
├── confusion_matrix.png
├── custom_predictions.png
└── error_analysis.png
```

- `AvE.png`: training and validation accuracy
- `LvE.png`: training and validation loss
- `confusion_matrix.png`: standard test-set confusion matrix
- `custom_predictions.png`: 10-image prediction gallery with confidence
- `error_analysis.png`: up to 3 randomly selected misclassified test samples

## Results

**Update these values after the current 30-epoch model finishes training.**

| Metric | Current Run |
|---|---:|
| Best Validation Accuracy | Pending |
| Test Accuracy | Pending |
| Test Loss | Pending |
| Custom Smartphone Accuracy | Pending |

The previous 42.22% test / 30.00% custom results belong to the old model and should **not** be copied into the results for this new architecture.

## Repository Structure

```text
CNN-ImageClassification/
├── Final_Lab/
│   └── CNN/
│       └── 220125_CNN.ipynb
├── dataset/
│   └── 10 custom smartphone images
├── model/
│   └── 220125.pth
├── results/
│   ├── AvE.png
│   ├── LvE.png
│   ├── confusion_matrix.png
│   ├── custom_predictions.png
│   └── error_analysis.png
└── training/
    ├── train/
    ├── validation/
    └── test/
```

## Reproducibility

The notebook automatically resolves or clones the GitHub repository, loads the dataset, trains the CNN, evaluates the standard test set, and performs custom-image inference without requiring manual dataset uploads during runtime.

**Important:** `model/220125.pth` must be generated from the current 102,147-parameter architecture after training. The old 2,117,059-parameter checkpoint is incompatible with this model.
