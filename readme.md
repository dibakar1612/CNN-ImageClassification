# CNN Image Classification with PyTorch — Geometric Shapes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dibakar1612/CNN-ImageClassification/blob/main/Final_Lab/CNN/220125_CNN.ipynb)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-0.20.1-orange)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Student & Course Information

| Parameter | Details |
| :--- | :--- |
| **Student Name** | **Dibakar Sarker Akash** |
| **Student ID** | **220125** |
| **Registration No** | **2201024** |
| **Session** | **2022--2023** |
| **Department** | **Department of Computer Science and Engineering** |
| **Institution** | **Jashore University of Science and Technology (JUST)** |
| **Course** | **CSE-3208: Pattern Recognition \& Machine Learning Lab** |
| **Assignment** | **Assignment 8: Convolutional Neural Networks for Shape Classification** |
| **Dataset Option** | **Option 5 — Geometric Shapes (Circles, Squares, Triangles)** |
| **Target Branch** | **`main`** |

---

## 📖 Project Overview

This repository contains an automated Deep Learning pipeline developed in **PyTorch** for 3-class geometric shape classification (**Circles**, **Squares**, and **Triangles**). 

The pipeline includes:
1. **Automated Git Clone & Colab Path Handling:** Zero manual file upload required; click **"Run All"** in Google Colab to execute the complete workflow.
2. **Single Shared Image Preprocessing:** Unified $64 \times 64$ resizing, tensor conversion, and symmetric affine normalization ($\mu=0.5, \sigma=0.5 \implies [-1.0, 1.0]$) across training, validation, testing, and custom phone inference.
3. **Deep CNN Architecture:** 2-stage hierarchical feature extractor ($3 \times 3$ Convolutions + ReLU + $2 \times 2$ Max-Pooling) with a dense classification head ($2,117,059$ trainable parameters).
4. **Model Serialization:** Checkpoint state dictionary saved to `model/220125.pth` and auto-loaded if available.
5. **Multi-Faceted Evaluation:** Confusion matrix heatmap, training history curves, class-wise precision/recall/F1 metrics, and visual misclassification error analysis.
6. **Real-World Smartphone Generalization:** Evaluated on 10 physical photographs captured with a **Samsung Galaxy A12 (SM-A127F)** camera with forensic EXIF metadata.
7. **Comprehensive 44-Page Documentation:** Includes the complete LaTeX source and compiled PDF report (`Final_Lab/220125_CNN_Report.pdf`) matching the official 37-section standard template.

---

## 🚀 Quick Start in Google Colab (1-Click Execution)

Click the badge below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dibakar1612/CNN-ImageClassification/blob/main/Final_Lab/CNN/220125_CNN.ipynb)

In Google Colab, navigate to **Runtime $\to$ Run all** (`Ctrl + F9`). The notebook will:
- Automatically clone this repository.
- Train the CNN for 10 epochs (or load pretrained weights `220125.pth`).
- Evaluate on the 90 unseen test images and generate evaluation plots.
- Preprocess and run inference on the 10 custom phone images.
- Print predictions, confidence scores, and display the prediction gallery.

---

## 🧠 Model Architecture

```text
Input Image: (3, 64, 64) RGB
  │
  ▼
Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  ──> (32, 64, 64)
  │
  ▼
ReLU() + MaxPool2d(kernel_size=2, stride=2)                        ──> (32, 32, 32)
  │
  ▼
Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) ──> (64, 32, 32)
  │
  ▼
ReLU() + MaxPool2d(kernel_size=2, stride=2)                        ──> (64, 16, 16)
  │
  ▼
Flatten()                                                         ──> (16,384,)
  │
  ▼
Linear(in_features=16384, out_features=128) + ReLU()              ──> (128,)
  │
  ▼
Linear(in_features=128, out_features=3)                           ──> Logits: (3,)
```

### Parameter Calculation

$$\text{Conv1: } (3 \times 3 \times 3 \times 32) + 32 = 896$$
$$\text{Conv2: } (32 \times 3 \times 3 \times 64) + 64 = 18,496$$
$$\text{FC1: } (16,384 \times 128) + 128 = 2,097,280$$
$$\text{FC2: } (128 \times 3) + 3 = 387$$
$$\mathbf{\text{Total Trainable Parameters: }} 896 + 18,496 + 2,097,280 + 387 = \mathbf{2,117,059}$$

---

## 📊 Dataset Distribution & Results

### 1. Dataset Partitioning

| Split | Circles | Squares | Triangles | Total Images |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 304 ($56.0\%$) | 103 ($19.0\%$) | 136 ($25.0\%$) | **543** |
| **Validation Set** | 60 ($33.3\%$) | 60 ($33.3\%$) | 60 ($33.3\%$) | **180** |
| **Test Set (Unseen)** | 30 ($33.3\%$) | 30 ($33.3\%$) | 30 ($33.3\%$) | **90** |
| **Custom Mobile Photos** | 3 | 3 | 4 | **10** |

### 2. Quantitative Performance Summary

| Metric | Training (Epoch 10) | Validation (Epoch 10) | Test Set (90 Images) | Custom Phone (10 Photos) |
| :--- | :---: | :---: | :---: | :---: |
| **Loss** | $0.7354$ | $0.7891$ | $1.2578$ | --- |
| **Accuracy** | **$66.11\%$** | **$62.22\%$** | **$42.22\%$** (38/90) | **$30.00\%$** (3/10) |

### 3. Class-Wise Performance (Standard Test Set)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Circles** | $0.4737$ | **$0.9000$** ($27/30$) | **$0.6207$** | 30 |
| **Squares** | $0.2500$ | $0.1333$ ($4/30$) | $0.1739$ | 30 |
| **Triangles** | $0.4118$ | $0.2333$ ($7/30$) | $0.2979$ | 30 |
| **Macro Avg** | $0.3785$ | $0.4222$ | $0.3642$ | 90 |

---

## 📱 Custom Smartphone Image Predictions (Samsung Galaxy A12)

| Image File | Actual Class | Predicted Class | Confidence | Result |
| :--- | :--- | :--- | :---: | :---: |
| `Circle 1.jpeg` | Circle | **Circle** | $100.0\%$ | **Correct** |
| `Circle 2.jpeg` | Circle | **Circle** | $100.0\%$ | **Correct** |
| `Circle 3.jpeg` | Circle | **Circle** | $100.0\%$ | **Correct** |
| `Square 3.jpeg` | Square | Circle | $100.0\%$ | Misclassified |
| `square_1.png` | Square | Circle | $99.98\%$ | Misclassified |
| `square_2.png` | Square | Circle | $99.99\%$ | Misclassified |
| `Triangle 1.jpeg` | Triangle | Circle | $100.0\%$ | Misclassified |
| `Triangle 2.jpeg` | Triangle | Circle | $100.0\%$ | Misclassified |
| `Triangle 3.jpeg` | Triangle | Circle | $100.0\%$ | Misclassified |
| `Triangle 4.jpeg` | Triangle | Circle | $100.0\%$ | Misclassified |

---

## 📁 Repository Directory Hierarchy

```text
CNN-ImageClassification/
|
|-- README.md                      # Primary Repository Documentation
|-- .gitignore                     # Git ignore rules
\-- Final_Lab/
    |-- Assignment_8_CNN.pdf       # Assignment Specification
    |-- 220125_links.tex           # LaTeX source for 1-Page Summary
    |-- 220125_links.pdf           # Compiled 1-Page Submission Links Summary
    |-- 220125_CNN_Report.tex      # Full 44-Page LaTeX Report Source
    |-- 220125_CNN_Report.pdf      # Compiled 44-Page Comprehensive Report PDF
    \-- CNN/
        |-- 220125_CNN.ipynb       # Fully Automated PyTorch Notebook
        |-- readme.md              # Module Documentation
        |-- dataset/               # 10 Custom Smartphone Test Images
        |-- model/
        |   \-- 220125.pth         # Saved Model Checkpoint (2.1M parameters)
        |-- results/
        |   |-- AvE.png            # Accuracy vs Epochs Curve
        |   |-- LvE.png            # Loss vs Epochs Curve
        |   |-- confusion_matrix.png # Test Confusion Matrix Heatmap
        |   |-- custom_predictions.png # 10-Image Prediction Gallery
        |   \-- error_analysis.png # Misclassified Test Samples
        \-- training/              # Standard Dataset Splits (train/val/test)
```

---

## 📄 Submission Links

- **GitHub Repository:** [https://github.com/dibakar1612/CNN-ImageClassification](https://github.com/dibakar1612/CNN-ImageClassification)
- **Final Lab Directory:** [https://github.com/dibakar1612/CNN-ImageClassification/tree/main/Final_Lab](https://github.com/dibakar1612/CNN-ImageClassification/tree/main/Final_Lab)
- **Interactive Google Colab Notebook:** [Open in Colab](https://colab.research.google.com/github/dibakar1612/CNN-ImageClassification/blob/main/Final_Lab/CNN/220125_CNN.ipynb)
- **Comprehensive Report PDF:** [`Final_Lab/220125_CNN_Report.pdf`](Final_Lab/220125_CNN_Report.pdf)
- **1-Page Links Summary PDF:** [`Final_Lab/220125_links.pdf`](Final_Lab/220125_links.pdf)
