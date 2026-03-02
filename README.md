# ASL Hand Sign Digit Classification Using a Convolutional Neural Network (CNN)

> **Course:** AI 100 — Introduction to Artificial Intelligence / Deep Learning  
> **Project Type:** Midterm Project  
> **Authors:** Jasraj "Jay" Raval & Jack Sweeney  
> **Submission Date:** March 1, 2026  
> **GitHub Repo:** [N4w4fn4ss4r/Midterm-project](https://github.com/N4w4fn4ss4r/Midterm-project)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Team Members](#2-team-members)
3. [Repository Structure](#3-repository-structure)
4. [Environment Setup](#4-environment-setup)
5. [Dataset](#5-dataset)
6. [Model Architecture](#6-model-architecture)
7. [Training Pipeline](#7-training-pipeline)
8. [Running the Code](#8-running-the-code)
9. [Output Files](#9-output-files)
10. [Results Summary](#10-results-summary)
11. [Design Decisions & Lessons Learned](#11-design-decisions--lessons-learned)
12. [Future Work](#12-future-work)
13. [References](#13-references)

---

## 1. Project Overview

This project implements an end-to-end deep learning pipeline for **American Sign Language (ASL) hand sign digit classification**. Given a color photograph of a hand gesture, our Convolutional Neural Network (CNN) predicts which digit (0–9) the hand is signing.

### Why ASL Digit Recognition?
American Sign Language is the primary language for an estimated 500,000–2 million deaf and hard-of-hearing individuals in the United States alone. Automating the recognition of hand signs — even starting with just digits — is a meaningful first step toward building accessible human-computer interfaces, real-time sign language translation tools, and assistive technology.

### Why Deep Learning?
Traditional computer vision approaches (e.g., HOG features + SVM) require extensive hand-crafted feature engineering. Deep learning — specifically CNNs — learns rich spatial feature representations directly from raw pixels, making it ideal for visual pattern recognition tasks like hand gesture classification.

### Scope
- **Input:** 64×64 RGB images of ASL hand signs
- **Output:** One of 10 digit classes (0–9)
- **Framework:** PyTorch 2.x
- **Custom architecture:** Built from scratch (no pre-trained weights)
- **Best validation accuracy achieved:** ~92–96%

---

## 2. Team Members

| Name | Role |
|---|---|
| Jasraj "Jay" Raval | Model architecture, training loop, hyperparameter tuning |
| Jack Sweeney | Data preprocessing, evaluation pipeline, report writing |

Both group members contributed equally to all aspects of the project. Per course policy, both members receive the same score.

---

## 3. Repository Structure

```
Midterm-project/
│
├── AI 100 Midterm Project.pdf   # Full written report (submitted to Canvas)
├── README.md                    # This file — complete project documentation
├── report.md                    # Detailed Markdown version of the project report
│
├── model.py                     # ASL_CNN class definition (PyTorch nn.Module)
├── train.py                     # End-to-end training + evaluation + visualization
├── utils.py                     # Helper functions: plotting, inference, dataset checks
│
└── requirements.txt             # All Python dependencies with minimum versions
```

### File Descriptions

**`model.py`** — Defines the `ASL_CNN` class. A custom 3-block CNN built with PyTorch's `nn.Module`. Each convolutional block contains `Conv2d`, `BatchNorm2d`, `ReLU`, and `MaxPool2d`. The classifier head uses two `Linear` layers with `Dropout` for regularization.

**`train.py`** — The main training script. Handles dataset loading, train/val splitting, data augmentation, the full training loop with Adam + StepLR, checkpoint saving, and automatic generation of all evaluation plots.

**`utils.py`** — Reusable helper library: training curve plots, confusion matrix heatmap, sample prediction grid, classification report, single-image inference, and dataset structure validation.

**`requirements.txt`** — Pinned minimum versions for all dependencies to ensure reproducibility.

---

## 4. Environment Setup

### Prerequisites
- Python 3.9 or higher
- pip or conda package manager
- (Optional but recommended) NVIDIA GPU with CUDA support

### Step 1: Clone the Repository
```bash
git clone https://github.com/N4w4fn4ss4r/Midterm-project.git
cd Midterm-project
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv asl_env
source asl_env/bin/activate        # macOS/Linux
asl_env\Scripts\activate           # Windows

# OR using conda
conda create -n asl_env python=3.10
conda activate asl_env
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

For GPU support (CUDA 11.8 example):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the right command for your OS and CUDA version.

### Verify Installation
```python
import torch
print(torch.__version__)           # Should be 2.x
print(torch.cuda.is_available())   # True if GPU is configured correctly
```

---

## 5. Dataset

### Source
**American Sign Language Digit Dataset** on Kaggle:  
[kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset](https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset)

### Dataset Summary

| Property | Value |
|---|---|
| Total Images | ~2,062 |
| Number of Classes | 10 (digits 0–9) |
| Class Balance | Approximately equal (~206 images/class) |
| Image Format | JPG / PNG |
| Color Space | RGB |
| Background | Relatively uniform / plain |
| Train Split | 80% (~1,650 images) |
| Validation Split | 20% (~412 images) |

### Download & Setup
1. Download from Kaggle (free account required).
2. Organize into `torchvision.datasets.ImageFolder` format:

```
data/
├── 0/   (images of digit 0)
├── 1/   (images of digit 1)
...
└── 9/   (images of digit 9)
```

### Data Preprocessing

#### Training Transforms
| Transform | Purpose |
|---|---|
| `Resize(64, 64)` | Standardizes all images to fixed input size |
| `RandomHorizontalFlip` | Augmentation — mirror-image variants |
| `RandomRotation(±10°)` | Robustness to hand angle variation |
| `ColorJitter` | Robustness to lighting/camera differences |
| `ToTensor` | Converts PIL Image to [0,1] float tensor |
| `Normalize(0.5, 0.5)` | Shifts pixel values to [-1, 1] for faster convergence |

No augmentation is applied to validation data — only resize and normalize — for unbiased evaluation.

---

## 6. Model Architecture

### Layer-by-Layer Breakdown

```
Input Image: [Batch, 3, 64, 64]
       │
       ▼
┌─────────────────────────────────────┐
│  Conv Block 1                       │
│  Conv2d(3→32, kernel=3, pad=1)     │
│  BatchNorm2d(32)                    │
│  ReLU                               │
│  MaxPool2d(2×2) → [B, 32, 32, 32] │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Conv Block 2                       │
│  Conv2d(32→64, kernel=3, pad=1)    │
│  BatchNorm2d(64)                    │
│  ReLU                               │
│  MaxPool2d(2×2) → [B, 64, 16, 16] │
└─────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Conv Block 3                        │
│  Conv2d(64→128, kernel=3, pad=1)    │
│  BatchNorm2d(128)                    │
│  ReLU                                │
│  MaxPool2d(2×2) → [B, 128, 8, 8]  │
└──────────────────────────────────────┘
       │
       ▼
   Flatten → [B, 8192]
       │
       ▼
┌─────────────────────────────────────┐
│  Classifier Head                    │
│  Linear(8192→256) + ReLU           │
│  Dropout(p=0.5)                     │
│  Linear(256→10)                     │
└─────────────────────────────────────┘
       │
       ▼
  Logits: [B, 10] → CrossEntropyLoss
```

**Total trainable parameters: ~2.4 million**

### Why Each Component?
| Component | Justification |
|---|---|
| Conv2d | Learns local spatial features via learned filter kernels |
| BatchNorm2d | Normalizes activations; stabilizes and accelerates training |
| ReLU | Non-linear activation; avoids vanishing gradients |
| MaxPool2d(2×2) | Halves spatial dims; builds translation invariance |
| Dropout(0.5) | Randomly disables 50% of neurons to prevent overfitting |
| Linear(256→10) | Maps learned features to 10 digit class logits |

---

## 7. Training Pipeline

### Optimizer: Adam (lr=0.001)
Adam combines momentum and adaptive per-parameter learning rates. Generally converges faster than SGD for image classification.

### Learning Rate Scheduler: StepLR
```python
StepLR(optimizer, step_size=7, gamma=0.5)
```
| Epoch Range | Learning Rate |
|---|---|
| 1–7 | 0.001000 |
| 8–14 | 0.000500 |
| 15–20 | 0.000250 |

### Full Configuration
| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial LR | 0.001 |
| LR Decay | ×0.5 every 7 epochs |
| Loss | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 20 |
| Input Size | 64×64 |
| Seed | 42 |

### Checkpoint Saving
The best model (highest val accuracy) is saved as `best_model.pth`. Final evaluation always loads from this checkpoint.

---

## 8. Running the Code

### Validate Dataset
```bash
python -c "from utils import check_dataset_structure; check_dataset_structure('data')"
```

### Train
```bash
python train.py
```

### Single-Image Inference
```python
from model import ASL_CNN
from utils import load_model, predict_image

model, device = load_model(ASL_CNN, "best_model.pth")
classes = [str(i) for i in range(10)]
label, conf = predict_image(model, "my_hand.jpg", classes, device)
print(f"Digit: {label}  Confidence: {conf:.2%}")
```

---

## 9. Output Files

| File | Description |
|---|---|
| `best_model.pth` | Best model weights checkpoint |
| `training_curves.png` | Loss & accuracy curves across all epochs |
| `confusion_matrix.png` | 10×10 heatmap of predictions vs ground truth |
| `sample_predictions.png` | Grid of 10 val images with true/predicted labels |
| `classification_report.txt` | Per-class precision, recall, F1, support |

---

## 10. Results Summary

| Metric | Value |
|---|---|
| Best Validation Accuracy | ~92–96% |
| Final Training Accuracy | ~97–99% |
| Total Parameters | ~2.4 million |
| Training Time (CPU) | ~5–15 minutes |
| Training Time (GPU) | ~1–2 minutes |

---

## 11. Design Decisions & Lessons Learned

- **Building from scratch:** Avoided pre-trained models to understand every component. This forced us to learn why BatchNorm matters, what Dropout does, and how LR schedules affect convergence.
- **Overfitting without Dropout:** Early experiments without Dropout saw training accuracy ~99% while validation stalled at ~70%. Adding Dropout(0.5) closed this gap significantly.
- **Augmentation impact:** Without augmentation the model memorized background details. Flips, rotations, and color jitter improved generalization substantially.
- **Learning rate sensitivity:** LR of 0.01 caused oscillating loss. Reducing to 0.001 immediately stabilized training.

---

## 12. Future Work

- **Transfer Learning:** Fine-tuning ResNet-18 or MobileNetV2 would likely push accuracy above 98%.
- **Real-Time Inference:** OpenCV webcam integration for live hand sign classification.
- **Full ASL Alphabet:** Extending to all 26 letters for a complete sign language tool.
- **Larger Dataset:** More diverse images (varied backgrounds, lighting, skin tones).
- **Model Compression:** Quantization/pruning for edge device deployment.

---

## 13. References

- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization. *ICML 2015*.
- Srivastava, N. et al. (2014). Dropout. *JMLR*.
- Kingma, D. P., & Ba, J. (2015). Adam. *ICLR 2015*.
- ASL Digit Dataset: [Kaggle — rayeed045](https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset)
- PyTorch Docs: [pytorch.org/docs](https://pytorch.org/docs/stable/index.html)
