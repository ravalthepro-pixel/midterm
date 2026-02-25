# ASL Hand Sign Digit Classification
### AI 100 — Deep Learning Midterm Project

A Convolutional Neural Network (CNN) trained to classify American Sign Language (ASL) hand sign digits (0–9) from images.

---

## 📁 Repository Structure

```
├── train.py                    # Main training script
├── report.docx                 # Project report (PDF submitted separately)
├── training_curves.png         # Loss & accuracy plots (generated after training)
├── confusion_matrix.png        # Confusion matrix (generated after training)
├── sample_predictions.png      # Visual predictions (generated after training)
├── classification_report.txt   # Precision/Recall/F1 per class
└── README.md
```

---

## 🗂️ Dataset

**American Sign Language Digit Dataset**  
Download from Kaggle: https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset

After downloading, extract so the folder structure looks like:
```
asl_digits/
  0/   ← images of the ASL sign for digit 0
  1/
  2/
  ...
  9/
```

Place the `asl_digits/` folder in the same directory as `train.py`.

---

## ⚙️ Requirements

Install dependencies with:

```bash
pip install torch torchvision matplotlib scikit-learn seaborn Pillow
```

> **GPU (optional):** If you have an NVIDIA GPU with CUDA, PyTorch will automatically use it. Otherwise it runs on CPU (slower but works fine).

---

## 🚀 Running the Code

```bash
python train.py
```

This will:
1. Load and preprocess the dataset (80% train / 20% validation split)
2. Train the CNN for 20 epochs
3. Save the best model weights to `best_model.pth`
4. Generate and save all result plots and reports

---

## 🧠 Model Architecture

A custom 3-block CNN:

| Layer | Output Size |
|-------|-------------|
| Input image | 64 × 64 × 3 |
| Conv Block 1 (32 filters) + MaxPool | 32 × 32 × 32 |
| Conv Block 2 (64 filters) + MaxPool | 16 × 16 × 64 |
| Conv Block 3 (128 filters) + MaxPool | 8 × 8 × 128 |
| Flatten → FC(256) → Dropout(0.5) | 256 |
| Output FC(10) + Softmax | 10 classes |

---

## 📊 Results

Results are generated automatically after running `train.py`:
- `training_curves.png` — Train/Val loss and accuracy over epochs
- `confusion_matrix.png` — Per-class confusion matrix heatmap
- `sample_predictions.png` — Sample images with true vs. predicted labels
- `classification_report.txt` — Precision, Recall, F1-score per digit class
