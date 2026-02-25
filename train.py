"""
ASL Hand Sign Digit Classification (0-9)
Deep Learning Midterm Project
AI 100 - Group Project

Dataset: ASL Digit Dataset
  Download from: https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset
  After downloading, extract so your folder structure looks like:
    asl_digits/
      0/  (images of sign for digit 0)
      1/
      ...
      9/

Usage:
  python train.py

Requirements:
  pip install torch torchvision matplotlib scikit-learn seaborn Pillow
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for VS Code
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "asl_digits"   # folder with subfolders 0-9
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
LR          = 0.001
IMG_SIZE    = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [str(i) for i in range(10)]

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2. DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load full dataset then split 80/20
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
n_total  = len(full_dataset)
n_train  = int(0.8 * n_total)
n_val    = n_total - n_train
train_set, val_set = random_split(full_dataset, [n_train, n_val])

# Apply val transforms to validation split
val_set.dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transforms)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Dataset: {n_total} images | Train: {n_train} | Val: {n_val}")

# ─────────────────────────────────────────────
# 3. CNN MODEL DEFINITION
# ─────────────────────────────────────────────
class ASL_CNN(nn.Module):
    """
    A simple 3-block CNN for 10-class classification.
    
    Architecture:
      Block 1: Conv(3→32) -> BN -> ReLU -> MaxPool
      Block 2: Conv(32→64) -> BN -> ReLU -> MaxPool
      Block 3: Conv(64→128) -> BN -> ReLU -> MaxPool
      Classifier: Flatten -> FC(2048->256) -> Dropout -> FC(256->10)
    """
    def __init__(self, num_classes=10):
        super(ASL_CNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 64 -> 32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32 -> 16

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = ASL_CNN(num_classes=10).to(DEVICE)
print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 4. TRAINING SETUP
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

print("\n" + "="*60)
print("Starting Training...")
print("="*60)

best_val_acc = 0.0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # ── Train ──
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    train_loss = running_loss / total
    train_acc  = 100.0 * correct / total

    # ── Validate ──
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    val_loss = running_loss / total
    val_acc  = 100.0 * correct / total

    scheduler.step()

    train_losses.append(train_loss);  val_losses.append(val_loss)
    train_accs.append(train_acc);     val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.1f}%  |  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.1f}%")

elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed:.0f}s  |  Best Val Acc: {best_val_acc:.1f}%")

# ─────────────────────────────────────────────
# 6. PLOT: LOSS & ACCURACY CURVES
# ─────────────────────────────────────────────
epochs_range = range(1, NUM_EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ASL Digit CNN — Training Results", fontsize=14, fontweight='bold')

ax1.plot(epochs_range, train_losses, 'b-o', label='Train Loss', markersize=4)
ax1.plot(epochs_range, val_losses,   'r-o', label='Val Loss',   markersize=4)
ax1.set_title("Loss over Epochs");  ax1.set_xlabel("Epoch");  ax1.set_ylabel("Loss")
ax1.legend();  ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, train_accs, 'b-o', label='Train Acc', markersize=4)
ax2.plot(epochs_range, val_accs,   'r-o', label='Val Acc',   markersize=4)
ax2.set_title("Accuracy over Epochs");  ax2.set_xlabel("Epoch");  ax2.set_ylabel("Accuracy (%)")
ax2.legend();  ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_curves.png")

# ─────────────────────────────────────────────
# 7. CONFUSION MATRIX
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Confusion Matrix — ASL Digit CNN", fontsize=13, fontweight='bold')
ax.set_xlabel("Predicted Label");  ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# ─────────────────────────────────────────────
# 8. CLASSIFICATION REPORT
# ─────────────────────────────────────────────
report = classification_report(all_labels, all_preds,
                                target_names=[f"Digit {c}" for c in CLASS_NAMES])
print("\nClassification Report:")
print(report)

with open("classification_report.txt", "w") as f:
    f.write("ASL Digit CNN — Classification Report\n")
    f.write("="*45 + "\n\n")
    f.write(report)
print("Saved: classification_report.txt")

# ─────────────────────────────────────────────
# 9. SAMPLE PREDICTIONS VISUALIZATION
# ─────────────────────────────────────────────
model.eval()
images_shown, labels_shown, preds_shown = [], [], []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(DEVICE))
        _, preds = outputs.max(1)
        images_shown.extend(images[:5])
        labels_shown.extend(labels[:5].numpy())
        preds_shown.extend(preds.cpu()[:5].numpy())
        if len(images_shown) >= 10:
            break

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Sample Predictions (Green = Correct, Red = Wrong)", fontsize=12, fontweight='bold')
mean = np.array([0.5, 0.5, 0.5]);  std = np.array([0.5, 0.5, 0.5])

for idx, ax in enumerate(axes.flatten()):
    img = images_shown[idx].numpy().transpose(1, 2, 0)
    img = np.clip(std * img + mean, 0, 1)
    ax.imshow(img)
    true_lbl = CLASS_NAMES[labels_shown[idx]]
    pred_lbl = CLASS_NAMES[preds_shown[idx]]
    color = "green" if true_lbl == pred_lbl else "red"
    ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", color=color, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: sample_predictions.png")

print("\nAll done! Files created:")
print("  best_model.pth          — saved model weights")
print("  training_curves.png     — loss & accuracy plots")
print("  confusion_matrix.png    — per-class confusion matrix")
print("  sample_predictions.png  — visual sample predictions")
print("  classification_report.txt — precision/recall/F1 per class")
