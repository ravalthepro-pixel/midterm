"""
================================================================================
ASL Hand Sign Digit Classification — Training Script
================================================================================
Course  : AI 100 — Deep Learning Midterm Project
Authors : Jasraj "Jay" Raval & Jack Sweeney
Date    : March 1, 2026
GitHub  : https://github.com/N4w4fn4ss4r/Midterm-project
--------------------------------------------------------------------------------

What this script does
---------------------
1. Loads the ASL Digit Dataset using torchvision.datasets.ImageFolder.
2. Splits into 80% training / 20% validation (fixed seed for reproducibility).
3. Applies separate transforms: augmentation for train, resize+normalize for val.
4. Instantiates the ASL_CNN model and prints trainable parameter count.
5. Trains for 20 epochs with Adam optimizer and StepLR scheduler.
6. Saves the best model checkpoint (by validation accuracy) to best_model.pth.
7. After training, loads the best checkpoint and runs final evaluation.
8. Generates and saves:
    - training_curves.png     (loss + accuracy curves)
    - confusion_matrix.png    (10×10 heatmap)
    - sample_predictions.png  (10 sample val images with true/pred labels)
    - classification_report.txt (per-class precision/recall/F1)

Dataset Folder Structure Required
----------------------------------
    data/
    ├── 0/   *.jpg / *.png
    ├── 1/   *.jpg / *.png
    ...
    └── 9/   *.jpg / *.png

Download from:
    https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset

Run
---
    python train.py

Requirements
------------
    pip install -r requirements.txt
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import time
import random

# ── Third Party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ── Local Modules ─────────────────────────────────────────────────────────────
from model import ASL_CNN
from utils import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_sample_predictions,
    save_classification_report,
    check_dataset_structure,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR      = "data"           # Root folder with one subfolder per class
IMG_SIZE      = 64               # Resize all images to IMG_SIZE × IMG_SIZE
BATCH_SIZE    = 32               # Number of images per mini-batch
EPOCHS        = 20               # Total training epochs
LR            = 0.001            # Initial learning rate for Adam
LR_STEP_SIZE  = 7                # Decay LR every N epochs
LR_GAMMA      = 0.5              # Multiply LR by this factor at each step
VAL_SPLIT     = 0.2              # Fraction of data reserved for validation
SEED          = 42               # Random seed for reproducibility
CHECKPOINT    = "best_model.pth" # Filename to save best model weights
NUM_WORKERS   = 2                # DataLoader worker processes (0 on Windows)


# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# Set all random seeds so the train/val split and weight initialization are
# the same every time the script is run, enabling fair comparison across runs.
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE DETECTION
# Use GPU (CUDA) if available, otherwise fall back to CPU.
# Training on CPU takes 5–15 minutes; GPU takes ~1–2 minutes.
# ══════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print(f"  ASL CNN Training — Jasraj Raval & Jack Sweeney")
print("=" * 60)
print(f"  Device     : {device}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA ver   : {torch.version.cuda}")
print(f"  PyTorch    : {torch.__version__}")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET VALIDATION
# Verify the expected folder structure before attempting to load data.
# This prevents cryptic errors from a mis-organized dataset.
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/6] Validating dataset structure...")
if not check_dataset_structure(DATA_DIR, expected_classes=10):
    raise RuntimeError(
        f"\nDataset validation failed. Please ensure '{DATA_DIR}/' contains "
        "10 subdirectories named '0' through '9', each containing image files.\n"
        "Download from: https://www.kaggle.com/datasets/rayeed045/"
        "american-sign-language-digit-dataset"
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

# Training transforms include augmentation to improve generalization.
# Augmentation creates slightly different versions of each image on each epoch,
# effectively increasing the dataset size and preventing overfitting.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Augmentation: random horizontal flip (p=0.5)
    # Many ASL digit signs are meaningful when mirrored, so flipping is safe.
    transforms.RandomHorizontalFlip(p=0.5),

    # Augmentation: random rotation up to ±10 degrees
    # Accounts for variation in hand tilt across different images/photographers.
    transforms.RandomRotation(degrees=10),

    # Augmentation: random color jitter
    # Accounts for variation in lighting, camera white balance, and exposure.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    # Convert PIL Image → PyTorch Tensor with shape [3, H, W], values in [0, 1]
    transforms.ToTensor(),

    # Normalize per-channel to [-1, 1] using mean=0.5, std=0.5.
    # This centers the input distribution around zero, which helps gradient
    # descent converge faster and more stably.
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Validation transforms: NO augmentation.
# We evaluate on clean, unmodified images to get an unbiased accuracy estimate.
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING AND SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/6] Loading and splitting dataset...")

# Load full dataset once to get total size and generate the index split.
# We do NOT use this instance directly for training — we create two separate
# instances (with different transforms) and use index subsets.
_full = datasets.ImageFolder(root=DATA_DIR)
n_total = len(_full)
n_val   = int(n_total * VAL_SPLIT)
n_train = n_total - n_val

# Generate a reproducible random permutation of all indices, then split.
all_indices = list(range(n_total))
rng = np.random.default_rng(seed=SEED)
rng.shuffle(all_indices)
train_indices = all_indices[:n_train]
val_indices   = all_indices[n_train:]

# Create two ImageFolder instances — one per transform — and apply index subsets.
train_dataset = Subset(
    datasets.ImageFolder(root=DATA_DIR, transform=train_transform),
    train_indices
)
val_dataset = Subset(
    datasets.ImageFolder(root=DATA_DIR, transform=val_transform),
    val_indices
)

# DataLoaders handle batching, shuffling, and parallel data loading.
# shuffle=True for training ensures different batch compositions each epoch.
# shuffle=False for validation (order doesn't matter; we want deterministic eval).
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False,
)

classes = _full.classes   # e.g., ['0', '1', ..., '9']
print(f"  Classes    : {classes}")
print(f"  Total imgs : {n_total}")
print(f"  Train      : {n_train} ({100*(1-VAL_SPLIT):.0f}%)")
print(f"  Validation : {n_val} ({100*VAL_SPLIT:.0f}%)")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Train iters: {len(train_loader)} batches/epoch")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL, LOSS, OPTIMIZER, SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3/6] Building model...")

model = ASL_CNN(num_classes=len(classes)).to(device)
n_params = model.count_parameters()
print(f"  Architecture: ASL_CNN (custom 3-block CNN)")
print(f"  Parameters  : {n_params:,}")

# CrossEntropyLoss combines LogSoftmax + NLLLoss.
# It expects raw logits (not softmax output) and class indices as targets.
criterion = nn.CrossEntropyLoss()

# Adam optimizer: adaptive per-parameter learning rates + momentum.
# Generally converges faster than SGD for image classification.
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# StepLR: multiply LR by gamma every step_size epochs.
# Allows aggressive early updates and fine-grained tuning near convergence.
scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

print(f"  Optimizer   : Adam (lr={LR}, weight_decay=1e-4)")
print(f"  Scheduler   : StepLR (step={LR_STEP_SIZE}, gamma={LR_GAMMA})")
print(f"  Loss        : CrossEntropyLoss")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[4/6] Training for {EPOCHS} epochs...")
print("-" * 70)
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
      f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
print("-" * 70)

# Track metrics for plotting
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_acc  = 0.0
best_epoch    = 0
start_time    = time.time()

for epoch in range(1, EPOCHS + 1):

    # ── Training Phase ────────────────────────────────────────────────────────
    model.train()   # Enables Dropout and BatchNorm in training mode
    running_loss = 0.0
    correct      = 0
    total        = 0

    for batch_imgs, batch_labels in train_loader:
        batch_imgs   = batch_imgs.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        # Zero gradients from the previous step.
        # Gradients accumulate by default in PyTorch; must be cleared each step.
        optimizer.zero_grad()

        # Forward pass: compute predicted logits
        logits = model(batch_imgs)

        # Compute loss between predictions and ground truth labels
        loss = criterion(logits, batch_labels)

        # Backward pass: compute gradients via backpropagation
        loss.backward()

        # Gradient clipping: prevents exploding gradients (optional but safe)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model weights using computed gradients
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item() * batch_imgs.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == batch_labels).sum().item()
        total        += batch_labels.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total

    # ── Validation Phase ──────────────────────────────────────────────────────
    model.eval()    # Disables Dropout; uses running stats for BatchNorm
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():   # Disable gradient computation for efficiency
        for batch_imgs, batch_labels in val_loader:
            batch_imgs   = batch_imgs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            logits = model(batch_imgs)
            loss   = criterion(logits, batch_labels)

            running_loss += loss.item() * batch_imgs.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == batch_labels).sum().item()
            total        += batch_labels.size(0)

    val_loss = running_loss / total
    val_acc  = correct / total

    # Step the LR scheduler after each epoch
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # Record metrics
    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc);    val_accs.append(val_acc)

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        torch.save(model.state_dict(), CHECKPOINT)
        checkpoint_marker = " ← best"
    else:
        checkpoint_marker = ""

    print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | "
          f"{val_loss:>8.4f} | {val_acc:>7.4f} | {current_lr:>8.6f}"
          f"{checkpoint_marker}")

elapsed = time.time() - start_time
print("-" * 70)
print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
print(f"Best checkpoint saved to: {CHECKPOINT}")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION — Load Best Checkpoint
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/6] Evaluating best checkpoint ({CHECKPOINT})...")
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

all_preds  = []
all_labels = []
all_images = []

with torch.no_grad():
    for batch_imgs, batch_labels in val_loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        preds = model(batch_imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.numpy())
        all_images.append(batch_imgs.cpu())

all_images = torch.cat(all_images, dim=0)   # [N_val, 3, 64, 64]


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE OUTPUT FILES
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/6] Generating evaluation outputs...")

plot_training_curves(train_losses, val_losses, train_accs, val_accs)
plot_confusion_matrix(all_labels, all_preds, classes)
plot_sample_predictions(all_images, all_labels, all_preds, classes, n=10)
save_classification_report(all_labels, all_preds, classes)

print("\n" + "=" * 60)
print("  All done! Output files:")
print("    best_model.pth")
print("    training_curves.png")
print("    confusion_matrix.png")
print("    sample_predictions.png")
print("    classification_report.txt")
print("=" * 60)
