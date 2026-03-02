"""
================================================================================
ASL Hand Sign Digit Classification — Utility Functions
================================================================================
Course  : AI 100 — Deep Learning Midterm Project
Authors : Jasraj "Jay" Raval & Jack Sweeney
Date    : March 1, 2026
GitHub  : https://github.com/N4w4fn4ss4r/Midterm-project
--------------------------------------------------------------------------------

This module provides reusable helper functions for:
    1. Visualization  — training curves, confusion matrix, sample predictions
    2. Evaluation     — classification report generation
    3. Inference      — single-image and batch prediction pipeline
    4. Dataset checks — validate ImageFolder structure before training

All visualization functions save files to disk and print a confirmation message.
They do not display plots interactively (plt.show is not called), so this module
works correctly in headless environments (servers, Colab, VS Code, terminal).

Usage
-----
    from utils import (
        plot_training_curves,
        plot_confusion_matrix,
        plot_sample_predictions,
        save_classification_report,
        load_model,
        predict_image,
        check_dataset_structure,
    )
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os

# ── Third Party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")    # Non-interactive backend: safe for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
from torchvision import transforms


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    train_losses: list,
    val_losses:   list,
    train_accs:   list,
    val_accs:     list,
    save_path:    str = "training_curves.png",
) -> None:
    """
    Plot and save training and validation loss + accuracy curves.

    Creates a side-by-side figure with two subplots:
        Left:  Loss curves (train and val) across epochs
        Right: Accuracy curves (train and val) across epochs

    A healthy training run shows:
        - Both train and val loss decreasing over time
        - Train and val accuracy increasing, with a small stable gap
        - No sharp validation loss spike (which would indicate overfitting)

    Args:
        train_losses (list[float]): Per-epoch training loss values.
        val_losses   (list[float]): Per-epoch validation loss values.
        train_accs   (list[float]): Per-epoch training accuracy values [0, 1].
        val_accs     (list[float]): Per-epoch validation accuracy values [0, 1].
        save_path    (str)        : Output file path. Default: "training_curves.png".

    Returns:
        None. Saves figure to save_path and prints confirmation.

    Example:
        >>> plot_training_curves([1.2, 0.8, 0.5], [1.3, 0.9, 0.7],
        ...                      [0.4, 0.7, 0.85], [0.35, 0.65, 0.80])
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "ASL CNN Training Curves\n"
        "Authors: Jasraj \"Jay\" Raval & Jack Sweeney | AI 100 Midterm",
        fontsize=12, y=1.02
    )

    # ── Loss subplot ─────────────────────────────────────────────────────────
    ax1.plot(epochs, train_losses, label="Train Loss",
             color="#2563EB", linewidth=2, marker="o", markersize=4)
    ax1.plot(epochs, val_losses, label="Val Loss",
             color="#DC2626", linewidth=2, marker="s", markersize=4,
             linestyle="--")
    ax1.set_title("Loss Curves", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(1, len(train_losses))

    # Annotate final values
    ax1.annotate(f"{train_losses[-1]:.3f}",
                 (len(train_losses), train_losses[-1]),
                 textcoords="offset points", xytext=(8, 0), fontsize=9,
                 color="#2563EB")
    ax1.annotate(f"{val_losses[-1]:.3f}",
                 (len(val_losses), val_losses[-1]),
                 textcoords="offset points", xytext=(8, 0), fontsize=9,
                 color="#DC2626")

    # ── Accuracy subplot ──────────────────────────────────────────────────────
    ax2.plot(epochs, [a * 100 for a in train_accs], label="Train Acc",
             color="#2563EB", linewidth=2, marker="o", markersize=4)
    ax2.plot(epochs, [a * 100 for a in val_accs], label="Val Acc",
             color="#DC2626", linewidth=2, marker="s", markersize=4,
             linestyle="--")
    ax2.set_title("Accuracy Curves", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xlim(1, len(train_accs))

    # Annotate best val accuracy
    best_val_epoch = int(np.argmax(val_accs)) + 1
    best_val       = max(val_accs) * 100
    ax2.axvline(x=best_val_epoch, color="gray", linestyle=":", alpha=0.6)
    ax2.annotate(f"Best: {best_val:.1f}%\n(epoch {best_val_epoch})",
                 (best_val_epoch, best_val),
                 textcoords="offset points", xytext=(8, -20), fontsize=9,
                 color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_confusion_matrix(
    all_labels: list,
    all_preds:  list,
    classes:    list,
    save_path:  str = "confusion_matrix.png",
) -> None:
    """
    Plot and save a confusion matrix heatmap using Seaborn.

    The confusion matrix is a 10×10 grid:
        - Rows    = true class labels
        - Columns = predicted class labels
        - Diagonal cells = correct predictions (want these to be high)
        - Off-diagonal   = misclassifications (want these to be 0)

    Color intensity encodes the count in each cell. Annotation shows the
    exact count inside each cell for precise interpretation.

    Interpretation guidance:
        - A strong, bright diagonal = high accuracy across all classes.
        - Off-diagonal clusters reveal which digit pairs are confused most often,
          typically those with similar hand shapes in ASL.

    Args:
        all_labels (list[int]): Ground truth class indices from validation set.
        all_preds  (list[int]): Predicted class indices from the model.
        classes    (list[str]): Class name strings, e.g. ['0', '1', ..., '9'].
        save_path  (str)      : Output file path. Default: "confusion_matrix.png".

    Returns:
        None. Saves figure to save_path and prints confirmation.
    """
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_title(
        "Confusion Matrix — ASL CNN\n"
        "Authors: Jasraj \"Jay\" Raval & Jack Sweeney | AI 100 Midterm",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.tick_params(axis="both", labelsize=11)

    # Overall accuracy annotation
    total_correct = np.trace(cm)
    overall_acc   = total_correct / cm.sum()
    ax.text(
        0.98, 0.01, f"Overall Accuracy: {overall_acc:.2%}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_sample_predictions(
    images:     torch.Tensor,
    all_labels: list,
    all_preds:  list,
    classes:    list,
    n:          int = 10,
    save_path:  str = "sample_predictions.png",
) -> None:
    """
    Plot and save a grid of sample validation images with predicted labels.

    Shows n randomly selected validation images side by side. Each image
    displays:
        - The image itself (denormalized to [0, 1] for display)
        - True label (e.g., "True: 4")
        - Predicted label (e.g., "Pred: 4")
        - Green title = correct prediction
        - Red title   = incorrect prediction

    This provides an intuitive, human-readable quality check of the model's
    behavior on real validation images.

    Args:
        images     (Tensor)    : Val images, shape [N, 3, H, W], normalized to [-1,1].
        all_labels (list[int]) : Ground truth class indices.
        all_preds  (list[int]) : Predicted class indices.
        classes    (list[str]) : Class name strings.
        n          (int)       : Number of images to display. Default: 10.
        save_path  (str)       : Output file path. Default: "sample_predictions.png".

    Returns:
        None. Saves figure to save_path and prints confirmation.
    """
    n = min(n, len(all_preds))
    indices = np.random.choice(len(all_preds), n, replace=False)

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    fig.suptitle(
        "Sample Validation Predictions — ASL CNN\n"
        "Green = Correct  |  Red = Incorrect\n"
        "Authors: Jasraj \"Jay\" Raval & Jack Sweeney | AI 100 Midterm",
        fontsize=11, y=1.01
    )

    axes_flat = axes.flatten() if rows > 1 else [axes] * cols

    for i, idx in enumerate(indices):
        ax = axes_flat[i]

        # Denormalize: [-1, 1] → [0, 1] for display
        img = images[idx].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)

        ax.imshow(img)
        true_label = classes[all_labels[idx]]
        pred_label = classes[all_preds[idx]]
        correct    = (true_label == pred_label)
        color      = "#16A34A" if correct else "#DC2626"   # green or red

        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}",
            color=color, fontsize=10, fontweight="bold"
        )
        ax.axis("off")

        # Add colored border to frame
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    # Hide unused subplots
    for j in range(len(indices), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def save_classification_report(
    all_labels: list,
    all_preds:  list,
    classes:    list,
    save_path:  str = "classification_report.txt",
) -> str:
    """
    Generate, print, and save a per-class classification report.

    The report (from sklearn.metrics.classification_report) includes:
        - Precision: Of all images predicted as class X, what fraction were X?
        - Recall:    Of all images that are class X, what fraction did we find?
        - F1-Score:  Harmonic mean of precision and recall.
        - Support:   Number of true instances of each class in val set.

    High F1-scores across all classes confirm consistent per-class performance,
    not just high average accuracy.

    Args:
        all_labels (list[int]): Ground truth class indices.
        all_preds  (list[int]): Predicted class indices.
        classes    (list[str]): Class name strings.
        save_path  (str)      : Output file path. Default: "classification_report.txt".

    Returns:
        report (str): The full classification report as a string.
    """
    report = classification_report(all_labels, all_preds, target_names=classes,
                                   digits=4)

    header = (
        "=" * 60 + "\n"
        "ASL CNN — Per-Class Classification Report\n"
        "Authors: Jasraj \"Jay\" Raval & Jack Sweeney\n"
        "AI 100 — Deep Learning Midterm Project\n"
        "=" * 60 + "\n\n"
    )

    full_report = header + report

    print("\nClassification Report:")
    print(report)

    with open(save_path, "w") as f:
        f.write(full_report)

    print(f"  [Saved] {save_path}")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_class,
    weights_path: str,
    num_classes:  int = 10,
    device:       torch.device = None,
):
    """
    Load a trained ASL_CNN model from a saved .pth weights file.

    Args:
        model_class  : The model class (e.g., ASL_CNN). Not instantiated yet.
        weights_path (str)         : Path to the .pth checkpoint file.
        num_classes  (int)         : Number of output classes. Default: 10.
        device       (torch.device): Target device. If None, auto-detects GPU.

    Returns:
        model  : Loaded model in eval() mode, moved to device.
        device : The torch.device used.

    Example:
        >>> from model import ASL_CNN
        >>> from utils import load_model
        >>> model, device = load_model(ASL_CNN, "best_model.pth")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class(num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  Loaded model from: {weights_path}  (device: {device})")
    return model, device


def predict_image(
    model,
    image_path:  str,
    classes:     list,
    device:      torch.device,
    img_size:    int = 64,
) -> tuple:
    """
    Run inference on a single image file and return the predicted class + confidence.

    Applies the same preprocessing as the validation transform:
        Resize(64×64) → ToTensor → Normalize(0.5, 0.5)

    Args:
        model      : Trained ASL_CNN model in eval() mode.
        image_path (str)         : Path to image file (JPG, PNG, etc.)
        classes    (list[str])   : List of class name strings.
        device     (torch.device): Device to run inference on.
        img_size   (int)         : Expected input size. Default: 64.

    Returns:
        label      (str)  : Predicted class name (e.g., "4").
        confidence (float): Softmax probability of the predicted class [0, 1].

    Example:
        >>> label, conf = predict_image(model, "test_hand.jpg", classes, device)
        >>> print(f"Predicted: {label}  Confidence: {conf:.2%}")
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)   # [1, 3, H, W]

    with torch.no_grad():
        logits      = model(tensor)
        probs       = torch.softmax(logits, dim=1)
        conf, idx   = probs.max(dim=1)

    label      = classes[idx.item()]
    confidence = conf.item()
    return label, confidence


def predict_batch(
    model,
    image_paths: list,
    classes:     list,
    device:      torch.device,
    img_size:    int = 64,
) -> list:
    """
    Run inference on a list of image paths and return predictions for each.

    More efficient than calling predict_image() in a loop because images
    are processed as a single batched tensor.

    Args:
        model        : Trained ASL_CNN model in eval() mode.
        image_paths  (list[str])   : List of image file paths.
        classes      (list[str])   : List of class name strings.
        device       (torch.device): Device to run inference on.
        img_size     (int)         : Expected input size. Default: 64.

    Returns:
        results (list[dict]): List of dicts with keys:
            'path'       (str)   : Original image path.
            'label'      (str)   : Predicted class name.
            'confidence' (float) : Softmax probability of predicted class.

    Example:
        >>> paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        >>> results = predict_batch(model, paths, classes, device)
        >>> for r in results:
        ...     print(f"{r['path']}: {r['label']} ({r['confidence']:.2%})")
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    tensors = torch.stack([
        transform(Image.open(p).convert("RGB")) for p in image_paths
    ]).to(device)  # [N, 3, H, W]

    with torch.no_grad():
        logits = model(tensors)
        probs  = torch.softmax(logits, dim=1)
        confs, indices = probs.max(dim=1)

    results = [
        {
            "path":       image_paths[i],
            "label":      classes[indices[i].item()],
            "confidence": confs[i].item(),
        }
        for i in range(len(image_paths))
    ]
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DATASET VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def check_dataset_structure(
    data_dir:         str,
    expected_classes: int = 10,
) -> bool:
    """
    Verify that data_dir has the correct ImageFolder structure for training.

    torchvision.datasets.ImageFolder requires the following layout:
        data_dir/
        ├── class_0/  (images)
        ├── class_1/  (images)
        ...
        └── class_N/  (images)

    This function checks:
        1. data_dir exists and is a directory
        2. It contains exactly expected_classes subdirectories
        3. Each subdirectory contains at least one image file

    Args:
        data_dir         (str): Root directory of the dataset.
        expected_classes (int): Expected number of class folders. Default: 10.

    Returns:
        bool: True if structure is valid, False otherwise.

    Side effects:
        Prints a summary of found folders and image counts.

    Example:
        >>> check_dataset_structure("data", expected_classes=10)
    """
    print(f"\nDataset structure check: '{data_dir}/'")
    print("-" * 40)

    if not os.path.isdir(data_dir):
        print(f"  ERROR: Directory '{data_dir}' not found.")
        print(f"  Create it and place class subfolders inside.")
        return False

    # Find all subdirectories (class folders)
    subdirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and not d.startswith(".")   # ignore hidden folders
    ])

    if len(subdirs) == 0:
        print(f"  ERROR: No subdirectories found in '{data_dir}'.")
        return False

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    total_images     = 0
    class_counts     = {}
    all_valid        = True

    for sub in subdirs:
        folder_path = os.path.join(data_dir, sub)
        images      = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        count = len(images)
        class_counts[sub] = count
        total_images += count

        status = "OK" if count > 0 else "EMPTY"
        if count == 0:
            all_valid = False
        print(f"  [{status}] Class '{sub}': {count:>4} images")

    print("-" * 40)
    print(f"  Total images  : {total_images}")
    print(f"  Classes found : {len(subdirs)} (expected {expected_classes})")

    if len(subdirs) != expected_classes:
        print(f"\n  WARNING: Expected {expected_classes} class folders, "
              f"found {len(subdirs)}.")
        all_valid = False

    if all_valid:
        print(f"\n  Dataset structure OK.")
    else:
        print(f"\n  Dataset structure has issues. Please fix before training.")

    return all_valid and (len(subdirs) == expected_classes)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing utils.py...")
    print("All utility functions loaded successfully.")
    print("Run 'python train.py' to start training.")
