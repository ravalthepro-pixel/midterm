"""
================================================================================
ASL Hand Sign Digit Classification — Model Definition
================================================================================
Course  : AI 100 — Deep Learning Midterm Project
Authors : Jasraj "Jay" Raval & Jack Sweeney
Date    : March 1, 2026
GitHub  : https://github.com/N4w4fn4ss4r/Midterm-project
--------------------------------------------------------------------------------

Overview
--------
This module defines `ASL_CNN`, a custom Convolutional Neural Network for
classifying American Sign Language (ASL) hand sign digit images into one of
10 classes (digits 0–9).

Architecture summary:
    Input  : [B, 3, 64, 64]  — normalized RGB image
    Block 1: Conv(3→32)  + BatchNorm + ReLU + MaxPool → [B, 32, 32, 32]
    Block 2: Conv(32→64) + BatchNorm + ReLU + MaxPool → [B, 64, 16, 16]
    Block 3: Conv(64→128)+ BatchNorm + ReLU + MaxPool → [B, 128, 8, 8]
    Flatten: [B, 8192]
    FC1    : Linear(8192→256) + ReLU + Dropout(0.5)
    Output : Linear(256→10)  — raw logits

Trainable parameters: ~2.4 million

Design Decisions
----------------
- 3×3 kernels with padding=1 preserve spatial dimensions before pooling.
  Two stacked 3×3 convolutions (as in VGGNet) have the same receptive field
  as one 5×5 convolution but with fewer parameters and more non-linearity.

- BatchNorm2d after each Conv layer normalizes activations per mini-batch,
  stabilizing training and enabling faster convergence. It also provides mild
  regularization, reducing dependence on Dropout in the convolutional layers.

- ReLU activation is used throughout. It avoids the vanishing gradient problem
  that affects sigmoid/tanh activations in deep networks and is computationally
  efficient (a simple threshold operation).

- MaxPool2d(2×2) halves spatial resolution after each block, building
  translation invariance and reducing the number of parameters in later layers.

- Dropout(p=0.5) is applied in the classifier head only. Dropout in conv layers
  has shown less benefit than in fully connected layers (BatchNorm provides
  sufficient regularization for conv layers). p=0.5 is the standard default.

- The number of filters doubles with each block (32 → 64 → 128). As spatial
  resolution decreases, channel depth increases to maintain representational
  capacity — a design pattern used in VGGNet, ResNet, and most modern CNNs.

Usage
-----
    from model import ASL_CNN
    model = ASL_CNN(num_classes=10)
    print(model)

    # Forward pass
    import torch
    x = torch.randn(8, 3, 64, 64)   # batch of 8 images
    logits = model(x)                # shape: [8, 10]
    probs = torch.softmax(logits, dim=1)

References
----------
- LeCun et al. (1998) Gradient-based learning applied to document recognition.
- Simonyan & Zisserman (2015) Very Deep Convolutional Networks (VGGNet).
- Ioffe & Szegedy (2015) Batch Normalization.
- Srivastava et al. (2014) Dropout: A Simple Way to Prevent Neural Networks
  from Overfitting.
================================================================================
"""

import torch
import torch.nn as nn


class ASL_CNN(nn.Module):
    """
    Custom Convolutional Neural Network for ASL digit classification (0–9).

    The network consists of three convolutional blocks followed by a
    fully connected classifier head. Each convolutional block applies:
        Conv2d → BatchNorm2d → ReLU → MaxPool2d

    The classifier applies:
        Linear → ReLU → Dropout → Linear (output logits)

    Args:
        num_classes (int): Number of output classes. Default: 10 (digits 0–9).

    Input:
        x (Tensor): Batch of RGB images with shape [B, 3, 64, 64].
                    Pixel values should be normalized to [-1, 1] using
                    mean=0.5, std=0.5 per channel.

    Output:
        Tensor of shape [B, num_classes] containing raw (unnormalized) logits.
        Pass through softmax for class probabilities, or directly to
        torch.nn.CrossEntropyLoss during training.

    Example:
        >>> model = ASL_CNN(num_classes=10)
        >>> x = torch.randn(4, 3, 64, 64)
        >>> logits = model(x)
        >>> print(logits.shape)   # torch.Size([4, 10])
    """

    def __init__(self, num_classes: int = 10):
        super(ASL_CNN, self).__init__()

        # ── Convolutional Feature Extractor ─────────────────────────────────
        # Three progressively deeper convolutional blocks.
        # Spatial dims are halved at each MaxPool: 64→32→16→8.
        # Channel depth doubles at each block: 3→32→64→128.
        self.features = nn.Sequential(

            # ── Block 1: 64×64×3 → 32×32×32 ──────────────────────────────
            # Learns low-level features: edges, color gradients, corners.
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,      # padding=1 preserves spatial dims before pool
                bias=False      # bias=False when using BatchNorm (BN adds bias)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64×64 → 32×32

            # ── Block 2: 32×32×32 → 16×16×64 ──────────────────────────────
            # Learns mid-level features: curves, finger contours, hand regions.
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32×32 → 16×16

            # ── Block 3: 16×16×64 → 8×8×128 ───────────────────────────────
            # Learns high-level features: full digit configurations,
            # which fingers are extended vs. curled, overall hand shape.
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16×16 → 8×8
        )

        # ── Fully Connected Classifier Head ─────────────────────────────────
        # After the feature extractor, spatial feature maps are flattened to
        # a 1D vector of size 128 × 8 × 8 = 8,192, then passed through
        # two dense layers to produce class logits.
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # [B, 128, 8, 8] → [B, 8192]

            nn.Linear(128 * 8 * 8, 256),           # First dense layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                      # Regularization: disables 50%
                                                    # of neurons during training

            nn.Linear(256, num_classes),            # Output logits: one per class
        )

        # ── Weight Initialization ────────────────────────────────────────────
        # Apply Kaiming (He) initialization to Conv and Linear layers.
        # This helps avoid vanishing/exploding gradients at the start of training.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input batch of shape [B, 3, 64, 64].

        Returns:
            Tensor of shape [B, num_classes] — raw class logits.
        """
        x = self.features(x)       # Extract spatial features
        x = self.classifier(x)     # Classify
        return x

    def _initialize_weights(self):
        """
        Apply Kaiming He initialization to Conv2d and Linear layers.

        Kaiming initialization sets initial weights such that the variance of
        activations is preserved across layers for ReLU-activated networks,
        preventing vanishing or exploding gradients at initialization.

        BatchNorm weight (gamma) is initialized to 1 and bias (beta) to 0,
        which corresponds to identity transformation at initialization — a
        safe starting point.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)   # gamma = 1
                nn.init.zeros_(module.bias)    # beta  = 0

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in',
                                        nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_feature_maps(self, x: torch.Tensor) -> list:
        """
        Return intermediate feature maps from each conv block.
        Useful for visualization and debugging.

        Args:
            x (Tensor): Input batch of shape [B, 3, 64, 64].

        Returns:
            List of Tensors: [block1_output, block2_output, block3_output]
        """
        feature_maps = []
        # Block 1: layers 0–3 (Conv, BN, ReLU, Pool)
        x = self.features[0](x)   # Conv
        x = self.features[1](x)   # BN
        x = self.features[2](x)   # ReLU
        x = self.features[3](x)   # MaxPool
        feature_maps.append(x.detach())

        # Block 2: layers 4–7
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x = self.features[7](x)
        feature_maps.append(x.detach())

        # Block 3: layers 8–11
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        feature_maps.append(x.detach())

        return feature_maps


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ASL_CNN(num_classes=10)
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")

    # Test forward pass
    dummy_input = torch.randn(4, 3, 64, 64)
    logits = model(dummy_input)
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (4, 10), "Output shape mismatch!"
    print("\nSanity check passed.")
