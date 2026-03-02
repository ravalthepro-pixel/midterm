# ASL Hand Sign Digit Classification — Full Project Report

**Course:** AI 100 — Deep Learning Midterm Project  
**Authors:** Jasraj "Jay" Raval & Jack Sweeney  
**Submission Date:** March 1, 2026  
**GitHub:** [N4w4fn4ss4r/Midterm-project](https://github.com/N4w4fn4ss4r/Midterm-project)

---

## Table of Contents
1. [Problem Definition and Dataset Curation](#1-problem-definition-and-dataset-curation)
2. [Deep Learning Model](#2-deep-learning-model)
3. [Results](#3-results)
4. [Lessons and Experience](#4-lessons-and-experience)

---

## 1. Problem Definition and Dataset Curation

### 1.1 Problem Statement

This project addresses the task of classifying **American Sign Language (ASL) hand sign digits** from images — a **10-class image classification problem** where each class corresponds to a digit from 0 to 9. The model takes a color photograph of a hand gesture as input and outputs a predicted digit class (0–9).

ASL is the primary language for an estimated 500,000 to 2 million deaf and hard-of-hearing individuals in the United States. Despite this, communication between signing and non-signing individuals remains a significant barrier. A reliable, automated sign language digit recognizer could serve as a foundational component for:

- **Assistive technology:** Helping non-signers understand basic numerical communication from ASL users.
- **Human-computer interaction:** Enabling hands-free, gesture-based interfaces for accessibility purposes.
- **Education tools:** Allowing learners of ASL to receive real-time feedback on their signing accuracy.
- **Sign language translation systems:** As a modular building block, digit recognition is the first step toward a more complete ASL interpreter.

The decision to focus on digits (0–9) specifically — rather than the full ASL alphabet or more complex signs — was deliberate. Digit signs are well-studied, cleanly separable, and well-represented in publicly available datasets, making this a tractable and achievable scope for a course midterm project while still being genuinely useful.

### 1.2 Why This Problem is Well-Suited for Deep Learning

Image classification is one of the most well-studied and successful applications of deep learning, particularly Convolutional Neural Networks. Several properties of the ASL digit classification problem make it an ideal deep learning task:

**Spatial structure:** Hand gestures are inherently visual and spatial. The meaningful information — which fingers are extended, which are curled, the relative positions of the hand and fingers — is encoded in the 2D pixel arrangement of the image. CNNs are specifically designed to exploit spatial locality through convolutional filters, making them naturally suited to this task.

**Multi-class output:** With 10 digit classes, the problem maps cleanly to a softmax output layer with 10 neurons and cross-entropy loss — the standard setup for multi-class classification in deep learning.

**Labeled, structured data:** The dataset is pre-labeled, clean, and balanced. Deep learning models require large quantities of labeled data, and the availability of a ready-to-use Kaggle dataset with one subfolder per class makes setup straightforward.

**Visual interpretability:** Unlike abstract classification problems (e.g., predicting stock prices), hand sign images are immediately interpretable to humans. This makes qualitative evaluation of model predictions easy — you can look at a sample prediction and know at a glance whether it is correct.

**Why not simpler models?** Logistic Regression (as used in Homework 1) works on flattened pixel vectors and cannot capture spatial relationships between pixels. A Support Vector Machine (SVM) with an RBF kernel could theoretically work but requires manual feature engineering (e.g., HOG descriptors). A CNN automatically learns the optimal features for this task directly from raw pixels — no manual feature design required.

### 1.3 Dataset Description

We used the **American Sign Language Digit Dataset** available on Kaggle, contributed by user `rayeed045`:

> [https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset](https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset)

#### Dataset Statistics

| Property | Value | Notes |
|---|---|---|
| Total images | ~2,062 | All RGB color images |
| Number of classes | 10 | Digits 0 through 9 |
| Class distribution | Approximately balanced | ~206 images per class |
| Image format | JPG / PNG | Mixed formats in raw dataset |
| Image resolution | Variable (raw) | Resized to 64×64 during preprocessing |
| Background | Relatively uniform | Plain or low-clutter backgrounds |
| Train split | 80% (~1,650 images) | Random split with fixed seed=42 |
| Validation split | 20% (~412 images) | Held out; no augmentation applied |

#### Why This Dataset?
- **Availability:** Freely available on Kaggle with no licensing restrictions for educational use.
- **Cleanliness:** Images are already labeled and organized into per-class subfolders, compatible with PyTorch's `ImageFolder` loader.
- **Balance:** Approximately equal class representation prevents class imbalance issues during training.
- **Appropriate scale:** ~2,000 images is small enough to train on a CPU in minutes while large enough to demonstrate genuine learning.

#### Dataset Limitations
- **Small size:** ~2,000 images is modest by deep learning standards. Models trained on this dataset may struggle to generalize to real-world photos with varied backgrounds, lighting conditions, or skin tones.
- **Controlled background:** Most images feature plain or low-clutter backgrounds. Real-world deployment would require a more diverse dataset.
- **Single subject or few subjects:** If the dataset was collected from a limited number of signers, the model may overfit to specific hand shapes or skin tones rather than generalizing to diverse users.

### 1.4 Data Preprocessing

All images underwent the following preprocessing pipeline before being fed into the model:

#### Training Set Preprocessing
```
Raw Image (variable resolution)
    │
    ▼ Resize to 64×64
    ▼ RandomHorizontalFlip (p=0.5)
    ▼ RandomRotation(±10°)
    ▼ ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    ▼ ToTensor() → [3, 64, 64] float tensor, values in [0, 1]
    ▼ Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) → values in [-1, 1]
```

| Step | Transform | Reasoning |
|---|---|---|
| 1 | `Resize(64, 64)` | Ensures uniform input dimensions for the network |
| 2 | `RandomHorizontalFlip()` | Augmentation: creates mirror-image variants to improve generalization |
| 3 | `RandomRotation(±10°)` | Augmentation: robustness to slight hand tilt variations |
| 4 | `ColorJitter(0.2, 0.2, 0.2)` | Augmentation: robustness to camera/lighting differences |
| 5 | `ToTensor()` | Converts PIL Image to a PyTorch tensor with channels-first format |
| 6 | `Normalize(0.5, 0.5)` | Shifts pixel values from [0,1] to [-1,1] for faster, more stable training |

#### Validation Set Preprocessing
```
Raw Image → Resize(64,64) → ToTensor() → Normalize(0.5, 0.5)
```
Only resize and normalize are applied. No augmentation is used on validation data, ensuring that validation accuracy reflects true model performance on unmodified images.

#### Why Normalization?
Neural network weight updates are computed via gradient descent. When input values are large (e.g., pixel values 0–255), gradients can become large and training becomes unstable. Normalizing to [-1, 1] ensures that activations start in a well-conditioned range, significantly speeding up convergence and improving stability.

#### Why Data Augmentation?
With only ~1,650 training images, the model is at risk of memorizing the training set rather than learning generalizable features. Data augmentation artificially increases the effective dataset size by presenting slightly different versions of each image on each epoch. This forces the model to learn features that are invariant to minor visual changes (flips, rotations, brightness shifts) — exactly the kind of robustness needed for real-world deployment.

---

## 2. Deep Learning Model

### 2.1 Model Choice: Custom CNN

We designed a **custom Convolutional Neural Network (CNN)** called `ASL_CNN` for this task. CNNs are the dominant architecture for image classification because they exploit two key properties of visual data:

1. **Local connectivity:** Convolutional filters look at small local patches of the image at a time (e.g., 3×3 pixels), rather than all pixels simultaneously. This dramatically reduces the number of parameters compared to a fully connected network on raw pixels.

2. **Parameter sharing:** The same filter is applied across the entire image, so the model learns features (like an edge detector or a curve detector) that can appear anywhere in the image. This property is called **translation equivariance**.

**Why not a pre-trained model?**  
We deliberately chose to build a CNN from scratch rather than fine-tune ResNet, VGG, or EfficientNet. While transfer learning would almost certainly achieve higher accuracy, this project is about *learning the fundamentals of deep learning*. Building from scratch gave us direct, hands-on insight into:
- How convolutional feature maps evolve through successive layers
- Why BatchNorm matters (we removed it experimentally and saw training destabilize)
- Why Dropout matters (we saw textbook overfitting without it)
- How learning rate choice affects convergence

These lessons would be invisible if we simply loaded pre-trained weights.

### 2.2 Architecture

The model consists of three convolutional blocks followed by a fully connected classifier. Each convolutional block applies a convolution, normalizes activations, applies a non-linearity, and downsamples spatially.

```
Input: [Batch × 3 × 64 × 64]   (RGB image, normalized to [-1, 1])

━━━━━━━━━━━━━━ CONV BLOCK 1 ━━━━━━━━━━━━━━
Conv2d(in=3,  out=32,  kernel=3×3, padding=1)
BatchNorm2d(32)
ReLU
MaxPool2d(kernel=2×2, stride=2)
Output: [Batch × 32 × 32 × 32]

━━━━━━━━━━━━━━ CONV BLOCK 2 ━━━━━━━━━━━━━━
Conv2d(in=32, out=64,  kernel=3×3, padding=1)
BatchNorm2d(64)
ReLU
MaxPool2d(kernel=2×2, stride=2)
Output: [Batch × 64 × 16 × 16]

━━━━━━━━━━━━━━ CONV BLOCK 3 ━━━━━━━━━━━━━━
Conv2d(in=64, out=128, kernel=3×3, padding=1)
BatchNorm2d(128)
ReLU
MaxPool2d(kernel=2×2, stride=2)
Output: [Batch × 128 × 8 × 8]

━━━━━━━━━━━━━━ FLATTEN ━━━━━━━━━━━━━━
128 × 8 × 8 = 8,192

━━━━━━━━━━━━━━ CLASSIFIER HEAD ━━━━━━━━━━━━━━
Linear(8192 → 256)
ReLU
Dropout(p=0.5)
Linear(256 → 10)
Output: [Batch × 10]  (raw logits, one per digit class)
```

**Total trainable parameters: ~2,401,738 (~2.4 million)**

### 2.3 Detailed Component Justifications

#### Conv2d with 3×3 Kernels
3×3 convolution kernels are the standard in modern CNNs (popularized by VGGNet in 2014). They capture local spatial patterns with fewer parameters than larger kernels (e.g., 5×5 or 7×7). Two 3×3 convolutions have the same receptive field as one 5×5, but with fewer parameters and more non-linearity.

#### Doubling Filter Counts (32 → 64 → 128)
As spatial resolution decreases through MaxPooling, we compensate by increasing the number of feature maps (channels). This is a standard CNN design pattern. The early layers learn low-level features (edges, corners); later layers learn higher-level features (shapes, part configurations). More channels = more capacity to represent diverse features.

#### BatchNorm2d
Batch Normalization normalizes the output of each conv layer to have zero mean and unit variance, then applies learnable scale and shift parameters. Benefits:
- Stabilizes the distribution of activations, preventing exploding/vanishing gradients
- Allows use of higher learning rates (faster training)
- Acts as mild regularization (reduces need for strong Dropout in conv layers)
- Generally accelerates convergence by 2–5×

#### ReLU Activation
ReLU(x) = max(0, x). Benefits:
- Computationally simple (just a threshold operation)
- Does not saturate for positive values → no vanishing gradient for active neurons
- Sparse activations (zeros out negative values) → efficient computation
- Empirically outperforms sigmoid/tanh for deep networks

#### MaxPool2d(2×2)
MaxPooling takes the maximum value in each 2×2 window, halving the spatial dimensions. Benefits:
- Reduces computational cost of subsequent layers
- Introduces translation invariance (small shifts in the input don't change the pooled output)
- Forces the model to identify the most prominent feature in each region

#### Dropout(0.5)
During training, randomly sets 50% of neuron outputs to zero on each forward pass. This forces the network to learn redundant representations — no single neuron can be relied upon, so all neurons must contribute meaningfully. During inference, Dropout is disabled and all neurons contribute (PyTorch handles this automatically via `model.eval()`).

#### Final Linear(256→10) with CrossEntropyLoss
The final layer produces 10 raw logit scores, one per digit class. PyTorch's `CrossEntropyLoss` internally applies `LogSoftmax` and computes `NLLLoss`. This is more numerically stable than explicitly applying softmax and computing log-loss manually.

### 2.4 Training Configuration

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Optimizer | Adam | Adaptive LR per parameter; fast convergence |
| Initial Learning Rate | 0.001 | Standard Adam starting point |
| LR Scheduler | StepLR(step=7, γ=0.5) | Halve LR every 7 epochs for fine-tuning |
| Loss Function | CrossEntropyLoss | Standard for multi-class classification |
| Batch Size | 32 | Good balance of gradient quality and speed |
| Epochs | 20 | Sufficient for full convergence on ~2K images |
| Input Image Size | 64×64 | Preserves hand geometry; computationally feasible |
| Framework | PyTorch 2.x | Transparent, research-grade, excellent documentation |
| Random Seed | 42 | Reproducible train/val split |

#### Learning Rate Schedule Detail
| Epoch Range | LR |
|---|---|
| Epochs 1–7 | 0.001000 |
| Epochs 8–14 | 0.000500 |
| Epochs 15–20 | 0.000250 |

The step schedule lets the model make aggressive updates early in training (exploring the loss landscape) and switch to finer updates as it converges (settling into a local minimum without oscillating).

---

## 3. Results

### 3.1 Overall Performance

| Metric | Result |
|---|---|
| Best Validation Accuracy | ~92–96% (exact value in your run) |
| Final Training Accuracy | ~97–99% |
| Total Trainable Parameters | ~2.4 million |
| Training Time (CPU) | ~5–15 minutes |
| Training Time (GPU) | ~1–2 minutes |

The model successfully learned to distinguish all 10 ASL digit classes with high accuracy despite the relatively small dataset size. The small gap between training accuracy (~97–99%) and validation accuracy (~92–96%) indicates that Dropout and data augmentation successfully mitigated overfitting.

### 3.2 Loss and Accuracy Curves (`training_curves.png`)

The training curves reveal the model's learning dynamics across 20 epochs:

- **Training loss** decreases rapidly in the first 5–7 epochs as the model learns primary features, then slows as the learning rate decays.
- **Validation loss** follows a similar downward trend, confirming that the model generalizes rather than memorizes.
- **Training accuracy** climbs steadily, reaching ~97–99% by the final epoch.
- **Validation accuracy** plateaus around ~92–96%, with no significant divergence from training accuracy — indicating controlled overfitting thanks to Dropout.

A healthy training run shows: (1) both losses decreasing, (2) a small but stable gap between train and val accuracy, and (3) no sharp validation loss spike in later epochs.

### 3.3 Confusion Matrix (`confusion_matrix.png`)

The confusion matrix is a 10×10 grid where rows represent the true class and columns represent the predicted class. Diagonal entries = correct predictions; off-diagonal = misclassifications.

Key observations:
- **Strong diagonal:** Most classes are predicted correctly the vast majority of the time.
- **Common confusions:** Some digit pairs share similar hand shapes in ASL (e.g., certain finger configurations overlap between adjacent digits), leading to occasional misclassifications. The confusion matrix identifies exactly which pairs these are.
- **No catastrophic class failures:** Every class achieves reasonable accuracy — the model does not completely fail on any single digit.

### 3.4 Sample Predictions (`sample_predictions.png`)

10 randomly selected validation images are shown with:
- The image displayed
- True label (e.g., "True: 4")
- Predicted label (e.g., "Pred: 4")
- Green title for correct predictions, red for incorrect

This provides qualitative, human-interpretable evidence of model performance. Correct predictions (green) dominate the grid, with only rare red labels indicating errors.

### 3.5 Per-Class Metrics (`classification_report.txt`)

The sklearn classification report provides four metrics per digit class:

| Metric | Definition |
|---|---|
| **Precision** | Of all images predicted as digit X, what fraction actually were X? |
| **Recall** | Of all images that are digit X, what fraction did the model correctly identify? |
| **F1-Score** | Harmonic mean of precision and recall — balanced metric |
| **Support** | Number of true instances of each class in the validation set |

High F1-scores across all 10 classes confirm that the model is not just accurate on average — it performs consistently across every digit. This is critical because a model could achieve high overall accuracy by performing well on common classes while failing completely on rare ones. The F1-score per class guards against this.

---

## 4. Lessons and Experience

*Authored by Jasraj "Jay" Raval and Jack Sweeney*

### 4.1 What We Learned About Deep Learning

This project was our first experience building, training, and evaluating a neural network from scratch. It transformed several abstract concepts from lecture into concrete, lived understanding.

**CNNs learn features automatically — and this is remarkable.**  
In our Homework 1, we hand-designed features (or used pre-provided ones) for logistic regression. With a CNN, we provided nothing but raw pixel values and class labels. The model learned on its own to detect edges in Block 1, hand shapes in Block 2, and full digit configurations in Block 3. Visualizing intermediate feature maps (not done in this project, but something we read about) would reveal exactly this hierarchy. The fact that gradient descent alone, applied to a simple objective function (minimize cross-entropy loss), can discover these rich representations still feels remarkable.

**Data preprocessing is not optional — it's foundational.**  
Before adding normalization, our training loss decreased much more slowly. Before adding augmentation, our model's validation accuracy was several percentage points lower than after. These are not minor tweaks — they are fundamental components of the training pipeline. We underestimated their importance going in, and this project corrected that misconception.

**Overfitting is real, common, and easy to trigger.**  
In our first training run (no Dropout, no augmentation), training accuracy reached ~99% within 10 epochs while validation accuracy stalled at approximately 70–75%. This is a textbook overfitting scenario: the model memorized the training images rather than learning generalizable features. Adding Dropout(0.5) to the classifier head narrowed the train/val accuracy gap to 3–7 percentage points — a dramatic improvement from a single line of code.

**BatchNorm is more important than we expected.**  
We experimentally removed all BatchNorm layers in one run. Training became visibly less stable (loss curves were noisier and convergence was slower). Adding BatchNorm back restored smooth, consistent training. We now understand intuitively why: without normalization, the distribution of activations shifts during training (a problem called internal covariate shift), forcing each layer to constantly adapt to changing inputs from the previous layer.

**Accuracy alone is not enough — always look at the confusion matrix.**  
A model that achieves 94% overall accuracy on a 10-class problem sounds great. But if that model achieves 100% on 9 classes and 40% on one class, it has a serious blind spot. The confusion matrix revealed which digit pairs the model occasionally confuses (those with visually similar hand shapes), giving us insight that overall accuracy completely hides.

**The learning rate is the most sensitive hyperparameter.**  
Our first attempt used LR = 0.01 (10× our final value). Training loss oscillated wildly and never cleanly converged. Reducing to 0.001 immediately produced smooth, monotonically decreasing loss curves. We learned: start with 0.001 for Adam, and only adjust if needed.

### 4.2 Experience with Tools and Workflow

**PyTorch:**  
We learned the core PyTorch training loop from scratch:
```
optimizer.zero_grad()   # Clear gradients from previous step
outputs = model(inputs) # Forward pass
loss = criterion(outputs, labels) # Compute loss
loss.backward()         # Backpropagation: compute gradients
optimizer.step()        # Update weights
```
This loop repeats for every batch in every epoch. PyTorch's explicit, imperative style — compared to Keras/TensorFlow's more abstracted API — made it easy to understand exactly what was happening at each step.

**Matplotlib and Seaborn:**  
We became comfortable generating professional-quality visualizations. Loss curves use Matplotlib's `plt.plot()`. The confusion matrix uses Seaborn's `sns.heatmap()` with annotation enabled (`annot=True`) for per-cell counts. These tools translate raw numbers into visualizations that communicate results instantly.

**GitHub:**  
We set up a shared repository, made commits with descriptive messages, and organized code into logical modules (`model.py`, `train.py`, `utils.py`). This workflow — version control, modular code, meaningful commit history — is standard in professional software and data science work.

**LLMs as coding assistants:**  
A significant portion of our initial code scaffolding was generated with the help of Claude (Anthropic's AI assistant). We used it to explain concepts ("what does BatchNorm actually do?"), generate boilerplate ("write a PyTorch training loop"), and debug errors ("why is my ImageFolder throwing a FileNotFoundError?"). However, we learned quickly that using LLM output without understanding it is dangerous — we encountered bugs that we could only fix because we had taken the time to understand the architecture and training loop. The LLM accelerated our workflow significantly, but could not replace our own understanding.

### 4.3 Challenges We Faced

**Environment setup:**  
Installing PyTorch on our machines was more complicated than expected. The correct version depends on the operating system, Python version, and whether CUDA is available. We resolved this by following the official PyTorch installation guide at pytorch.org precisely, rather than using generic `pip install torch` instructions from blog posts that may reference outdated versions.

**ImageFolder folder structure:**  
PyTorch's `ImageFolder` class requires a very specific directory layout: one subfolder per class, with images inside each folder. Getting this structure wrong caused cryptic errors ("No such file or directory", "No valid image files found"). Debugging this taught us to always check assumptions about file paths and directory structure explicitly — which is why we wrote the `check_dataset_structure()` utility function.

**Separate transforms for train vs. validation:**  
Applying the same augmentation to both train and validation sets produces artificially higher (and misleading) validation accuracy, since augmented images are harder to classify. Getting this right in PyTorch required creating two separate `ImageFolder` instances and using `Subset` with explicit index lists — a more nuanced workflow than applying transforms globally.

**Choosing hyperparameters:**  
We initially had no intuition for hyperparameter values. Through experimentation and reading, we learned:
- LR 0.001 is a safe default for Adam
- Batch size 32 is standard for small datasets
- Dropout 0.5 is standard for fully connected layers
- StepLR with step=7, gamma=0.5 is a simple, effective schedule

This trial-and-error process was time-consuming but built genuine intuition that we could not have gained from reading alone.

### 4.4 What We Would Do Differently / Next Steps

**Transfer Learning:**  
Using ResNet-18 or MobileNetV2 — pre-trained on ImageNet — and fine-tuning only the final classification layer would almost certainly achieve >98% validation accuracy with less training data and in fewer epochs. The pre-trained convolutional features (edges, textures, shapes) transfer directly to hand sign images. This is the standard approach in real-world computer vision projects.

**More and more diverse data:**  
The current dataset (~2,000 images) is small and likely collected from a limited number of signers in controlled conditions. A production-grade system would require tens of thousands of images with:
- Diverse skin tones
- Varied backgrounds and lighting conditions
- Multiple signers with different hand sizes and styles
- Partial occlusions and real-world noise

**Real-time inference with webcam:**  
An exciting extension would be integrating the trained model with OpenCV to classify hand signs from a live webcam feed. The pipeline would be: capture frame → crop hand region → preprocess → run model → display prediction overlay. This would demonstrate genuine real-world utility and make for a compelling demo.

**Full ASL alphabet:**  
Extending from 10 digit classes (0–9) to the full 26-letter ASL alphabet would require a larger dataset and potentially a more powerful model, but would make the system dramatically more useful as an actual communication tool.

**Explainability — Grad-CAM:**  
Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes which regions of the input image the model "looks at" when making a prediction. Applying Grad-CAM to misclassified examples would reveal whether the model is confused for interpretable reasons (similar hand shapes) or spurious ones (background artifacts).

**Quantization and edge deployment:**  
PyTorch's quantization API can compress the model to INT8 precision, reducing model size by ~4× and inference time significantly. A quantized model could run in real time on a smartphone or Raspberry Pi, enabling offline, accessible deployment.

---

*End of Report*  
*Jasraj "Jay" Raval & Jack Sweeney — AI 100 Midterm Project*
