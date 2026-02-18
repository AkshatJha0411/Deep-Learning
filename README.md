# Deep Learning Assignments

This repository contains the implementation and analysis for Deep Learning assignments, covering the fundamentals of neural networks (from scratch) to Convolutional Neural Networks (CNNs).

<!-- ### Equations from the Assignment 1 and Report

Here are the key equations extracted and formatted properly from the provided documents. I've presented each on a new line in a mathematical equation style (using LaTeX-like notation for clarity, as it's common for equations in text).

1. **Forward Pass** (from Assignment Part 1.1):  
   \[ Z = W \cdot X + b \]  
   (Followed by ReLU activation.)

2. **SGD Update Rule** (from Assignment Part 1.1 and Report Optimization):  
   \[ W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W} \]  
   (Where \(\eta\) is the learning rate.)

3. **He-Initialization for Weights** (from Report Architecture, for ReLU):  
   \[ W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right) \]  
   (Variance = 2 / fan_in; biases initialized to zeros.)

(Note: No other explicit equations are visible in the provided image texts. Binary Cross-Entropy loss is mentioned but not equation-formatted in the images.) -->

# Fully Connected Neural Net (FCNN) Assignment

## Overview
**Title:** Coding Assignment: Fully Connected Neural Net (FCNN)  
**Objective:** Build a Deep Learning engine from the ground up using NumPy, then scale to industrial frameworks like PyTorch/TensorFlow. Focus on understanding neural network internals, feature scaling, spatial awareness, and robustness in deep models.  

**Datasets Used:**  
- Part 1: UCI Adult Census Income (binary classification: income >50K vs. <=50K).  
- Part 2: MNIST Handwritten Digits (multi-class: 0-9).  
- Part 3: Tiny ImageNet (10 classes, multi-class image classification).  

## Part 1: The "No-Framework" Challenge (NumPy Only)
**Dataset:** UCI Adult Census Income.  
**Goal:** Understand the "Black Box" by building it from scratch.  

### Question 1.1: Building the Engine
**Description:** Implement a 3-layer FCNN using only NumPy/Python (no frameworks). Key components:  
- Layer Initialization: He-Initialization for weights (variance = 2/fan_in), zeros for biases.  
- Forward Pass: Z = W * X + b, followed by ReLU activation (Sigmoid for output).  
- Optimizer: Basic SGD (W_new = W_old - learning_rate * ∂L/∂W).  
- Backpropagation: Manually derive/code chain rule for gradients of weights/biases.  
- Deliverable: Training loop outputting loss every 100 iterations; achieve >=75% test accuracy.  

**Analysis:** A from-scratch model achieves ~75-85% accuracy, demonstrating core mechanics work. SGD converges smoothly on balanced data, but gradients can explode/vanish without proper init.

### Question 1.2: The Importance of "Pre-Processing"
**Description:** Train NumPy model twice: raw data vs. Min-Max Scaled data. Compare results; document gradient issues with varying feature scales (e.g., "Age" vs. "Capital Gain").  

**Analysis:** Scaled + SGD: 81.62% (smooth convergence). Raw + SGD: 75.50% (stagnated, gradients exploded/vanished due to elongated loss landscape). Raw + Adam: 84.72% (adaptive rates handle scales better). Preprocessing is crucial; unscaled features cause inefficient learning.

## Part 2: Vision & Feature Interpretation (Frameworks Allowed)
**Dataset:** MNIST Handwritten Digits.  
**Goal:** Bridge pixels to patterns using PyTorch/TensorFlow.  

### Question 2.1: Weight Visualization
**Description:** Build FCNN in a framework. Extract/reshape first hidden layer weights to 28x28 heatmaps for 10 neurons. Observe/describe shapes (dots, lines, or noise?).  

**Analysis:** Weights show clear structures like strokes/edges/curves (not noise). Neurons act as pattern detectors for digit parts, revealing interpretable features even in flattened inputs. Scrambled weights are noisier but still patterned.

### Question 2.2: The "Flattening" Experiment
**Description:** Shuffle pixels in all MNIST images (fixed pattern). Train on scrambled data; compare accuracy to normal MNIST. Observe why similar performance (vs. human difficulty).  

**Analysis:** Normal: 98.15%; Scrambled: 98.24% (near-identical). FCNN treats input as unordered vector, ignoring spatial relations—relearns permuted rep. Highlights limitation: no built-in spatial awareness; needs conv nets for images.

## Part 3: Stress Testing & Robustness
**Dataset:** Tiny ImageNet (10 classes).  

### Question 3.1: Vanishing Gradients & Modern Fixes
**Description:** Build deep FCNN (8+ layers).  
- Experiment A: Sigmoid activations throughout.  
- Experiment B: ReLU + Batch Normalization.  
- Plot first-layer gradient norm; explain faster training.  

**Analysis:** Exp B (ReLU+BN) trains faster/higher acc (~34.8% val) with stable gradients (~0.25-1.7). Exp A: Tiny norms (~1e-6 early), vanishes due to Sigmoid saturation. ReLU avoids saturation; BN scales inputs for stability.

### Question 3.2: The Ablation Study
**Description:** "Switch-Off" test: Report accuracy changes by:  
1. Removing Dropout.  
2. Changing LR by factor of 10 (high vs. low).  
3. Switching Adam to Vanilla SGD.  
- Deliverable: Summary table of biggest impacts.  

**Analysis:** Base (ReLU+BN+Dropout0.5+LR0.001+Adam): 37.4%. No Dropout: +6.4% (reduced underfitting). LRx10: -1.6% (instability). LR/10: -26.8% (slow convergence). SGD: -27.2% (no adaptivity). Optimizer/LR have largest impact; adaptive methods essential for deep/image data.

## Key Overall Insights
- FCNNs learn interpretable but non-spatial features; sensitive to scaling/depth.  
- Modern fixes (ReLU, BN, Adam) enable stable deep training.  
- Experiments emphasize hands-on understanding over black-box use.  

---

<!-- ### Equations from the Assignment 2 and Report

Here are the key equations extracted and formatted properly from the provided documents. I've presented each on a new line in a mathematical equation style (using LaTeX-like notation for clarity, as it's common for equations in text).

1. **Convolution Output Shape** (from Assignment Part 1.1 and Report Section 1.1):  
   \[ H_{\text{out}} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1 \]  
   (Similarly for \( W_{\text{out}} \). With K=3, S=1, P=1 → H_out = H.)

2. **Max-Pooling Output Shape** (implied from Assignment Part 1.1):  
   \[ H_{\text{out}} = \left\lfloor \frac{H - K}{S} \right\rfloor + 1 \]  
   (With K=2, S=2 → halves H/W.)

3. **Internal Covariate Shift Reduction via BatchNorm** (from Report Section 4.1, implied in plots):  
   \[ \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \]  
   \[ y = \gamma \hat{x} + \beta \]  
   (Where \(\mu_B\), \(\sigma_B\) are batch mean/variance; \(\gamma\), \(\beta\) learnable.)

(Note: Other concepts like weight sharing or augmentation are descriptive; no additional explicit equations in the provided texts.) -->

# Convolutional Neural Net (CNN) Assignment

## Overview
**Title:** Coding Assignment: Convolutional Neural Net (CNN)  
**Objective:** Understanding the spatial intelligence of CNNs through geometry, invariance, feature visualization, and optimization.  

**Datasets Used:**  
- MNIST Handwritten Digits (multi-class: 0-9).  
- Tiny-ImageNet-10 (10 classes, multi-class image classification; from Assignment 1).  

## Part 1: The Geometry of Convolutions
**Goal:** Move from "Black Box" coding to understanding spatial dimensions.  

### Question 1.1: The Manual Dimension Map
**Description:** Design a CNN with 3 Conv layers (3x3 kernels), 2 Max-Pooling (2x2, stride 2), and 1 FC layer. For input 64x64x3 (Tiny-ImageNet-10), calculate height/width/channels at each step. Implement in PyTorch/Keras with custom forward/Functional API; verify via print(x.shape) or model.summary(). Discuss parameter increase if pooling removed ("Parameter Explosion").  

**Analysis:** Manual shapes match PyTorch (e.g., Input: 64x64x3 → Conv1: 64x64x16 → Pool1: 32x32x32 → Flatten: 16384 dims). Without pooling, FC params explode (e.g., from ~100k to millions) due to larger flattened input, causing overfitting/compute issues.

## Part 2: The "Why CNN?" Experiment (Spatial Invariance)
**Goal:** Prove why we moved away from FCNNs for vision.  

### Question 2.1: The Robustness Duel
**Description:** Train on MNIST: Model A (best FCNN from Assignment 1) vs. Model B (simple 2-layer CNN). Create "Shifted MNIST" by translating images 4 pixels right. Report accuracy drop; analyze why CNN is robust (weight sharing).  

**Analysis:** Normal: FCNN ~98%, CNN ~99%. Shifted: FCNN drops ~20-30% (loses spatial structure); CNN drops ~5-10% (translation tolerance via shared weights and pooling). Weight sharing enables local pattern detection invariant to position.

## Part 3: Feature Extraction & Visual Interpretability
**Goal:** Visualizing how the machine "sees" textures vs. objects.  

### Question 3.1: The Filter Gallery
**Description:** Extract/plot first-layer kernels as grid for CNNs trained on MNIST/Tiny-ImageNet-10. Identify Gabor-like filters (edges/color blobs); compare to FCNN weights from Assignment 1 and across datasets.  

**Analysis:** MNIST filters: Edge/line detectors (Gabor-like). Tiny-ImageNet: More color-opponency (RGB blobs). Vs. FCNN: Structured (not noisy pixels); dataset-wise: MNIST grayscale edges vs. Tiny-ImageNet color/texture focus.

### Question 3.2: The Receptive Field Experiment
**Description:** For one image per dataset, visualize activation maps after 1st and final Conv layers alongside original. Observe focus shift (edges → objects); compare dataset differences.  

**Analysis:** 1st layer: Local edges/textures. Final: Holistic object parts. MNIST: Sharp digit outlines. Tiny-ImageNet: Broader color/shape focus. Shows hierarchical progression; RGB data adds color-specific activations.

## Part 4: Advanced Optimization & Robustness
**Goal:** Master the "Levers" of Deep Learning.  

### Question 4.1: The Depth vs. Normalization Duel
**Description:** Build deep CNN (6-8 layers) on Tiny-ImageNet-10. Exp A: No BatchNorm. Exp B: BatchNorm2d after each Conv. Plot mean/variance of 5th layer activations over 500 batches to show BatchNorm reduces Internal Covariate Shift.  

**Analysis:** Exp B: Stable mean~0, variance~1 (no shift/explosion); faster convergence, higher acc (~35% val). Exp A: Exploding/vanishing activations. BatchNorm normalizes, enabling deeper nets.

### Question 4.2: Data Augmentation "Sanity Check"
**Description:** Implement pipeline with transforms.RandomRotation(30) and transforms.ColorJitter(). Train with/without; report test acc on 20% split. Discuss if it helps more on train/test (why?).  

**Analysis:** No Aug: Test ~14.6% (overfits). With Aug: Test ~14.6% (similar, but curves show better generalization; reduces train-test gap). Helps test more by promoting invariant features, preventing memorization.

## Key Overall Insights
- CNNs provide spatial inductive bias via weight sharing/pooling, hierarchical features, and stability fixes.  
- Superior to FCNNs for vision: Robust to shifts, interpretable filters, efficient deep training.  
- Experiments highlight geometry's role in efficiency and augmentation's in robustness.  
