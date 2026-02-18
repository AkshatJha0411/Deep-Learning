# Deep Learning Assignments

This repository contains the implementation and analysis for Deep Learning assignments, covering the fundamentals of neural networks (from scratch) to Convolutional Neural Networks (CNNs).

## Assignment 1: Neural Networks from Scratch & Vision

This assignment focuses on understanding the core mechanics of neural networks by building one from the ground up using NumPy and exploring feature interpretation in vision tasks.

### Part 1: The "No-Framework" Challenge (NumPy Only)
**Goal:** Understand the inner workings of a neural network "engine".
-   **Dataset:** UCI Adult Census Income (Features -> Income >$50K)
-   **Key Tasks:**
    -   **Building the Engine:** Implemented a 3-layer Fully Connected Neural Network (FCNN) using only NumPy.
        -   Forward propagation (ReLU, Sigmoid).
        -   Backward propagation (Gradient computations).
        -   Optimizers: SGD and Adam.
    -   **Pre-Processing:** Analyzed the critical role of data normalization (Min-Max scaling) in convergence and accuracy.

### Part 2: Vision & Feature Interpretation
**Goal:** Bridge the gap between raw pixels and learned patterns using PyTorch/TensorFlow.
-   **Dataset:** MNIST Handwritten Digits
-   **Key Tasks:**
    -   **Weight Visualization:** Trained an FCNN and visualized the weights of the first hidden layer as 28x28 heatmaps to understand what shapes or patterns individual neurons are learning (e.g., strokes, edges).

---

## Assignment 2: Convolutional Neural Networks (CNNs)

This assignment delves into the geometry of convolutions, the architectural advantages of CNNs over FCNNs, and their property of spatial invariance.

### Part 1: Geometry of Convolutions
**Goal:** Master the spatial arithmetic of CNNs.
-   **Tasks:**
    -   **Manual Dimension Calculation:** Manually computed feature map dimensions $(H, W, C)$ at each stage of a specific architecture:
        -   $3\times$ Conv layers ($3\times3$ kernels)
        -   $2\times$ MaxPool layers ($2\times2$, stride 2)
        -   $1\times$ Fully Connected layer
    -   **Verification:** Implemented the exact architecture in PyTorch to verify manual calculations against actual tensor shapes.
    -   **Parameter Analysis:** Demonstrated the "parameter explosion" that occurs in the final Fully Connected layer when pooling layers are removed, highlighting the efficiency of pooling in reducing dimensionality.
-   **Dataset used for dimension checks:** Tiny ImageNet (resized to 64x64).

### Part 2: The "Why CNN?" Experiment (Spatial Invariance)
**Goal:** Demonstrate the robustness of CNNs to spatial transformations compared to FCNNs.
-   **Dataset:** MNIST Handwritten Digits
-   **Experiment:**
    -   **Robustness Duel:** Trained two models on standard MNIST:
        1.  **Model A:** Best FCNN from Assignment 1.
        2.  **Model B:** A simple 2-layer CNN.
    -   **The Test:** Evaluated both models on a "Shifted MNIST" test set where every image was translated **4 pixels to the right**.
-   **Analysis:**
    -   Reported the accuracy drop for both models.
    -   Explained CNN's superior performance due to **weight sharing** and **spatial invariance** (or equivariance + pooling), contrast to FCNN's sensitivity to absolute pixel positions.
