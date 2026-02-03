# 🧠 CNN Theory (For Viva)

## 1. What is Computer Vision?
Computer Vision is a field of AI that enables computers to "see" and interpret images or videos, just like human vision.
*   **Goal:** To extract meaningful information (like detecting a disease) from visual inputs.

---

## 2. What is a CNN (Convolutional Neural Network)?
A CNN is a specialized type of Deep Learning algorithm designed for processing grid-like data, such as images.
*   **Analogy:** Imagine looking at a picture through a small square window (filter). You slide this window across the image to find patterns. That's exactly what a CNN does!

---

## 3. Why CNN and not Standard ML (like SVM/Random Forest)?
*   **Spatial Hierarchy:** Standard ML flattens the image immediately, destroying the spatial structure (which pixel is next to which). CNN preserves this structure.
*   **Parameter Efficiency:** A standard Neural Network would need millions of weights for a single image. CNN shares weights (filters), making it efficient.
*   **Feature Learning:** In ML, you have to manually tell the computer "look for yellow spots." CNN *learns* to look for spots itself.

---

## 4. Key Layers of CNN (The "Building Blocks")

### A. Convolution Layer (The Feature Extractor)
*   **Operation:** A small matrix (kernel/filter) slides over the image performing element-wise multiplication.
*   **Goal:** To detect features like edges, corners, and textures.
*   **Filter/Kernel:** The small 3x3 matrix that does the scanning.

### B. ReLU Activation (The Non-Linearity)
*   **Name:** Rectified Linear Unit.
*   **Formula:** `f(x) = max(0, x)`
*   **Function:** It converts all negative values to zero.
*   **Why?** It introduces non-linearity, allowing the model to learn complex patterns (not just straight lines).

### C. Pooling Layer (The Downsampler)
*   **Operation:** Reduces the dimensions (width/height) of the feature map.
*   **Max Pooling:** Takes the largest value from a 2x2 window.
*   **Why?**
    1.  Reduces computational cost (fewer parameters).
    2.  Makes the model invariant to small shifts (if the leaf moves slightly, it still detects it).

### D. Flattening Layer
*   **Function:** Converts the 2D matrix into a long 1D vector.
*   **Why?** Because the final classification layers (Dense layers) expect a 1D input.

### E. Fully Connected (Dense) Layer
*   **Function:** A traditional neural network layer where every neuron is connected to every neuron in the previous layer.
*   **Goal:** To combine high-level features and perform the final classification.

### F. Softmax Output Layer
*   **Function:** Converts the raw output scores (logits) into probabilities that sum to 1.
*   **Example:** `[0.1, 0.8, 0.1]` means 80% confidence in class 2.

---

## 5. Important Concepts

### Loss Function (Categorical Cross-Entropy)
*   It measures the "error" or difference between the predicted probability and the actual label. The model tries to minimize this value.

### Optimizer (Adam)
*   It updates the weights of the network based on the loss. "Adam" is a smart optimizer that adjusts the learning speed automatically.

### Overfitting & Dropout
*   **Overfitting:** When the model memorizes the training images but fails on new images.
*   **Dropout:** A technique where we randomly switch off some neurons (e.g., 50%) during training. This forces the network to learn robust features and prevents reliance on specific neurons.
