# 🎓 Viva Questions & Answers: Crop Disease Prediction

Read these questions until you are comfortable answering them naturally.

---

## 🟢 Basic Concepts

**Q1: What is the main objective of this module?**
**Answer:** To detect crop diseases accurately from leaf images using Deep Learning, enabling farmers to take timely action.

**Q2: Which algorithm did you use?**
**Answer:** I used a **Convolutional Neural Network (CNN)**, which is the state-of-the-art algorithm for image classification.

**Q3: What is the source of your dataset?**
**Answer:** I used the **PlantVillage dataset**, which is a standard open-source dataset for plant diseases.

**Q4: How many classes (diseases) can your model detect?**
**Answer:** It can detect **38 different classes**, including healthy and diseased leaves across 14 crop species.

**Q5: Why did you choose CNN over SVM or Random Forest?**
**Answer:** SVM and Random Forest lose spatial information when images are flattened. CNN preserves the spatial structure (shapes, textures) which is crucial for identifying visual disease patterns.

---

## 🟡 Technical Questions (Data & Preprocessing)

**Q6: What image size did you use?**
**Answer:** I resized all images to **224x224 pixels** to match the input requirement of the CNN architecture.

**Q7: Why do you normalize images (divide by 255)?**
**Answer:** To scale pixel values from the range 0-255 to 0-1. This helps the neural network converge (learn) faster and prevents exploding gradients.

**Q8: What is Data Augmentation?**
**Answer:** It is a technique to artificially increase the size of the training set by creating modified versions of images (rotating, flipping, zooming).

**Q9: Why is Data Augmentation important?**
**Answer:** It prevents **overfitting** and helps the model generalize better to real-world variations like different angles or orientations.

**Q10: How did you handle data imbalance?**
**Answer:** I used data augmentation to generate more samples for under-represented classes, ensuring the model doesn't become biased toward the majority class.

---

## 🟠 Technical Questions (CNN Architecture)

**Q11: Explain the layers in your CNN.**
**Answer:** It consists of **Convolutional layers** (to extract features), **Pooling layers** (to reduce size), **Flattening** (to convert to 1D), and **Dense layers** (for classification).

**Q12: What is the role of the Kernel/Filter?**
**Answer:** The kernel slides over the image to detect specific features like edges, curves, or textures.

**Q13: What does the ReLU activation function do?**
**Answer:** It introduces non-linearity by converting negative values to zero, allowing the model to learn complex patterns.

**Q14: Why do you use Max Pooling?**
**Answer:** To reduce the spatial dimensions (image size) while keeping the most important features. It reduces computation and controls overfitting.

**Q15: What is the function of the Softmax layer?**
**Answer:** It is the final output layer that converts the raw scores into probabilities. The class with the highest probability is the prediction.

**Q16: What is "Dropout"?**
**Answer:** Dropout is a regularization technique where we randomly turn off some neurons during training to prevent the model from memorizing the data (overfitting).

**Q17: Which optimizer did you use?**
**Answer:** I used the **Adam** optimizer because it handles sparse gradients well and adjusts the learning rate adaptively.

**Q18: What loss function did you use?**
**Answer:** **Categorical Cross-Entropy**, which is the standard loss function for multi-class classification problems.

---

## 🔴 Evaluation & Performance

**Q19: What is the accuracy of your model?**
**Answer:** (Check your training logs, typically:) "The model achieved a training accuracy of roughly 95% and a validation accuracy of around 92%."

**Q20: What is a Confusion Matrix?**
**Answer:** A table that shows how many predictions were correct and how many were wrong for each class. It helps identify which diseases are being confused with each other.

**Q21: What is the difference between Precision and Recall?**
**Answer:**
*   **Precision:** Out of all images predicted as "Disease X", how many were actually "Disease X"?
*   **Recall:** Out of all actual "Disease X" images, how many did the model correctly find?

**Q22: What is Overfitting?**
**Answer:** When the model performs excellent on training data but poor on new/test data. It means the model "memorized" the training set.

**Q23: How do you prevent Overfitting?**
**Answer:** By using **Dropout**, **Data Augmentation**, and **Max Pooling**.

---

## 🔵 Real-World & Future Scope

**Q24: What happens if I upload a picture of a dog?**
**Answer:** The model will try to classify it into one of the 38 plant classes (likely with low confidence) because it hasn't been trained to recognize "Not a Plant".

**Q25: How does this help a farmer?**
**Answer:** It provides instant, expert-level disease diagnosis without needing to wait for an agricultural officer, saving time and crops.

**Q26: Does it work offline?**
**Answer:** Currently, it requires an internet connection to send the image to the server, but the model *could* be deployed to a mobile app for offline use in the future.

**Q27: What are the limitations?**
**Answer:** It relies on image quality. Poor lighting or blurry images can lead to wrong predictions. It can only detect the 38 diseases it was trained on.

**Q28: How can you improve this model?**
**Answer:** By training on a larger, more diverse dataset (images taken in real fields, not labs) and using a more complex architecture like ResNet or MobileNet.

**Q29: How is the confidence score calculated?**
**Answer:** It is the highest probability value from the Softmax output layer (multiplied by 100).

**Q30: Can this detect multiple diseases on one leaf?**
**Answer:** My current model is a "Multi-Class" classifier (one label per image). To detect multiple diseases, we would need "Multi-Label" classification or Object Detection (YOLO).

---

## 🟣 Implementation Details (Important for Viva)

**Q31: I see 'opencv_disease.py' in your code. Are you using OpenCV or CNN?**
**Answer:**
*   **Primary Model:** My project is designed to use **CNN (Deep Learning)** because it provides the highest accuracy for image classification. The code for this is in `backend/utils/cnn_model.py`.
*   **Fallback:** I also included an **OpenCV + Random Forest** version (`opencv_disease.py`) as a lightweight fallback.
*   **Reason:** This allows the application to run on devices without a GPU or if TensorFlow has compatibility issues on a specific server.
*   **For Viva:** I am presenting the **CNN approach** as it is the core of my research and the superior technology.
