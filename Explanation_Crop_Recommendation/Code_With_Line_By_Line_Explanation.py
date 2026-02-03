# ==========================================
# CROP RECOMMENDATION SYSTEM (Random Forest)
# ==========================================
# This script demonstrates the complete pipeline for training the Crop Recommendation Model.
#
# KEY STEPS:
# 1. Load Dataset (Soil & Climate parameters)
# 2. Data Preprocessing (Cleaning, Label Encoding)
# 3. Feature Scaling (StandardScaler - CRITICAL STEP)
# 4. Model Training (Random Forest Classifier)
# 5. Model Evaluation (Accuracy, Confusion Matrix)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # For saving the trained model

# ==========================================
# 1. LOAD DATASET
# ==========================================
# The dataset contains agricultural parameters for 22 different crops.
# Path: F:\Major project\Crop_recommendation.csv
df = pd.read_csv('Crop_recommendation.csv')

# Display first 5 rows
print("Dataset Head:")
print(df.head())

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
# Separate Features (Inputs) and Target (Output)
X = df.drop('label', axis=1)  # N, P, K, temperature, humidity, ph, rainfall
y = df['label']               # Crop Name (e.g., Rice, Maize)

# Encode Labels
# The model needs numbers, not words. LabelEncoder converts "Rice" -> 1, "Maize" -> 2, etc.
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ==========================================
# 3. FEATURE SCALING (StandardScaler)
# ==========================================
# Why Scaling?
# Some features have large values (Rainfall ~ 200) and others small (pH ~ 6).
# Scaling ensures all features contribute equally to the result.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use in the website
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

# ==========================================
# 4. MODEL TRAINING (Random Forest)
# ==========================================
# We use Random Forest because it is robust, accurate, and handles non-linear relationships well.
# n_estimators=100: We build 100 decision trees.
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

print("Model Training Completed.")

# ==========================================
# 5. EVALUATION
# ==========================================
# Predict on the test set
y_pred = rf_classifier.predict(X_test_scaled)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# ==========================================
# 6. SAVING THE MODEL
# ==========================================
# Save the trained model and label encoder for the Flask App
joblib.dump(rf_classifier, 'crop_recommendation_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Model and Label Encoder saved successfully.")
