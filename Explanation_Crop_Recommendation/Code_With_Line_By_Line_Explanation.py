import pandas as pd  # For handling the dataset (loading CSV, manipulating tables)
import numpy as np   # For numerical operations
import matplotlib.pyplot as plt # For plotting graphs (optional, for visualization)
import seaborn as sns # For making beautiful charts (optional)
from sklearn.model_selection import train_test_split # To split data into training and testing sets
from sklearn.preprocessing import LabelEncoder, StandardScaler # For encoding text labels and scaling numbers
from sklearn.ensemble import RandomForestClassifier # The Machine Learning Algorithm we are using
from sklearn.metrics import accuracy_score, classification_report # To measure how good our model is

# ==========================================
# STEP 1: LOAD THE DATASET
# ==========================================
# We load the data from a CSV file into a pandas DataFrame (like an Excel sheet in Python)
# 'df' stands for DataFrame
df = pd.read_csv('Crop_recommendation.csv')

# Show the first 5 rows to check if data loaded correctly
print("First 5 rows of data:")
print(df.head())

# ==========================================
# STEP 2: PREPROCESSING (Preparing Data)
# ==========================================

# 2.1 Separate Features (Inputs) and Target (Output)
# X contains all columns EXCEPT 'label'. These are inputs: N, P, K, temp, etc.
X = df.drop('label', axis=1)

# y contains ONLY the 'label' column. This is what we want to predict (Crop Name).
y = df['label']

# 2.2 Label Encoding
# Machines don't understand text like 'Rice' or 'Maize'. They understand numbers.
# LabelEncoder converts 'Rice' -> 0, 'Maize' -> 1, etc.
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Print the mapping to show what crop became what number
print("Classes found:", le.classes_)

# ==========================================
# STEP 3: TRAIN-TEST SPLIT
# ==========================================
# We split our data into two parts:
# 1. Training Set (80%): To teach the model.
# 2. Testing Set (20%): To test the model on data it has NEVER seen before.
# random_state=42 ensures we get the same split every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# ==========================================
# STEP 4: FEATURE SCALING
# ==========================================
# Why? Rainfall is ~200, pH is ~6. The model might think Rainfall is more important just because the number is big.
# StandardScaler adjusts all values so they contribute equally.
scaler = StandardScaler()

# Fit on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the SAME scaler (do not fit again!)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# STEP 5: MODEL TRAINING (Random Forest)
# ==========================================
# We use Random Forest Classifier.
# Why? It is accurate, handles multiple features well, and doesn't overfit easily.
# n_estimators=100 means we are building 100 decision trees.
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Training the model... (this learns the patterns)")
model.fit(X_train_scaled, y_train)

# ==========================================
# STEP 6: PREDICTION & EVALUATION
# ==========================================
# Now we ask the model to predict crop names for the Test Data (which it hasn't seen)
y_pred = model.predict(X_test_scaled)

# Calculate Accuracy: (Correct Predictions / Total Predictions) * 100
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Detailed Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ==========================================
# STEP 7: MAKING A REAL PREDICTION
# ==========================================
# Let's say a farmer comes with this soil data:
# N=90, P=42, K=43, Temp=20, Hum=82, pH=6, Rain=200
new_data = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])

# We must scale this new data just like we scaled the training data
new_data_scaled = scaler.transform(new_data)

# Predict
prediction_index = model.predict(new_data_scaled) # Returns a number (e.g., 20)
predicted_crop = le.inverse_transform(prediction_index) # Converts 20 -> 'Rice'

print(f"\nExample Prediction:")
print(f"Input: {new_data}")
print(f"Recommended Crop: {predicted_crop[0]}")
