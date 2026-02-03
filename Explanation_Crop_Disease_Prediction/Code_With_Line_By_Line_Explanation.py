# ==========================================
# CROP DISEASE PREDICTION SYSTEM (CNN)
# ==========================================
# This script contains the complete code for training a Convolutional Neural Network (CNN)
# to detect plant diseases from leaf images.
#
# KEY CONCEPTS:
# 1. Image Preprocessing (Resizing, Normalization)
# 2. Data Augmentation (Creating variations of images)
# 3. CNN Architecture (Conv2D, MaxPooling, Dense Layers)
# 4. Training & Validation
# 5. Saving the Model

import tensorflow as tf  # The main Deep Learning library
from tensorflow.keras.models import Sequential  # To build the model layer by layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For loading & augmenting images
import matplotlib.pyplot as plt  # For plotting accuracy graphs
import numpy as np  # For numerical operations

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Define the size of images. CNN needs a fixed input size.
# 224x224 is a standard size for models like VGG/ResNet.
IMAGE_SIZE = (224, 224)

# Batch size: Number of images processed at once.
# 32 is a common choice (balance between speed and memory).
BATCH_SIZE = 32

# Path to your dataset folder (Change this to your actual path)
DATASET_DIR = 'dataset'

# ==========================================
# 2. DATA PREPROCESSING & AUGMENTATION
# ==========================================
# ImageDataGenerator is a powerful tool to load images and apply transformations.
# Rescale=1./255: Normalizes pixel values from 0-255 to 0-1. 
# This helps the Neural Network learn faster and more stable.

train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normalize pixels to 0-1
    rotation_range=20,          # Randomly rotate image by 20 degrees
    width_shift_range=0.2,      # Shift width by 20%
    height_shift_range=0.2,     # Shift height by 20%
    horizontal_flip=True,       # Flip image horizontally
    validation_split=0.2        # Use 20% of data for validation (testing during training)
)

# Load Training Data (80%)
print("Loading Training Data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,     # Resize all images to 224x224
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # 'categorical' because we have >2 classes (multi-class)
    subset='training'           # Select the training subset
)

# Load Validation Data (20%)
print("Loading Validation Data...")
validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'         # Select the validation subset
)

# ==========================================
# 3. BUILDING THE CNN MODEL
# ==========================================
# We use 'Sequential' to stack layers one after another.

model = Sequential()

# --- First Convolutional Block ---
# Conv2D: Extracts features (edges, textures).
# 32 filters: The model learns 32 different feature maps.
# (3, 3): Size of the filter kernel (3x3 pixels).
# Activation='relu': Introduces non-linearity (helps learn complex patterns).
# Input_shape: (224, 224, 3) -> Height, Width, RGB Channels.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))

# MaxPooling2D: Reduces image size by half (2x2).
# Keeps the most important features, discards details. Reduces computation.
model.add(MaxPooling2D(pool_size=(2, 2)))

# --- Second Convolutional Block ---
# We increase filters to 64 to learn more complex features (shapes, patterns).
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# --- Third Convolutional Block ---
# Increase to 128 filters.
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# --- Flattening ---
# Converts the 2D matrix (features) into a 1D vector (list of numbers).
# Necessary to connect to the Dense (Fully Connected) layers.
model.add(Flatten())

# --- Fully Connected Layers ---
# Dense(256): A layer with 256 neurons. It combines all features to make decisions.
model.add(Dense(256, activation='relu'))

# Dropout(0.5): Randomly turns off 50% of neurons during training.
# Why? Prevents Overfitting (memorizing the data).
model.add(Dropout(0.5))

# Output Layer
# Dense(num_classes): One neuron for each disease class.
# Activation='softmax': Converts output into probabilities (e.g., Apple_Scab: 80%, Healthy: 20%).
num_classes = train_generator.num_classes
model.add(Dense(num_classes, activation='softmax'))

# ==========================================
# 4. COMPILING THE MODEL
# ==========================================
# Optimizer='adam': Adapts the learning rate automatically. Best general-purpose optimizer.
# Loss='categorical_crossentropy': Standard loss function for multi-class classification.
# Metrics=['accuracy']: We want to track how accurate the model is.

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print Model Summary (Structure)
model.summary()

# ==========================================
# 5. TRAINING THE MODEL
# ==========================================
# epochs=10: Go through the entire dataset 10 times.
# validation_data: Check performance on unseen data after each epoch.

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=10
)

# ==========================================
# 6. SAVING THE MODEL
# ==========================================
# Save the trained model to a file. We can load this later for the website.
model.save('plant_disease_model.h5')
print("Model saved successfully as plant_disease_model.h5")

# ==========================================
# 7. PREDICTION (SINGLE IMAGE)
# ==========================================
# This function mimics what happens when a user uploads an image.

def predict_image(image_path):
    from tensorflow.keras.preprocessing import image
    
    # Load image and resize to 224x224 (same as training)
    img = image.load_img(image_path, target_size=(224, 224))
    
    # Convert image to numpy array
    img_array = image.img_to_array(img)
    
    # Add batch dimension (1, 224, 224, 3) because model expects a batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize (0-1)
    img_array /= 255.0
    
    # Predict
    predictions = model.predict(img_array)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(predictions)
    
    # Get the class name from the generator
    class_map = train_generator.class_indices
    # Invert map to get name from index
    class_names = {v: k for k, v in class_map.items()}
    
    result = class_names[predicted_index]
    confidence = np.max(predictions) * 100
    
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
    return result

# Example Usage (Uncomment to test)
# predict_image('test_leaf.jpg')
