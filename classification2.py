import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import os

# Load and preprocess images
def load_images(image_folder, image_size=(128, 128)):
    image_data = []
    labels = []
    for label, image_subfolder in enumerate(os.listdir(image_folder)):
        subfolder_path = os.path.join(image_folder, image_subfolder)
        if os.path.isdir(subfolder_path):
            for image_file in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, image_size)
                image_data.append(image)
                labels.append(label)
    
    return np.array(image_data), np.array(labels)

# Load your images
image_folder = '/content/images/'  # Example folder
X, y = load_images(image_folder)

# Normalize images (values between 0 and 1)
X = X / 255.0

# Build a CNN model
# Define the model
model = models.Sequential([
    # Convolutional Layer 1: Extract features from the image
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  
    layers.MaxPooling2D((2, 2)),  # Max pooling to downsample the image
    
    # Convolutional Layer 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),  # Max pooling to downsample again
    
    # Convolutional Layer 3
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten the feature maps into a 1D vector
    layers.Flatten(),
    
    # Fully connected layer with 64 neurons
    layers.Dense(64, activation='relu'),
    
    # Output layer: Single neuron for binary classification
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification (0 or 1)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary crossentropy for binary classification
              metrics=['accuracy'])

# Summary of the model
model.summary()
# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# To test with a new image
def test_image(model, test_image_path, image_size=(128, 128)):
    image = cv2.imread(test_image_path)
    image = cv2.resize(image, image_size)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize
    
    prediction = model.predict(image)
    return np.argmax(prediction)
