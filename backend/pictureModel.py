import os
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from rembg import remove  
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from flask import Blueprint, request, jsonify


import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler  
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array, load_img 

# Blueprint for prediction routes
picture_routes = Blueprint("picture_routes", __name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_PATH = "models/xgb_price_predict_model.pkl"
CLASSIFICATION_MODEL_PATH = "models/videoModel.joblib"
ENCODED_FEATURES_PATH = "encoded_features.json"
TRAINING_DATA_PATH = "datafile/updated_encoded.csv"

 

SHAPE_MODEL_PATH ="models/Quality picture shape model.h5"
CUT_MODEL_PATH = "models/Quality picture cut model.h5"
CLARITY_MODEL_PATH = "models/Quality picture clarity model.h5"

 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the models
shape_model = tf.keras.models.load_model(SHAPE_MODEL_PATH)
cut_model = tf.keras.models.load_model(CUT_MODEL_PATH)
clarity_model = tf.keras.models.load_model(CLARITY_MODEL_PATH)



# ==========================
# Define Class Labels (Ensure Correct Order)
# ==========================
shape_labels = [
    "Asscher - Octagon", "Cushion", "Emerald Cut", "Fancy", "Heart",
    "Marquise", "Oval", "Pear", "Princess", "Radiant",
    "Round", "Trillion"
]

cut_labels = [
    "Asscher", "Brilliant", "Emerald Cut", "Fancy Brilliant", "Fancy Cut",
    "Mixed Brilliant", "Modified Brilliant", "Princess Cut", "Radiant Cut",
    "Scissor Cut", "Step Cut", "Trillion Cut"
]

clarity_labels = ["Eye Clean", "Included", "Slightly Included", "Very Slightly Included"]




#####################----------Image Preprocessing----------------------------############################

def preprocess_shape_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_cut_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def preprocess_clarity_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img




 

####-------------------------------Define predict_gemstone function--------------------------------######
def predict_shape(image_path):
    try:
        img = preprocess_shape_image(image_path)
        predictions = shape_model.predict(img)
        predicted_index = np.argmax(predictions, axis=1)[0]
        return shape_labels[predicted_index]
    except Exception as e:
        print(f"Error in shape prediction: {e}")
        return None


def predict_cut(image_path):
    try:
        img = preprocess_cut_image(image_path)
        predictions = cut_model.predict(img)
        predicted_index = np.argmax(predictions, axis=1)[0]
        return cut_labels[predicted_index]
    except Exception as e:
        print(f"Error in cut prediction: {e}")
        return None



def predict_clarity(image_path):
    try:
        img = preprocess_clarity_image(image_path)
        predictions = clarity_model.predict(img)
        predicted_index = np.argmax(predictions, axis=1)[0]
        return clarity_labels[predicted_index]
    except Exception as e:
        print(f"Error in clarity prediction: {e}")
        return None

#-------------------------------Define main prediction function---------------------------------------------##
def predict_gemstone(image_path):
    """Predict shape, cut, and clarity for a gemstone image."""
    shape = predict_shape(image_path)
    cut = predict_cut(image_path)
    clarity = predict_clarity(image_path)

    return {
        "shape": shape,
        "cut": cut,
        "clarity": clarity,
    }



#####--------------------------------Define Flask Route for Image Upload and Prediction-------------###########



@picture_routes.route("/upload_image", methods=["POST"])
def upload_image():
    """API endpoint to process an image and predict gemstone attributes."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Predict gemstone attributes
    predictions = predict_gemstone(image_path)
    if not predictions:
        return jsonify({"error": "Failed to process image"}), 500
    
    print(" Predictions:", predictions)  # Print predictions for debugging

    # Return predictions as JSON
    return jsonify({
        "message": "Processing complete",
        "predicted_attributes": predictions
    })