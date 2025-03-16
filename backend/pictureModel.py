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


# Blueprint for prediction routes
picture_routes = Blueprint("picture_routes", __name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_PATH = "models/xgb_price_predict_model.pkl"
CLASSIFICATION_MODEL_PATH = "models/videoModel.joblib"
ENCODED_FEATURES_PATH = "encoded_features.json"
TRAINING_DATA_PATH = "datafile/updated_encoded.csv"

MODEL_PATH = "models/picture_model.h5"
SCALER_PATH = "picture_model_files/scaler.pkl"
ENCODER_PATH = "picture_model_files/encoders.pkl"
TARGET_SIZE = (128, 128)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the models
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)


#####################----------Image Preprocessing----------------------------############################

def remove_background(image_path):
    """Removes background from an image and ensures 3-channel RGB output."""
    try:
        input_image = Image.open(image_path)
        output_image = remove(input_image)  # Remove background
        return output_image.convert("RGB")  # Ensure 3-channel output
    except Exception as e:
        print(f"Error removing background: {e}")
        return None

def remove_watermark(image):
    """Placeholder for watermark removal (cropping example)."""
    image_array = np.array(image)
    height, width, _ = image_array.shape
    cropped_image = image_array[0:height-50, 0:width-50]  # Adjust crop
    return cropped_image

def load_and_preprocess_image(image_path):
    """Loads, preprocesses image, removes background & watermark."""
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} does not exist.")
        return None

    # Remove background (PIL-based, so no need to load image twice)
    img_rgb = remove_background(image_path)
    if img_rgb is None:
        return None

    # Convert PIL image to NumPy array (OpenCV-compatible)
    img_rgb = np.array(img_rgb)

    # Remove watermark
    img_rgb = remove_watermark(img_rgb)

    # Resize and normalize
    img_resized = cv2.resize(img_rgb, TARGET_SIZE).astype("float32") / 255.0

    return img_resized



######-------------------------------Clarity extraction for new image---------------------------############





def extract_clarity_features(gray, hsv):
    """Extract clarity-related features using texture analysis."""
    gray = np.uint8(gray * 255)

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    intensity_variance = np.var(gray)

    hue_std = np.std(hsv[:, :, 0])
    saturation_std = np.std(hsv[:, :, 1])

    return [contrast, homogeneity, energy, correlation, edge_density, intensity_variance, hue_std, saturation_std]



########---------------------------------Cut extraction for new image-----------------------################



def extract_cut_features(gray):
    """Extract cut-related features using contour analysis."""
    gray = np.uint8(gray * 255)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: No contours found.")
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area != 0 else 0
    edge_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    flipped_horizontal = cv2.flip(gray, 1)
    symmetry_horizontal = 1 - (np.mean(cv2.absdiff(gray, flipped_horizontal)) / 255)

    flipped_vertical = cv2.flip(gray, 0)
    symmetry_vertical = 1 - (np.mean(cv2.absdiff(gray, flipped_vertical)) / 255)

    symmetry = (symmetry_horizontal + symmetry_vertical) / 2

    return [aspect_ratio, perimeter, area, circularity, convexity, edge_sharpness, symmetry]




####-------------------------------Define predict_gemstone function--------------------------------######



def predict_gemstone(image_path):
    """Preprocess image, extract features, and predict gemstone quality."""
    img_array = load_and_preprocess_image(image_path)
    if img_array is None:
        print("Image preprocessing failed.")
        return

    img_cv2 = cv2.imread(image_path)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)

    # Extract features
    clarity_features = extract_clarity_features(gray, hsv)
    cut_features = extract_cut_features(gray)
    if clarity_features is None or cut_features is None:
        print("Feature extraction failed.")
        return

    # Combine extracted features
    tabular_features = np.array(clarity_features + cut_features).reshape(1, -1)

    # Scale tabular features
    tabular_features_scaled = scaler.transform(tabular_features)

    # Reshape image for model
    img_array = np.expand_dims(img_array, axis=0)  # (1, height, width, channels)

    # Make predictions
    predictions = model.predict([tabular_features_scaled, img_array])

    # Extract predicted class indices
    clarity_pred = np.argmax(predictions[0])
    cut_pred = np.argmax(predictions[1])
    shape_pred = np.argmax(predictions[2])

    # Convert indices to labels
    clarity_label = encoder["clarity"].categories_[0][clarity_pred]
    cut_label = encoder["cut"].categories_[0][cut_pred]
    shape_label = encoder["shape"].categories_[0][shape_pred]

    return {
        "clarity": clarity_label,
        "cut": cut_label,
        "shape": shape_label
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

    # Return predictions as JSON
    return jsonify({
        "message": "Processing complete",
        "predicted_attributes": predictions
    })





