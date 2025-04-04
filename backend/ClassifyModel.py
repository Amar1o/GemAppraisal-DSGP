from flask import Blueprint, request, jsonify, current_app
import os
import cv2
import numpy as np
import joblib
import pickle
from rembg import remove
from PIL import Image

# Blueprint for prediction routes
image_routes = Blueprint("image_routes", __name__)

MODEL_PATH = "models/sapphire_model2.pkl"
CLASS_NAMES = ["green_sapphire", "blue_sapphire", "non_sapphire"]

# Load the trained model
with open(MODEL_PATH, 'rb') as model_file:
    classification_model = pickle.load(model_file)


def extract_color_features(image_path):
    """
    Extract color features from the image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")

    # Remove background
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    try:
        img_no_bg = remove(pil_img)
        img_no_bg = np.array(img_no_bg)
    except Exception as e:
        raise ValueError(f"Failed to remove background: {e}")

    rgb_image = cv2.cvtColor(img_no_bg, cv2.COLOR_RGBA2RGB)

    # Color histograms
    r_hist, _ = np.histogram(rgb_image[:, :, 0], bins=256, range=(0, 256))
    g_hist, _ = np.histogram(rgb_image[:, :, 1], bins=256, range=(0, 256))
    b_hist, _ = np.histogram(rgb_image[:, :, 2], bins=256, range=(0, 256))

    r_hist = r_hist / np.sum(r_hist)
    g_hist = g_hist / np.sum(g_hist)
    b_hist = b_hist / np.sum(b_hist)

    # Average color values
    avg_r = np.mean(rgb_image[:, :, 0])
    avg_g = np.mean(rgb_image[:, :, 1])
    avg_b = np.mean(rgb_image[:, :, 2])

    return np.concatenate([r_hist, g_hist, b_hist, [avg_r, avg_g, avg_b]])


def predict_class(image_path, model, class_names):
    """
    Predict the class of a given image using a trained model.
    """
    test_image_features = extract_color_features(image_path)
    if test_image_features is None:
        return None, None

    test_image_features = test_image_features.reshape(1, -1)

    predicted_class_index = model.predict(test_image_features)[0]
    predicted_class = class_names[predicted_class_index]
    predicted_probabilities = model.predict_proba(test_image_features)[0].tolist()

    return predicted_class, predicted_probabilities


@image_routes.route("/classify", methods=["POST"])
def classify():
    try:
        current_app.logger.debug(f"Received data: {request.form}")
        current_app.logger.debug(f"Received files: {request.files}")

        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        image_path = 'temp_image.jpg'
        file.save(image_path)

        predicted_class, predicted_probabilities = predict_class(image_path, classification_model, CLASS_NAMES)

        if predicted_class is None:
            return jsonify({"error": "Prediction failed"}), 500

        class_parts = predicted_class.split("_")
        color = class_parts[0].capitalize() if len(class_parts) > 1 else "Unknown"

        predicted_attributes = {"color": color}

        current_app.logger.debug(f"Predicted Class: {predicted_class}")
        current_app.logger.debug(f"Class Probabilities: {predicted_probabilities}")
        current_app.logger.debug(f"Predicted Attributes: {predicted_attributes}")

        return jsonify({
            "predicted_class": predicted_class,
            "probabilities": predicted_probabilities,
            "attributes": predicted_attributes
        })

    except ValueError as ve:
        return jsonify({"error": f"Input validation error: {str(ve)}"}), 400
    except Exception as e:
        current_app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500