from flask import Blueprint, request, jsonify, current_app
import os
import cv2
import numpy as np
import joblib
from rembg import remove

# Blueprint for prediction routes
image_routes = Blueprint("image_routes", __name__)

MODEL_PATH = "models/sapphire_modelbalanced.pkl"

# Load the trained model
classification_model = joblib.load(MODEL_PATH)

def extract_color_histogram(image):
    image = cv2.resize(image, (128, 128))  # Resize to match the training size
    hist_features = []
    for i in range(3):  # BGR channels
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()  
        hist_features.extend(hist)
    return np.array(hist_features)

def predict_class(image_path, model):
    current_app.logger.debug(f"Using model for prediction: {model}")

    with open(image_path, "rb") as image_file:
        input_image = image_file.read()

    try:
        final_image_data = remove(input_image)
    except Exception as e:
        raise ValueError(f"Error removing background: {str(e)}")

    save_folder = r"backend/images"
    os.makedirs(save_folder, exist_ok=True)  
    save_path = os.path.join(save_folder, "processed_image.jpg")

    np_array = np.frombuffer(final_image_data, np.uint8)
    final_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if final_image is None:
        raise ValueError("Error decoding the image after background removal.")

    cv2.imwrite(save_path, final_image)
    current_app.logger.debug(f"Processed image saved at: {save_path}")

    test_image_features = extract_color_histogram(final_image).reshape(1, -1)
    predicted_class = model.predict(test_image_features)
    predicted_probabilities = model.predict_proba(test_image_features)

    return str(predicted_class[0]), predicted_probabilities.tolist()

@image_routes.route("/classify", methods=["POST"])
def classify():
    try:
        current_app.logger.debug(f"Received data: {request.form}")
        current_app.logger.debug(f"Received files: {request.files}")

        predicted_class, predicted_probabilities = None, None  
        predicted_attributes = {}

        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                image_path = 'temp_image.jpg'
                file.save(image_path)
                predicted_class, predicted_probabilities = predict_class(image_path, classification_model)

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
