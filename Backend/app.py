from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import xgboost as xgb
import json
import logging
logging.basicConfig(level=logging.DEBUG)
import cv2
from rembg import remove
import os
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allows all origins

# Load the trained model
model_pat = joblib.load(r"models\Classifysapphire.pkl")

# Load the JSON file
with open("encoded_features.json", "r") as file:
    encoded_features = json.load(file)



def extract_color_histogram(image):
    """
    Extracts color histogram features from the image for classification.
    :param image_path: Path to the image.
    :return: A flattened, normalized histogram feature array.
    """
    image = cv2.resize(image, (128, 128))  # Resize to match the training size
    hist_features = []

    # Extract histograms for R, G, and B channels
    for i in range(3):  # BGR channels
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten
        hist_features.extend(hist)

    return np.array(hist_features)


def predict_class(image_path, model):
    """
    Predict the class of the given image using a trained model.
    :param image_path: Path to the image.
    :param model: Loaded trained model.
    :return: Predicted class and class probabilities.
    """
    app.logger.debug(f"Using model for prediction: {model}")

    # Read the image in binary mode
    with open(image_path, "rb") as image_file:
        input_image = image_file.read()

    # Remove background
    try:
        final_image_data = remove(input_image)
    except Exception as e:
        raise ValueError(f"Error removing background: {str(e)}")

    # Define save path (ensure folder exists)
    save_folder = r"D:\notebook\Notebooks-Amar\backend\images"
    os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist
    save_path = os.path.join(save_folder, "processed_image.jpg")

    # Convert background-removed image to OpenCV format
    np_array = np.frombuffer(final_image_data, np.uint8)
    final_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if final_image is None:
        raise ValueError("Error decoding the image after background removal.")

    # Save the processed image
    cv2.imwrite(save_path, final_image)
    app.logger.debug(f"Processed image saved at: {save_path}")

    # Extract color histogram features from the processed image
    test_image_features = extract_color_histogram(final_image)
    test_image_features = test_image_features.reshape(1, -1)

    # Predict the class
    predicted_class = model.predict(test_image_features)
    predicted_probabilities = model.predict_proba(test_image_features)

    app.logger.debug(f"Predicted Class: {predicted_class[0]}")
    
    # return predicted_class[0], predicted_probabilities
    return str(predicted_class[0]), predicted_probabilities.tolist()

@app.route("/classify", methods=["POST"])
def classify():
    try:
        model = joblib.load(r"Classifysapphire.pkl")
        data = request.form
        app.logger.debug(f"Received data: {data}")
        app.logger.debug(f"Received data: {request.form}")
        app.logger.debug(f"Received files: {request.files}")
        # Validate required fields
        # required_fields = ['carat', 'cutQuality', 'shape', 'origin', 'color',
        #                    'colorIntensity', 'clarity', 'cut', 'treatment', 'type']
        # for field in required_fields:
        #     if not data.get(field):
        #         return jsonify({"error": f"Missing required field: {field}"}), 400

        predicted_class, predicted_probabilities = None, None  # Default values
        predicted_attributes = {}  # Dictionary to store extracted attributes

        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                image_path = 'temp_image.jpg'
                file.save(image_path)

                # Run classification for the uploaded image
                predicted_class, predicted_probabilities = predict_class(image_path, model)

                # Extract color dynamically from predicted class
                class_parts = predicted_class.split("_")  # Splitting on underscore
                if len(class_parts) > 1:
                    color = class_parts[0].capitalize()  # First part is color
                else:
                    color = "Unknown"  # Fallback in case no color is found

                predicted_attributes = {
                    "color": color
                }

                app.logger.debug(f"Predicted Class: {predicted_class}")
                app.logger.debug(f"Class Probabilities: {predicted_probabilities}")
                app.logger.debug(f"Predicted Attributes: {predicted_attributes}")

        return jsonify({
            "predicted_class": predicted_class,
            # "probabilities": predicted_probabilities.tolist() if predicted_probabilities else None,
            "probabilities": predicted_probabilities.tolist() if isinstance(predicted_probabilities, np.ndarray) else predicted_probabilities,

            "attributes": predicted_attributes  # Sending extracted color
        })

    except ValueError as ve:
        return jsonify({"error": f"Input validation error: {str(ve)}"}), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        
        # Validate required fields
        required_fields = ['carat', 'cutQuality', 'shape', 'origin', 'color',
                         'colorIntensity', 'clarity', 'cut', 'treatment', 'type']
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Add your file processing logic here
                pass

        # Convert parameters with error handling
        try:
            carat = float(data.get("carat"))
            cut_quality_encoded = float(encoded_features["cut_quality_mapping"][data.get("cutQuality")])
            shape_encoded = float(encoded_features["shape_mapping"][data.get("shape")])
            origin_encoded = float(encoded_features["origin_mapping"][data.get("origin")])
            color_encoded = float(encoded_features["color_mapping"][data.get("color")])
            color_intensity_encoded = float(encoded_features["color_intensity_mapping"][data.get("colorIntensity")])
            clarity_encoded = float(encoded_features["clarity_mapping"][data.get("clarity")])
            cut_encoded = float(encoded_features["cut_mapping"][data.get("cut")])
            treatment_encoded = float(encoded_features["treatment_mapping"][data.get("treatment")])
            type_encoded = float(encoded_features["type_mapping"][data.get("type")])
        except ValueError as e:
            return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400

        # Create model input for price prediction
        model_input = np.array([[carat, cut_quality_encoded, shape_encoded, origin_encoded,
                               color_encoded, color_intensity_encoded, clarity_encoded,
                               cut_encoded, treatment_encoded, type_encoded]])

        # Make prediction for price
        prediction = np.expm1(model.predict(model_input)[0])

        return jsonify({"price": float(prediction)})

    except ValueError as ve:
        return jsonify({"error": f"Input validation error: {str(ve)}"}), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500




if __name__ == "__main__":
    app.run(debug=True)
