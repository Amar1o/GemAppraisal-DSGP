from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import pandas as pd
import os
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Paths
DATA_FILE_PATH = r"C:\Users\Muralish\Desktop\DSGP\GemAppraisal-DSGP\notebook\Notebook-Norman\Dataset_Used\updated_final_combined_data.csv"
ATTRIBUTE_MODEL_FILE_PATH = r"C:\Users\Muralish\Desktop\DSGP\GemAppraisal-DSGP\notebook\Notebook-Norman\Model Training\rf_model.joblib"
PRICE_MODEL_FILE_PATH = "models/xgb_price_predict_model.pkl"
ENCODED_FEATURES_FILE = "encoded_features.json"

# Load models and encoded features
attribute_model = joblib.load(ATTRIBUTE_MODEL_FILE_PATH)
price_model = joblib.load(PRICE_MODEL_FILE_PATH)

with open(ENCODED_FEATURES_FILE, "r") as file:
    encoded_features = json.load(file)

# Store the latest classified attributes in memory
classified_results_cache = {}

@app.route('/price-calculator', methods=['POST'])
def price_calculator():
    """Process uploaded video file, extract ItemID, and classify gemstone attributes."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No video file uploaded.'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Empty file uploaded.'}), 400

        # Extract ItemID from filename (assumes format "ItemID.mp4")
        item_id, ext = os.path.splitext(file.filename)

        if ext.lower() != ".mp4":
            return jsonify({'error': 'Only MP4 files are allowed.'}), 400

        # Load dataset
        df = pd.read_csv(DATA_FILE_PATH)

        # Find matching row in dataset
        filtered_df = df[df["ItemID"].astype(str) == item_id]
        if filtered_df.empty:
            return jsonify({'error': f'No data found for ItemID: {item_id}'}), 404

        # Select feature columns (92 to 122)
        features = filtered_df.iloc[:, 92:122]

        # Make predictions
        predictions = attribute_model.predict(features)

        # Get target column names (assuming they were in columns 1-91)
        output_columns = df.columns[1:92]

        # Create a DataFrame for predictions
        predictions_df = pd.DataFrame(predictions, columns=output_columns)

        # Attach ItemID
        predictions_df.insert(0, "ItemID", item_id)

        # Extract best classification result
        best_classified_result = {"ItemID": item_id}
        for col in predictions_df.columns[1:]:
            if predictions_df[col].values[0] == 1:
                attribute_name, attribute_value = col.split("_", 1)
                best_classified_result[attribute_name] = attribute_value

        # Store in cache (overwrite existing value)
        classified_results_cache[item_id] = best_classified_result

        # Print extracted attributes for debugging
        print("\n✅ Extracted ItemID & Predicted Attributes:")
        print(json.dumps(best_classified_result, indent=4))

        return jsonify(best_classified_result)  # Send predicted values to frontend

    except Exception as e:
        app.logger.error(f"Error in price-calculator: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/get-attributes', methods=['GET'])
def get_attributes():
    """Retrieve gemstone attributes for a given ItemID from the stored cache instead of CSV"""
    try:
        item_id = request.args.get("itemID")  # Get ItemID from query params

        if not item_id:
            return jsonify({"error": "Missing ItemID parameter"}), 400

        # Check if attributes exist in cache
        if item_id not in classified_results_cache:
            return jsonify({"error": f"No cached data found for ItemID: {item_id}"}), 404

        # Retrieve from cache
        attributes = classified_results_cache[item_id]

        return jsonify(attributes)

    except Exception as e:
        app.logger.error(f"Error in get-attributes: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Predict gemstone price based on user-provided features."""
    try:
        data = request.form

        # Validate required fields
        required_fields = ['carat', 'cutQuality', 'shape', 'origin', 'color',
                           'colorIntensity', 'clarity', 'cut', 'treatment', 'type']
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400

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
        prediction = price_model.predict(model_input)[0]

        # Print predicted price for debugging
        print("\n💰 Predicted Price:", prediction)

        return jsonify({"price": float(prediction)})

    except ValueError as ve:
        return jsonify({"error": f"Input validation error: {str(ve)}"}), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("\n✅ Flask Backend is Running on http://127.0.0.1:5000/")
    app.run(debug=True)
