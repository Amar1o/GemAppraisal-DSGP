from flask import Blueprint, request, jsonify
import os
import numpy as np
import joblib
import json
import psycopg2
from datetime import datetime
from supabase_client import supabase

# Blueprint for prediction routes
prediction_routes = Blueprint("prediction_routes", __name__)

# Paths
MODEL_PATH = "models/xgb_price_predict_model.pkl"
ENCODED_FEATURES_PATH = "encoded_features.json"

# Load model
price_model = joblib.load(MODEL_PATH)


@prediction_routes.route("/", methods=["POST"])
def predict():
    try:
        data = request.form
        user_id = data.get("user_id")  # Ensure user ID is sent from the frontend
        media_url = data.get("media_url", None)  # Optional media URL

        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # Log received form data
        print(" Received form data:", data)

        # Validate required fields
        required_fields = ['carat', 'cutQuality', 'shape', 'origin', 'color',
                           'colorIntensity', 'clarity', 'cut', 'treatment', 'type']
        missing_fields = [field for field in required_fields if not data.get(field)]

        if missing_fields:
            print(" Missing fields:", missing_fields)
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Load encoded features from JSON file
        if not os.path.exists(ENCODED_FEATURES_PATH):
            print(" Error: Encoded features file not found.")
            return jsonify({"error": "Encoded features mapping file not found."}), 500

        with open(ENCODED_FEATURES_PATH, "r") as f:
            encoded_features = json.load(f)

        # Extract & encode form inputs safely
        try:
            carat = float(data.get("carat"))

            cut_quality_encoded = float(encoded_features["cut_quality_mapping"].get(data.get("cutQuality"), -1))
            shape_encoded = float(encoded_features["shape_mapping"].get(data.get("shape"), -1))
            origin_encoded = float(encoded_features["origin_mapping"].get(data.get("origin"), -1))
            color_encoded = float(encoded_features["color_mapping"].get(data.get("color"), -1))
            color_intensity_encoded = float(
                encoded_features["color_intensity_mapping"].get(data.get("colorIntensity"), -1))
            clarity_encoded = float(encoded_features["clarity_mapping"].get(data.get("clarity"), -1))
            cut_encoded = float(encoded_features["cut_mapping"].get(data.get("cut"), -1))
            treatment_encoded = float(encoded_features["treatment_mapping"].get(data.get("treatment"), -1))
            type_encoded = float(encoded_features["type_mapping"].get(data.get("type"), -1))

            # Log encoded values
            print(" Encoded values:", {
                "carat": carat,
                "cut_quality": cut_quality_encoded,
                "shape": shape_encoded,
                "origin": origin_encoded,
                "color": color_encoded,
                "color_intensity": color_intensity_encoded,
                "clarity": clarity_encoded,
                "cut": cut_encoded,
                "treatment": treatment_encoded,
                "type": type_encoded,
            })

            # Ensure valid encoding values
            if -1 in [cut_quality_encoded, shape_encoded, origin_encoded, color_encoded,
                      color_intensity_encoded, clarity_encoded, cut_encoded,
                      treatment_encoded, type_encoded]:
                print(" Invalid selection in dropdown values.")
                return jsonify({"error": "Invalid selection in dropdown values. Ensure all inputs are correct."}), 400

        except ValueError as e:
            print(" ValueError:", e)
            return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400

        # Prepare model input for price prediction
        model_input = np.array([[carat, cut_quality_encoded, shape_encoded, origin_encoded,
                                 color_encoded, color_intensity_encoded, clarity_encoded,
                                 cut_encoded, treatment_encoded, type_encoded]])

        # Log model input
        print(" Model input array:", model_input)

        # Ensure price model is loaded
        if price_model is None:
            print(" Error: Price prediction model not found.")
            return jsonify({"error": "Price prediction model is not loaded."}), 500

        # Make prediction
        predicted_price = price_model.predict(model_input)[0]
        predicted_price = np.expm1(predicted_price)  # Convert back from log-transformed price

        # Save appraisal to Supabase
        new_appraisal = {
            "user_id": user_id,
            "gem_species": data.get("type"),
            "shape": data.get("shape"),
            "color": data.get("color"),
            "color_intensity": data.get("colorIntensity"),
            "cut": data.get("cut"),
            "cut_quality": data.get("cutQuality"),
            "carat": carat,
            "origin": data.get("origin"),
            "treatment": data.get("treatment"),
            "estimated_price": round(float(predicted_price), 2),
            "gem_species_accuracy": 0.5,  # Placeholder for now
            "created_at": datetime.now().isoformat(),
            "media_url": media_url
        }

        # Check for existing appraisal
        existing_appraisal = (
            supabase.table("appraisals")
            .select("*")
            .eq("user_id", user_id)
            .eq("gem_species", data.get("type"))
            .eq("shape", data.get("shape"))
            .eq("color", data.get("color"))
            .eq("color_intensity", data.get("colorIntensity"))
            .eq("cut", data.get("cut"))
            .eq("cut_quality", data.get("cutQuality"))
            .eq("carat", carat)
            .eq("origin", data.get("origin"))
            .eq("treatment", data.get("treatment"))
            .execute()
        )
        if existing_appraisal.data:
            print("Duplicate entry found, returning existing appraisal.")
            return jsonify({"price": round(float(predicted_price), 2)})

        response = supabase.table("appraisals").insert(new_appraisal).execute()

        # Log the final price
        print(" Predicted Price:", predicted_price)

        return jsonify({"price": round(float(predicted_price), 2)})

    except Exception as e:
        print(" Prediction error:", str(e))
        return jsonify({"error": "Internal server error"}), 500
