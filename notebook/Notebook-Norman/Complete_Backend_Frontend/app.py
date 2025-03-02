from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import pandas as pd
import joblib

from PIL import Image
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allows all origins

# Define paths
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_PATH = "models/xgb_price_predict_model.pkl"
CLASSIFICATION_MODEL_PATH = "models/rf_model.joblib"
ENCODED_FEATURES_PATH = "encoded_features.json"
TRAINING_DATA_PATH = "models/updated_encoded.csv"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the models
price_model = joblib.load(MODEL_PATH)
classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
training_data = pd.read_csv(TRAINING_DATA_PATH)

# Extract target column names (from index 1 to 92, representing B to CN)
target_columns = training_data.columns[1:92].tolist()


# ---------------------- Video Processing and Feature Extraction ----------------------

def extract_frames(video_path, output_folder, frames_per_second=5):
    """Extracts frames from a video at a specified frame rate."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second) if fps > 0 else 1

    frame_count = 0
    success, frame = cap.read()
    os.makedirs(output_folder, exist_ok=True)

    while success:
        if frame_count % frame_interval == 0:
            frame_file = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_file, frame)
        success, frame = cap.read()
        frame_count += 1

    cap.release()


def process_image(image_path, output_folder="processed_images"):
    """Removes background, crops the image to 720x720, and saves it in a dedicated folder."""

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the image
    img = Image.open(image_path)

    # Remove background and convert back to a PIL Image
    img_no_bg = remove(img)
    img_no_bg = Image.fromarray(np.array(img_no_bg))  # Convert from NumPy array to PIL Image

    # Convert to RGB mode if it contains an alpha channel
    if img_no_bg.mode == "RGBA":
        img_no_bg = img_no_bg.convert("RGB")

    # Get image dimensions
    width, height = img_no_bg.size

    # Define cropping box (centered at 720x720)
    left = max(0, (width - 720) // 2)  # Ensure within bounds
    top = max(0, (height - 720) // 2)
    right = min(left + 720, width)  # Ensure within bounds
    bottom = min(top + 720, height)

    # Crop the image correctly
    cropped_img = img_no_bg.crop((left, top, right, bottom))

    # Define the path to save the processed image
    processed_image_path = os.path.join(output_folder, os.path.basename(image_path))

    # Ensure file is saved in JPEG format (replace PNG/JPEG extensions)
    processed_image_path = processed_image_path.replace(".png", ".jpg").replace(".jpeg", ".jpg")

    # Save the processed image
    cropped_img.save(processed_image_path, "JPEG")

    print(f"✅ Processed image saved at: {processed_image_path}")

    return processed_image_path


def extract_color_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Remove background correctly
    img_no_bg = remove(pil_img)
    img_no_bg = np.array(img_no_bg)

    # Ensure it has 3 channels (RGB)
    if img_no_bg.shape[-1] == 4:  # If alpha channel exists, remove it
        img_no_bg = cv2.cvtColor(img_no_bg, cv2.COLOR_RGBA2RGB)

    # Compute color features
    hist_r = cv2.calcHist([img_no_bg], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img_no_bg], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img_no_bg], [2], None, [256], [0, 256])

    hist_r = normalize(hist_r, axis=0, norm='l1').flatten()
    hist_g = normalize(hist_g, axis=0, norm='l1').flatten()
    hist_b = normalize(hist_b, axis=0, norm='l1').flatten()

    avg_r = np.mean(img_no_bg[:, :, 0])
    avg_g = np.mean(img_no_bg[:, :, 1])
    avg_b = np.mean(img_no_bg[:, :, 2])

    return {
        "Image": os.path.basename(image_path),
        "Avg Red": avg_r,
        "Avg Green": avg_g,
        "Avg Blue": avg_b,
        **{f'R Hist Bin {i}': hist_r[i] for i in range(len(hist_r))},
        **{f'G Hist Bin {i}': hist_g[i] for i in range(len(hist_g))},
        **{f'B Hist Bin {i}': hist_b[i] for i in range(len(hist_b))}
    }


# Cut Feature Extraction
def extract_geometric_features(image_path):
    features = {}
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read image {image_path}. Skipping.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
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
    symmetry_horizontal = cv2.absdiff(gray, flipped_horizontal)
    horizontal_symmetry_score = 1 - (np.mean(symmetry_horizontal) / 255)

    flipped_vertical = cv2.flip(gray, 0)
    symmetry_vertical = cv2.absdiff(gray, flipped_vertical)
    vertical_symmetry_score = 1 - (np.mean(symmetry_vertical) / 255)

    symmetry = (horizontal_symmetry_score + vertical_symmetry_score) / 2

    features['Image'] = os.path.basename(image_path)
    features['Aspect_Ratio'] = aspect_ratio
    features['Perimeter'] = perimeter
    features['Area'] = area
    features['Circularity'] = circularity
    features['Convexity'] = convexity
    features['Edge_Sharpness'] = edge_sharpness
    features['Symmetry'] = symmetry

    return features


def extract_clarity_features(image_path):
    features = {}
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features['Contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['Homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    features['Energy'] = graycoprops(glcm, 'energy')[0, 0]
    features['Correlation'] = graycoprops(glcm, 'correlation')[0, 0]

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    features['Edge_Density'] = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    features['Intensity_Variance'] = np.var(gray)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_std = np.std(hsv[:, :, 0])
    saturation_std = np.std(hsv[:, :, 1])
    features['Hue_Std'] = hue_std
    features['Saturation_Std'] = saturation_std

    features['Image'] = os.path.basename(image_path)

    return features


def process_images(input_folder, output_folder):
    extracted_features = []  # Store feature dictionaries

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)

            # Dynamically check if functions exist before calling them
            color_features = globals().get("extract_color_features", lambda x: {})(image_path) or {}
            cut_features = globals().get("extract_geometric_features", lambda x: {})(image_path) or {}
            clarity_features = globals().get("extract_clarity_features", lambda x: {})(image_path) or {}

            # Combine all extracted features into a single row
            combined_features = {**color_features, **cut_features, **clarity_features}

            # Append to the list
            extracted_features.append(combined_features)

    # Convert to DataFrame
    if extracted_features:
        df = pd.DataFrame(extracted_features)
        os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

        # Save to CSV
        output_file = os.path.join(output_folder, 'combined_features.csv')
        df.to_csv(output_file, index=False)
        print(f" Features saved to {output_file}")
    else:
        print(" No features extracted. Check image files or feature extraction functions.")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    """API endpoint to process a video, extract frames, preprocess images, and classify them."""

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded video
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    # Step 1: Extract frames and store them in the "frames" folder
    frames_folder = os.path.join(PROCESSED_FOLDER, "frames")
    extract_frames(video_path, frames_folder)

    # Step 2: Process extracted frames and save preprocessed images in "preprocessed" folder
    preprocessed_folder = os.path.join(PROCESSED_FOLDER, "preprocessed")
    os.makedirs(preprocessed_folder, exist_ok=True)

    preprocessed_paths = []
    for frame in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame)
        processed_path = process_image(frame_path, preprocessed_folder)  # Process and store preprocessed image
        preprocessed_paths.append(processed_path)

    # Step 3: Extract features from **preprocessed images** instead of frames
    feature_list = []
    for image_path in preprocessed_paths:
        # Extract features from preprocessed image
        color_features = extract_color_features(image_path) or {}
        cut_features = extract_geometric_features(image_path) or {}
        clarity_features = extract_clarity_features(image_path) or {}

        # Combine extracted features
        combined_features = {**color_features, **cut_features, **clarity_features}

        if combined_features:  # Avoid adding empty dictionaries
            feature_list.append(combined_features)

    # Ensure features were extracted
    if not feature_list:
        return jsonify({"error": "No features extracted from preprocessed images"}), 500

    # Convert to DataFrame and clean
    feature_df = pd.DataFrame(feature_list).fillna(0)

    # Save extracted features
    feature_output_path = os.path.join(PROCESSED_FOLDER, "feature_extraction", "combined_features.csv")
    os.makedirs(os.path.dirname(feature_output_path), exist_ok=True)
    feature_df.to_csv(feature_output_path, index=False)
    print(f"✅ Extracted features saved to {feature_output_path}")

    # ---------------------- STEP 1: Apply PCA Transformation ----------------------

    # Remove rows with NaN values (if any)
    data_clean = feature_df.dropna()

    # Drop 'Image' column if exists (since it's non-numeric)
    if 'Image' in data_clean.columns:
        features = data_clean.drop(columns=['Image'])
    else:
        features = data_clean.copy()

    # Normalize the features (Standardization)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA to retain 30 components
    pca = PCA(n_components=30)
    features_pca = pca.fit_transform(features_scaled)

    # Convert PCA results to DataFrame with column names (0 to 29)
    pca_df = pd.DataFrame(features_pca, columns=[str(i) for i in range(30)])

    # Save PCA results
    pca_output_path = os.path.join(PROCESSED_FOLDER, "feature_extraction", "pca_results.csv")
    pca_df.to_csv(pca_output_path, index=False)
    print(f"PCA results saved at {pca_output_path}")

    # ---------------------- STEP 2: Reformat Dataset and Place PCA Columns ----------------------

    # Reload the original dataset (without PCA transformation)
    data = feature_df.copy()

    # Drop PCA columns if they exist
    pca_columns = [str(i) for i in range(30)]
    data = data.drop(columns=pca_columns, errors='ignore')

    # Ensure the column order matches model expectations
    if "CO" in data.columns:
        co_index = data.columns.get_loc("CO")
        data = data.iloc[:, co_index:]
    else:
        data = pd.DataFrame()  # Reset to empty DataFrame if CO is missing

    # Remove all columns after DR
    if "DR" in data.columns:
        dr_index = data.columns.get_loc("DR") + 1
        data = data.iloc[:, :dr_index]

    # Ensure at least 92 columns before appending PCA data
    while len(data.columns) < 92:
        data[f'Placeholder_{len(data.columns)}'] = None

    # Append PCA columns at CO to DR
    data_final = pd.concat([data, pca_df], axis=1)

    # Save final structured dataset
    pca_updated_path = os.path.join(PROCESSED_FOLDER, "feature_extraction", "pca_updated.csv")
    data_final.to_csv(pca_updated_path, index=False)
    print(f"PCA results moved to columns CO to DR and saved at {pca_updated_path}")

    # ---------------------- STEP 3: Load and Predict with Trained Model ----------------------

    # Load the trained model
    model = joblib.load(CLASSIFICATION_MODEL_PATH)

    # Load the PCA-updated dataset for prediction
    data = pd.read_csv(pca_updated_path)

    # Select only the 30 PCA features used for training
    expected_feature_count = 30
    data = data.iloc[:, -expected_feature_count:]

    # Make predictions using the trained model
    predictions = model.predict(data)

    # Convert predictions to DataFrame with correct target column names
    if predictions.shape[1] == len(target_columns):
        predicted_df = pd.DataFrame(predictions, columns=target_columns)
    else:
        print("Warning: Number of predicted columns does not match expected column names.")
        predicted_df = pd.DataFrame(predictions[:, :len(target_columns)], columns=target_columns[:predictions.shape[1]])

    # Save predictions
    predicted_csv = os.path.join(PROCESSED_FOLDER, "feature_extraction", "predicted_targets.csv")
    predicted_df.to_csv(predicted_csv, index=False)

    print(f"Predicted target labels saved to {predicted_csv}")

    # ---------------------- STEP 4: Extract Final Top Targets ----------------------

    # Load the predicted target dataset
    data = pd.read_csv(predicted_csv)

    # Define categories and their respective prefixes
    categories = {
        "Color": "Color_",
        "Shape": "Shape_",
        "Cut": "Cut_",
        "Clarity": "Clarity_",
        "Color Intensity": "Color Intensity_"
    }

    # Find the most frequent `1` column in each category
    selected_values = {}
    for category, prefix in categories.items():
        category_columns = [col for col in data.columns if col.startswith(prefix)]
        if category_columns:
            ones_count = data[category_columns].eq(1).sum()
            if ones_count.max() > 0:
                top_column = ones_count.idxmax()
                selected_values[category] = top_column.replace(prefix, "")
            else:
                selected_values[category] = "Unknown"

    # Create a single-row DataFrame with selected attributes
    final_data = pd.DataFrame([selected_values])

    # Save the final result
    final_output_csv = os.path.join(PROCESSED_FOLDER, "feature_extraction", "final_top_targets.csv")
    final_data.to_csv(final_output_csv, index=False)

    print(f"Final top target attributes saved in {final_output_csv}")

    # Return results as JSON for frontend use
    return jsonify({
        "message": "Processing complete",
        "final_output": final_output_csv,
        "predicted_attributes": selected_values
    })


# Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        # Log received form data
        print("📌 Received form data:", data)

        # Validate required fields
        required_fields = ['carat', 'cutQuality', 'shape', 'origin', 'color',
                           'colorIntensity', 'clarity', 'cut', 'treatment', 'type']
        missing_fields = [field for field in required_fields if not data.get(field)]

        if missing_fields:
            print("🚨 Missing fields:", missing_fields)
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Load encoded features from JSON file
        if not os.path.exists(ENCODED_FEATURES_PATH):
            print("🚨 Error: Encoded features file not found.")
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
            print("📌 Encoded values:", {
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
                print("🚨 Invalid selection in dropdown values.")
                return jsonify({"error": "Invalid selection in dropdown values. Ensure all inputs are correct."}), 400

        except ValueError as e:
            print("🚨 ValueError:", e)
            return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400

        # Prepare model input for price prediction
        model_input = np.array([[carat, cut_quality_encoded, shape_encoded, origin_encoded,
                                 color_encoded, color_intensity_encoded, clarity_encoded,
                                 cut_encoded, treatment_encoded, type_encoded]])

        # Log model input
        print("📌 Model input array:", model_input)

        # Ensure price model is loaded
        if price_model is None:
            print("🚨 Error: Price prediction model not found.")
            return jsonify({"error": "Price prediction model is not loaded."}), 500

        # Make prediction
        predicted_price = price_model.predict(model_input)[0]
        predicted_price = np.expm1(predicted_price)  # Convert back from log-transformed price

        # Log the final price
        print("✅ Predicted Price:", predicted_price)

        return jsonify({"price": round(float(predicted_price), 2)})

    except Exception as e:
        print("🚨 Prediction error:", str(e))
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
