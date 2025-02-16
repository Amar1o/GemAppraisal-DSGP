import pandas as pd
import numpy as np
import pickle  # For saving and loading the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2

# # Function to train the model on multiple classes (Green & White Sapphire)
def train_model(csv_paths, class_names, save_path):
    """
    Train a Random Forest model for multiple classes.
    :param csv_paths: Dictionary containing class names as keys and their CSV paths as values.
    :param class_names: List of class names.
    :param save_path: Path to save the trained model.
    """
    all_data = []
    all_labels = []

    # Load datasets for each class and merge
    for class_name, csv_path in csv_paths.items():
        data = pd.read_csv(csv_path)
        X = data.iloc[:, 1:].values  # Extract features (ignore the first column - image_path)
        y = np.array([class_name] * len(X))  # Assign the class label
        all_data.append(X)
        all_labels.append(y)

    # Combine all data
    X = np.vstack(all_data)
    y = np.concatenate(all_labels)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the trained model
    with open(save_path, 'wb') as model_file:
        pickle.dump(rf_classifier, model_file)
    print(f"Model trained and saved at: {save_path}")

# # Function to predict using a test image
def predict_class(image_path, model_path):
    """
    Predict the class of a given image using a trained model.
    :param image_path: Path to the test image.
    :param model_path: Path to the trained model.
    """
    # Load the trained model
    with open(model_path, 'rb') as model_file:
        rf_classifier = pickle.load(model_file)

    # Extract color histogram features from the test image
    def extract_color_histogram(image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))  # Resize to match the training size
        hist_features = []

        # Extract histograms for R, G, and B channels
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten
            hist_features.extend(hist)

        return np.array(hist_features)

    test_image_features = extract_color_histogram(image_path)
    test_image_features = test_image_features.reshape(1, -1)  # Reshape for classifier

    # Predict the class
    predicted_class = rf_classifier.predict(test_image_features)
    predicted_probabilities = rf_classifier.predict_proba(test_image_features)

    # Output the prediction
    print(f"Predicted Class: {predicted_class[0]}")
    print(f"Class Probabilities: {predicted_probabilities}")

# Paths to CSV files
csv_paths = {
    "green_sapphire": r"E:\DSGP\Feature\image_color_histogramsgreen.csv",  # Replace with actual CSV path
    "white_sapphire": r"E:\DSGP\Feature\image_color_histogramswhite.csv"   # Replace with actual CSV path
}

# List of class names
class_names = ["green_sapphire", "white_sapphire"]

# Path to save the trained model
model_save_path = "sapphire_model.pkl"

# Path to test image
image_path = r"E:\DSGP\Feature\52_image2.png"  # Replace with actual image path

# Train the model on both classes
train_model(csv_paths, class_names, model_save_path)

# Predict with a test image
predict_class(image_path, model_save_path)


# import pandas as pd
# import numpy as np
# import pickle  # For saving and loading the model
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import cv2

# # Function to train the model on multiple classes (Green & White Sapphire) with one-hot encoding
# def train_model(csv_paths, save_path):
#     """
#     Train a Random Forest model for multiple classes using one-hot encoding.
#     :param csv_paths: Dictionary containing class names as keys and their CSV paths as values.
#     :param save_path: Path to save the trained model.
#     """
#     all_data = []
#     all_labels = []

#     # Load datasets for each class and merge
#     for class_name, csv_path in csv_paths.items():
#         data = pd.read_csv(csv_path)
#         X = data.iloc[:, 1:].values  # Extract features (ignore first column - image_path)
#         y = [class_name] * len(X)  # Store class labels as a list
#         all_data.append(X)
#         all_labels.extend(y)  # Append labels to list

#     # Combine all data
#     X = np.vstack(all_data)
#     y = pd.DataFrame(all_labels, columns=['class'])  # Convert labels to DataFrame

#     # One-hot encode class labels using pd.get_dummies()
#     y_encoded = pd.get_dummies(y['class']).values  # Convert categorical labels to one-hot encoding

#     # Split dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#     # Initialize and train the Random Forest model
#     rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf_classifier.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = rf_classifier.predict(X_test)
#     y_test_labels = np.argmax(y_test, axis=1)
#     y_pred_labels = np.argmax(y_pred, axis=1)

#     print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))
#     print("\nClassification Report:\n", classification_report(y_test_labels, y_pred_labels))
#     print("\nConfusion Matrix:\n", confusion_matrix(y_test_labels, y_pred_labels))

#     # Save the trained model
#     with open(save_path, 'wb') as model_file:
#         pickle.dump(rf_classifier, model_file)  # Save model
#     print(f"Model trained and saved at: {save_path}")

# # Function to predict using a test image
# def predict_class(image_path, model_path):
#     """
#     Predict the class of a given image using a trained model.
#     :param image_path: Path to the test image.
#     :param model_path: Path to the trained model.
#     """
#     # Load the trained model
#     with open(model_path, 'rb') as model_file:
#         rf_classifier = pickle.load(model_file)

#     # Extract color histogram features from the test image
#     def extract_color_histogram(image_path):
#         image = cv2.imread(image_path)
#         image = cv2.resize(image, (128, 128))  # Resize to match the training size
#         hist_features = []

#         # Extract histograms for R, G, and B channels
#         for i in range(3):  # BGR channels
#             hist = cv2.calcHist([image], [i], None, [256], [0, 256])
#             hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten
#             hist_features.extend(hist)

#         return np.array(hist_features)

#     test_image_features = extract_color_histogram(image_path)
#     test_image_features = test_image_features.reshape(1, -1)  # Reshape for classifier

#     # Predict the class (getting probabilities and selecting highest probability class)
#     predicted_probabilities = rf_classifier.predict_proba(test_image_features)
#     predicted_class_index = np.argmax(predicted_probabilities)
    
#     # Since we used pd.get_dummies() for encoding, we need to extract class names
#     class_labels = list(pd.get_dummies(["green_sapphire", "white_sapphire"]).columns)
#     predicted_class = class_labels[predicted_class_index]

#     # Output the prediction
#     print(f"Predicted Class: {predicted_class}")
#     print(f"Class Probabilities: {predicted_probabilities}")

# # Paths to CSV files
# csv_paths = {
#     "green_sapphire": r"E:\DSGP\Feature\image_color_histogramsgreen.csv",  # Replace with actual CSV path
#     "white_sapphire": r"E:\DSGP\Feature\image_color_histogramswhite.csv"   # Replace with actual CSV path
# }

# # Path to save the trained model
# model_save_path = "sapphire_model_onehotencoded.pkl"

# # Path to test image
# image_path = r"E:\DSGP\Feature\52_image2.png"  # Replace with actual image path

# # Train the model on both classes
# train_model(csv_paths, model_save_path)

# # Predict with a test image
# predict_class(image_path, model_save_path)
