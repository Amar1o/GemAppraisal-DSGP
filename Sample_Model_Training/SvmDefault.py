# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# Load data
file_path = r"C:\Users\Muralish\Desktop\Sapphires_Cleaned\Blue Sapphires\A Model training\final_combined_results.csv"
data = pd.read_csv(file_path)

# Data cleaning: drop rows with NaN or infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
data.dropna(inplace=True)  # Drop rows containing NaN values

# Separate features (A to T) and targets (U to BK)
X = data.iloc[:, :20]  # Features (numerical and PCA data)
Y = data.iloc[:, 20:]  # Targets (multi-label classification)

# Train, validation, and test split
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y.values.argmax(axis=1)
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp.values.argmax(axis=1)
)

# Train default SVM model
default_svm_model = SVC(random_state=42, probability=True)  # Default parameters
default_svm_model.fit(X_train, Y_train.values.argmax(axis=1))

# Predictions for default SVM
default_train_preds = default_svm_model.predict(X_train)
default_val_preds = default_svm_model.predict(X_val)
default_test_preds = default_svm_model.predict(X_test)

# Evaluate default SVM
default_train_f1 = f1_score(Y_train.values.argmax(axis=1), default_train_preds, average='macro', zero_division=1)
default_val_f1 = f1_score(Y_val.values.argmax(axis=1), default_val_preds, average='macro', zero_division=1)
default_test_f1 = f1_score(Y_test.values.argmax(axis=1), default_test_preds, average='macro', zero_division=1)

# Display results
print("\nDefault SVM Results")
print("Train F1-Score:", default_train_f1)
print("Validation F1-Score:", default_val_f1)
print("Test F1-Score:", default_test_f1)
