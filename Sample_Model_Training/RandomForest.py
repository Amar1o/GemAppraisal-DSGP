import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer

# Load data
file_path = r"C:\Users\Muralish\Desktop\Sapphires_Cleaned\Blue Sapphires\A Model training\final_combined_results.csv"
data = pd.read_csv(file_path)

# Drop rows with NaN or infinite values
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

# Define scorer for hyperparameter tuning (macro-averaged F1-score)
scorer = make_scorer(f1_score, average='macro', zero_division=1)

# Define hyperparameter grid for Random Forest (updated to prevent overfitting)
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# RandomizedSearchCV for hyperparameter tuning
rf_model = RandomForestClassifier(random_state=42)
grid_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=50,
    cv=3,
    scoring=scorer,
    random_state=42,
    n_jobs=-1
)

# Fit the RandomizedSearchCV to the training data
grid_search.fit(X_train, Y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best F1-Score on Training Data:", grid_search.best_score_)

# Use the best estimator from RandomizedSearchCV
best_rf_model = grid_search.best_estimator_

# Predictions on train, validation, and test sets (tuned model)
tuned_train_preds = best_rf_model.predict(X_train)
tuned_val_preds = best_rf_model.predict(X_val)
tuned_test_preds = best_rf_model.predict(X_test)

# Evaluate the tuned model with F1-score and accuracy
tuned_train_f1 = f1_score(Y_train, tuned_train_preds, average='macro', zero_division=1)
tuned_val_f1 = f1_score(Y_val, tuned_val_preds, average='macro', zero_division=1)
tuned_test_f1 = f1_score(Y_test, tuned_test_preds, average='macro', zero_division=1)

tuned_train_accuracy = accuracy_score(Y_train, tuned_train_preds)
tuned_val_accuracy = accuracy_score(Y_val, tuned_val_preds)
tuned_test_accuracy = accuracy_score(Y_test, tuned_test_preds)

# Default Random Forest (no tuning)
default_rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
default_rf_model.fit(X_train, Y_train)

# Predictions on train, validation, and test sets (default model)
default_train_preds = default_rf_model.predict(X_train)
default_val_preds = default_rf_model.predict(X_val)
default_test_preds = default_rf_model.predict(X_test)

# Evaluate the default model with F1-score and accuracy
default_train_f1 = f1_score(Y_train, default_train_preds, average='macro', zero_division=1)
default_val_f1 = f1_score(Y_val, default_val_preds, average='macro', zero_division=1)
default_test_f1 = f1_score(Y_test, default_test_preds, average='macro', zero_division=1)

default_train_accuracy = accuracy_score(Y_train, default_train_preds)
default_val_accuracy = accuracy_score(Y_val, default_val_preds)
default_test_accuracy = accuracy_score(Y_test, default_test_preds)

# Print Results
print("\nDefault Random Forest Results")
print(f"Train Accuracy: {default_train_accuracy:.4f}")
print(f"Validation Accuracy: {default_val_accuracy:.4f}")
print(f"Test Accuracy: {default_test_accuracy:.4f}")
print(f"Train F1-Score: {default_train_f1:.4f}")
print(f"Validation F1-Score: {default_val_f1:.4f}")
print(f"Test F1-Score: {default_test_f1:.4f}")

print("\nTuned Random Forest Results")
print(f"Train Accuracy: {tuned_train_accuracy:.4f}")
print(f"Validation Accuracy: {tuned_val_accuracy:.4f}")
print(f"Test Accuracy: {tuned_test_accuracy:.4f}")
print(f"Train F1-Score: {tuned_train_f1:.4f}")
print(f"Validation F1-Score: {tuned_val_f1:.4f}")
print(f"Test F1-Score: {tuned_test_f1:.4f}")
