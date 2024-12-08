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
    'max_depth': [5, 10, 15, 20],  # Limit depth to avoid overfitting
    'min_samples_split': [5, 10, 15],  # Increased min_samples_split to avoid overfitting
    'min_samples_leaf': [1, 2, 4],  # Limit leaf nodes
    'max_features': ['sqrt', 'log2'],  # Limiting features considered for each split
    'bootstrap': [True, False]  # Use bootstrap samples to help with overfitting
}

# RandomizedSearchCV for hyperparameter tuning
rf_model = RandomForestClassifier(random_state=42)
grid_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter combinations to try
    cv=3,  # 3-fold cross-validation
    scoring=scorer,  # Macro F1-score
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Fit the RandomizedSearchCV to the training data
grid_search.fit(X_train, Y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best F1-Score on Training Data:", grid_search.best_score_)

# Use the best estimator from RandomizedSearchCV
best_rf_model = grid_search.best_estimator_

# Predictions on train, validation, and test sets
train_preds = best_rf_model.predict(X_train)
val_preds = best_rf_model.predict(X_val)
test_preds = best_rf_model.predict(X_test)

# Handle invalid values (NaN, inf) in predictions and targets
train_preds = np.nan_to_num(train_preds, nan=0)  # Replace NaN with 0
val_preds = np.nan_to_num(val_preds, nan=0)
test_preds = np.nan_to_num(test_preds, nan=0)

# Ensure targets are valid too
Y_train = np.nan_to_num(Y_train, nan=0)
Y_val = np.nan_to_num(Y_val, nan=0)
Y_test = np.nan_to_num(Y_test, nan=0)

# Evaluate the tuned model with F1-score and zero_division handling
train_f1 = f1_score(Y_train, train_preds, average='macro', zero_division=1)
val_f1 = f1_score(Y_val, val_preds, average='macro', zero_division=1)
test_f1 = f1_score(Y_test, test_preds, average='macro', zero_division=1)

print("\nTuned Random Forest Results")
print("Train F1-Score:", train_f1)
print("Validation F1-Score:", val_f1)
print("Test F1-Score:", test_f1)

# Evaluate with accuracy metrics
train_accuracy = accuracy_score(Y_train, train_preds)
val_accuracy = accuracy_score(Y_val, val_preds)
test_accuracy = accuracy_score(Y_test, test_preds)

print("\nTuned Random Forest Accuracy Results")
print("Train Accuracy:", train_accuracy)
print("Validation Accuracy:", val_accuracy)
print("Test Accuracy:", test_accuracy)

# Optional: Compare to default Random Forest without tuning
default_rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
default_rf_model.fit(X_train, Y_train)

# Default model evaluation
default_train_preds = default_rf_model.predict(X_train)
default_val_preds = default_rf_model.predict(X_val)
default_test_preds = default_rf_model.predict(X_test)

default_train_accuracy = accuracy_score(Y_train, default_train_preds)
default_val_accuracy = accuracy_score(Y_val, default_val_preds)
default_test_accuracy = accuracy_score(Y_test, default_test_preds)

print("\nDefault Random Forest Accuracy Results")
print("Train Accuracy:", default_train_accuracy)
print("Validation Accuracy:", default_val_accuracy)
print("Test Accuracy:", default_test_accuracy)
