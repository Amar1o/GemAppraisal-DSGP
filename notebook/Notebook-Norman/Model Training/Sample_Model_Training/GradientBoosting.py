import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, make_scorer

# Load data
file_path = r"C:\Users\Muralish\Desktop\Sapphires_Cleaned\Blue Sapphires\A Model training\final_combined_results.csv"
data = pd.read_csv(file_path)

# Drop rows with NaN or infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
data.dropna(inplace=True)  # Drop rows containing NaN values

# Separate features (A to T) and targets (U to BK)
X = data.iloc[:, :20]  # Features (numerical and PCA data)
Y = data.iloc[:, 20:]  # Targets (multi-label classification)

# Convert multi-label targets to single-label by taking the index of the maximum value (argmax)
Y = Y.values.argmax(axis=1)

# Train, validation, and test split
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp
)

# Define scorer for hyperparameter tuning (macro-averaged F1-score)
scorer = make_scorer(f1_score, average='macro', zero_division=1)

# Default Gradient Boosting Classifier
default_gb_model = GradientBoostingClassifier(random_state=42)
default_gb_model.fit(X_train, Y_train)

default_train_preds = default_gb_model.predict(X_train)
default_val_preds = default_gb_model.predict(X_val)
default_test_preds = default_gb_model.predict(X_test)

# Default model evaluation
default_train_f1 = f1_score(Y_train, default_train_preds, average='macro', zero_division=1)
default_val_f1 = f1_score(Y_val, default_val_preds, average='macro', zero_division=1)
default_test_f1 = f1_score(Y_test, default_test_preds, average='macro', zero_division=1)

default_train_accuracy = accuracy_score(Y_train, default_train_preds)
default_val_accuracy = accuracy_score(Y_val, default_val_preds)
default_test_accuracy = accuracy_score(Y_test, default_test_preds)

print("\nDefault Gradient Boosting Results")
print("Train Accuracy:", default_train_accuracy)
print("Validation Accuracy:", default_val_accuracy)
print("Test Accuracy:", default_test_accuracy)
print("Train F1-Score:", default_train_f1)
print("Validation F1-Score:", default_val_f1)
print("Test F1-Score:", default_test_f1)

# Define hyperparameter grid for Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 1.0]
}

# RandomizedSearchCV for Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_grid_search = RandomizedSearchCV(
    estimator=gb_model,
    param_distributions=gb_param_grid,
    n_iter=50,
    cv=3,
    scoring=scorer,
    random_state=42,
    n_jobs=-1
)

# Fit Gradient Boosting RandomizedSearchCV
gb_grid_search.fit(X_train, Y_train)

# Best parameters and score for Gradient Boosting
print("\nGradient Boosting - Best Parameters:", gb_grid_search.best_params_)
print("Gradient Boosting - Best F1-Score on Training Data:", gb_grid_search.best_score_)

# Use the best estimator
best_gb_model = gb_grid_search.best_estimator_

# Predictions
gb_train_preds = best_gb_model.predict(X_train)
gb_val_preds = best_gb_model.predict(X_val)
gb_test_preds = best_gb_model.predict(X_test)

# Tuned model evaluation
gb_train_f1 = f1_score(Y_train, gb_train_preds, average='macro', zero_division=1)
gb_val_f1 = f1_score(Y_val, gb_val_preds, average='macro', zero_division=1)
gb_test_f1 = f1_score(Y_test, gb_test_preds, average='macro', zero_division=1)

gb_train_accuracy = accuracy_score(Y_train, gb_train_preds)
gb_val_accuracy = accuracy_score(Y_val, gb_val_preds)
gb_test_accuracy = accuracy_score(Y_test, gb_test_preds)

print("\nTuned Gradient Boosting Results")
print("Train Accuracy:", gb_train_accuracy)
print("Validation Accuracy:", gb_val_accuracy)
print("Test Accuracy:", gb_test_accuracy)
print("Train F1-Score:", gb_train_f1)
print("Validation F1-Score:", gb_val_f1)
print("Test F1-Score:", gb_test_f1)
