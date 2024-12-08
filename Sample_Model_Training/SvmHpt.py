# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, make_scorer

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

# Define scorer for hyperparameter tuning (macro-averaged F1-score)
scorer = make_scorer(f1_score, average='macro', zero_division=1)

# Define hyperparameter grid for SVM
svm_param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Kernel coefficient
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel types
    'degree': [2, 3, 4],  # Degree for 'poly' kernel
    'class_weight': [None, 'balanced']  # Balance classes automatically
}

# RandomizedSearchCV for SVM
svm_model = SVC(random_state=42, probability=True)  # Enable probability for compatibility with scoring
svm_grid_search = RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=svm_param_grid,
    n_iter=50,  # Number of parameter combinations to try
    cv=3,  # 3-fold cross-validation
    scoring=scorer,  # Macro F1-score
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Fit the RandomizedSearchCV to the training data
svm_grid_search.fit(X_train, Y_train.values.argmax(axis=1))  # SVM expects a single label, use argmax for multi-label

# Best parameters and score for SVM
print("SVM - Best Parameters:", svm_grid_search.best_params_)
print("SVM - Best F1-Score on Training Data:", svm_grid_search.best_score_)

# Use the best estimator
best_svm_model = svm_grid_search.best_estimator_

# Predictions
svm_train_preds = best_svm_model.predict(X_train)
svm_val_preds = best_svm_model.predict(X_val)
svm_test_preds = best_svm_model.predict(X_test)

# Evaluate SVM
svm_train_f1 = f1_score(Y_train.values.argmax(axis=1), svm_train_preds, average='macro', zero_division=1)
svm_val_f1 = f1_score(Y_val.values.argmax(axis=1), svm_val_preds, average='macro', zero_division=1)
svm_test_f1 = f1_score(Y_test.values.argmax(axis=1), svm_test_preds, average='macro', zero_division=1)

print("\nSVM Results")
print("Train F1-Score:", svm_train_f1)
print("Validation F1-Score:", svm_val_f1)
print("Test F1-Score:", svm_test_f1)
