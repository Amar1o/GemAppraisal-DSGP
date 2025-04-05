## Model Performance Evaluation

| Model             | Description                         | Train F1-Score (micro) | Train Hamming Loss | Validate F1-Score (micro) | Validate Hamming Loss | Test F1-Score (micro) | Test Hamming Loss |
|------------------|---------------------------------|------------------|---------------|------------------|---------------|------------------|---------------|
| Logistic Regression | Used OneVsRestClassifier      | 38.4%            | 4.7%          | 38.4%            | 4.7%          | 38.3%            | 4.7%          |
| XGBoost           | Used MultiOutputClassifier     | 92.9%            | 0.8%          | 86.3%            | 1.4%          | 86.3%            | 1.4%          |
| Random Forest     | Used MultiOutputClassifier     | 99.8%            | 0.01%         | 89.4%            | 1.1%          | 89.4%            | 1.1%          |




![hammingLoss](https://github.com/user-attachments/assets/54c11604-f673-4c46-905c-f68e097cb6d5)
