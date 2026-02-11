# Heart Disease Prediction – Machine Learning Assignment

---

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict the presence of heart disease based on clinical features. The models are evaluated using standard classification metrics and compared systematically.

---

## 2. Dataset Description

The Heart Disease dataset contains clinical attributes such as:

- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Cholesterol levels  
- Fasting blood sugar  
- ECG results  
- Maximum heart rate  
- Exercise-induced angina  
- ST depression  
- Slope  
- Number of vessels  
- Thalassemia  
- Binary target variable  

### Data Splitting Strategy:

- 80% Training set  
- 20% Test set (held-out for evaluation)  
- Stratified split with `random_state=42` for reproducibility  

---

## 3. Models Used

- Logistic Regression  
- Decision Tree  
- k-Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

---

## 4. Model Comparison Table (Evaluation Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.810 | 0.930 | 0.762 | 0.914 | 0.831 | 0.631 |
| Decision Tree | 0.985 | 0.986 | 1.000 | 0.971 | 0.986 | 0.971 |
| kNN | 0.863 | 0.963 | 0.874 | 0.857 | 0.865 | 0.727 |
| Naive Bayes | 0.829 | 0.904 | 0.807 | 0.876 | 0.840 | 0.660 |
| Random Forest (Ensemble) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost (Ensemble) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

---

## 5. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Logistic Regression achieved good baseline performance with high recall (0.914), indicating strong sensitivity in detecting positive cases. However, lower precision suggests some false positives. |
| Decision Tree | Decision Tree achieved very high performance across all metrics (Accuracy ≈ 0.985). The near-perfect scores indicate strong fitting to the dataset. |
| k-Nearest Neighbors (kNN) | kNN showed balanced performance with strong AUC (0.963), demonstrating good class separability after feature scaling. |
| Naive Bayes | Naive Bayes performed moderately well. While recall is strong, the independence assumption may limit overall performance compared to more flexible models. |
| Random Forest (Ensemble) | Random Forest achieved perfect scores across all metrics. While highly impressive, such perfect results may indicate potential overfitting on this dataset. |
| XGBoost (Ensemble) | XGBoost also achieved perfect evaluation metrics. Gradient boosting effectively captured complex relationships, though perfect performance should be interpreted cautiously. |

---

## 6. Conclusion

The ensemble models (Random Forest and XGBoost) achieved the highest performance, with perfect evaluation scores on the test split. Decision Tree also performed exceptionally well. Logistic Regression and Naive Bayes provided strong baseline comparisons, while kNN demonstrated competitive performance. Overall, ensemble methods provided the best generalization capability for this dataset.
