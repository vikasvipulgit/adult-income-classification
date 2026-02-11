# Adult Income Classification using Machine Learning

## 📌 Project Overview

This project implements multiple supervised machine learning classification models on the Adult Income Dataset.  
The goal is to predict whether an individual's annual income exceeds $50K based on demographic and employment-related features.

The project includes:
- Implementation of 6 classification models
- Performance comparison using multiple evaluation metrics
- Interactive Streamlit web application
- Deployment on Streamlit Community Cloud

---

## 📊 Dataset Description

Dataset: Adult Income Dataset (UCI Machine Learning Repository)

- Instances: 48,000+
- Features: 14 input features
- Target Variable: Income (<=50K or >50K)

The dataset contains both numerical and categorical features such as:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Capital Gain
- Hours per Week
- Native Country

Target:
- 0 → Income <=50K
- 1 → Income >50K

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

1. Missing values handled by removing rows with missing entries
2. Target variable encoded into binary format
3. One-hot encoding applied to categorical features
4. Feature scaling performed using StandardScaler
5. Dataset split into:
   - 80% Training Data
   - 20% Testing Data
6. Stratified sampling used to preserve class distribution

---

## 🤖 Machine Learning Models Implemented

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

All models were trained on the same dataset and evaluated using identical metrics.

---

## 📈 Evaluation Metrics

Each model was evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Additionally, the following visualizations were generated:

- ROC Curve Comparison
- Accuracy Comparison Bar Chart
- Confusion Matrix (Best Performing Model)

---

## 🖥️ Streamlit Web Application

The project includes an interactive Streamlit application that allows:

- Viewing dataset overview
- Training all models
- Predicting on test data
- Comparing evaluation metrics
- Visualizing ROC curves
- Viewing confusion matrix

---

## 🚀 Deployment

The application is deployed using Streamlit Community Cloud.

🔗 Live Application Link:
(Insert your Streamlit deployment link here)

---

## 📂 Project Structure

adult-income-classification/
│
├── app.py
├── adult.csv
├── requirements.txt
└── README.md


---

## 📦 Requirements

The following Python libraries are required:

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost

Install dependencies using:

pip install -r requirements.txt


---

## ▶️ Running the Application Locally

To run the app locally:

streamlit run app.py


---

## 🏆 Results Summary

## Model Performance Observations

## Model Evaluation Results

| ML Model Name        | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|----------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression  | 0.8549   | 0.9031  | 0.7405    | 0.6116  | 0.6699   | 0.5824  |
| Decision Tree        | 0.8128   | 0.7472  | 0.6093    | 0.6205  | 0.6148   | 0.4913  |
| KNN                  | 0.8213   | 0.8349  | 0.6477    | 0.5651  | 0.6035   | 0.4908  |
| Naive Bayes          | 0.4543   | 0.6827  | 0.3014    | 0.9611  | 0.4589   | 0.2579  |
| Random Forest        | 0.8524   | 0.9005  | 0.7277    | 0.6186  | 0.6687   | 0.5778  |
| XGBoost              | 0.8683   | 0.9216  | 0.7706    | 0.6448  | 0.7021   | 0.6224  |



| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieved strong performance with **85.49% accuracy** and **0.903 AUC**, indicating good class separability. Precision (0.74) and recall (0.61) are reasonably balanced, resulting in an F1-score of 0.67 and MCC of 0.58. It performs well as a robust baseline linear model. |
| Decision Tree | Achieved **81.28% accuracy** and lower AUC (0.747). Although recall (0.62) is comparable to Logistic Regression, lower precision (0.61) and MCC (0.49) suggest weaker generalization and possible overfitting compared to ensemble methods. |
| KNN | Produced **82.13% accuracy** and AUC of 0.835. Performance is moderate with slightly lower recall (0.56). MCC (0.49) indicates reasonable predictive ability but inferior to Logistic Regression and ensemble models. |
| Naive Bayes | Showed poor overall performance with **45.43% accuracy** and very low MCC (0.26). Although recall is extremely high (0.96), very low precision (0.30) indicates bias toward predicting the positive class, reducing reliability. |
| Random Forest (Ensemble) | Achieved strong results with **85.24% accuracy** and **0.900 AUC**. Balanced precision (0.73) and recall (0.62) produce solid F1-score (0.67) and MCC (0.58). Performs more robustly than single Decision Tree due to ensemble learning. |
| XGBoost (Ensemble) | Delivered the best performance with highest **86.83% accuracy**, highest **0.922 AUC**, highest F1-score (0.70), and highest MCC (0.62). Demonstrates superior generalization and best overall predictive capability on this dataset. |

---

## 📚 Conclusion

This project demonstrates:
- Implementation of multiple classification algorithms
- Proper preprocessing of structured data
- Model evaluation and comparison
- Deployment of a machine learning web application

The comparison highlights the effectiveness of ensemble methods for structured tabular datasets.

---