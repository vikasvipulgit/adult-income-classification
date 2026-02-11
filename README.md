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

Among the implemented models, ensemble models such as Random Forest and XGBoost generally achieved higher performance compared to individual classifiers.

Model comparison was done based on Accuracy, AUC, and MCC scores.

---

## 📚 Conclusion

This project demonstrates:
- Implementation of multiple classification algorithms
- Proper preprocessing of structured data
- Model evaluation and comparison
- Deployment of a machine learning web application

The comparison highlights the effectiveness of ensemble methods for structured tabular datasets.

---