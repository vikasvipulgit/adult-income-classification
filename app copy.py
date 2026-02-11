import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

st.set_page_config(page_title="Adult Income Classification", layout="wide")

st.title("Adult Income Classification App")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("adult.csv")
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

df = load_data()

st.subheader("Dataset Overview")
st.write(df.head())
st.write("Shape:", df.shape)

# ----------------------------
# Preprocessing
# ----------------------------
@st.cache_data
def preprocess_data(df):
    df = df.copy()

    # Encode target
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

    X = df.drop("income", axis=1)
    y = df["income"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns

X, y, feature_names = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# Train Models
# ----------------------------
if st.button("Train All Models"):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }

    results_df = pd.DataFrame(results).T

    st.subheader("Model Evaluation Results")
    st.dataframe(results_df)

    # Best model
    best_model = results_df["Accuracy"].idxmax()
    st.success(f"Best Model Based on Accuracy: {best_model}")

    # ROC Curve Plot
    st.subheader("ROC Curves")

    fig, ax = plt.subplots()

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=name)

    ax.plot([0,1], [0,1], linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    st.pyplot(fig)

# ----------------------------
# Prediction Section
# ----------------------------
st.subheader("Make a Prediction")

if st.checkbox("Show Prediction Form"):

    input_data = {}

    for feature in feature_names:
        input_data[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict Income"):

        input_df = pd.DataFrame([input_data])
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("Prediction: Income >50K")
        else:
            st.error("Prediction: Income <=50K")
"""Simple entrypoint for the Adult Income Classification project.

Run with: python app.py
"""
from models import load_model, predict
from utils import load_data, preprocess


def main():
    print("Loading model...")
    model = load_model()

    print("Loading sample data from data/ ...")
    df = load_data("data/sample.csv")
    if df is None:
        print("No sample data found in data/sample.csv. Exiting.")
        return

    X = preprocess(df)
    preds = predict(model, X)
    print("Predictions:\n", preds[:5])


if __name__ == "__main__":
    main()
