import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
    confusion_matrix
)

st.set_page_config(page_title="Adult Income Classification", layout="wide")
st.title("Adult Income Classification Models")

# ----------------------------
# Load Dataset (Local File)
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("adult.csv")
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

df = load_data()

st.subheader("Dataset Information")
st.write("Shape:", df.shape)
st.write(df.head())

# ----------------------------
# Preprocessing
# ----------------------------
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

X = df.drop("income", axis=1)
y = df["income"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# Models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

trained_models = {}
results = {}

# ----------------------------
# Train Models Button
# ----------------------------
if st.button("Train Models"):
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    st.success("All models trained successfully!")

# ----------------------------
# Predict on Test Data Button
# ----------------------------
if st.button("Predict on Test Data"):

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

    st.subheader("Evaluation Metrics")
    st.dataframe(results_df)

    # ----------------------------
    # ROC Curves
    # ----------------------------
    st.subheader("ROC Curve Comparison")
    fig1, ax1 = plt.subplots()

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, label=name)

    ax1.plot([0,1], [0,1], linestyle='--')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()

    st.pyplot(fig1)

    # ----------------------------
    # Accuracy Bar Plot
    # ----------------------------
    st.subheader("Accuracy Comparison")
    fig2, ax2 = plt.subplots()
    results_df["Accuracy"].plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Accuracy")
    st.pyplot(fig2)

    # ----------------------------
    # Confusion Matrix (Best Model by Accuracy)
    # ----------------------------
    best_model_name = results_df["Accuracy"].idxmax()
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_best)

    st.subheader(f"Confusion Matrix ({best_model_name})")
    fig3, ax3 = plt.subplots()
    ax3.imshow(cm)

    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig3)
