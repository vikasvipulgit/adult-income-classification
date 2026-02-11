import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

from preprocessing import load_data, preprocess_data, split_data
from model.models import get_models, train_models, evaluate_models

st.set_page_config(page_title="Adult Income Classification", layout="wide")
st.title("Adult Income Classification Models")

# ----------------------------
# Load & Preprocess
# ----------------------------
df = load_data("data/adult.csv")

st.subheader("Dataset Shape")
st.write(df.shape)

X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# ----------------------------
# Train Button
# ----------------------------
if st.button("Train Models"):

    models = get_models()
    trained_models = train_models(models, X_train, y_train)

    st.session_state["trained_models"] = trained_models
    st.success("Models trained successfully!")

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict on Test Data"):

    if "trained_models" not in st.session_state:
        st.warning("Please train the models first.")
    else:
        trained_models = st.session_state["trained_models"]

        results_df = evaluate_models(trained_models, X_test, y_test)

        st.subheader("Evaluation Metrics")
        st.dataframe(results_df)

        # ROC Curves
        st.subheader("ROC Curve Comparison")
        fig1, ax1 = plt.subplots()

        for name, model in trained_models.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax1.plot(fpr, tpr, label=name)

        ax1.plot([0, 1], [0, 1], linestyle="--")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend()
        st.pyplot(fig1)

        # Accuracy Bar Chart
        st.subheader("Accuracy Comparison")
        fig2, ax2 = plt.subplots()
        results_df["Accuracy"].plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Accuracy")
        st.pyplot(fig2)

        # Confusion Matrix for Best Model
        best_model_name = results_df["Accuracy"].idxmax()
        best_model = trained_models[best_model_name]
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
