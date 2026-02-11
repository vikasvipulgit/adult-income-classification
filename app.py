import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

from preprocessing import load_data, preprocess_data, split_data
from model.models import get_models, train_models, evaluate_models

st.set_page_config(page_title="Adult Income Classification", layout="wide")

st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    color: black;
}

/* Section headers */
h1, h2, h3 {
    color: #ffffff;
}

/* Buttons */
div.stButton > button {
    background-color: #ff4b2b;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #ff416c;
    color: white;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


st.title("Adult Income Classification Models")

# ----------------------------
# Load & Preprocess
# ----------------------------
df = load_data("data/adult.csv")

X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# ----------------------------
# Model Selection Dropdown
# ----------------------------
all_models = get_models()
model_names = list(all_models.keys())
model_names.append("All Models")

st.markdown("""
<h3 style='font-weight:700; font-size:22px; margin-bottom:5px;'>
🚀 Select Model to Train
</h3>
""", unsafe_allow_html=True)

selected_model = st.selectbox(
    "",
    model_names
)


# ----------------------------
# Train & Evaluate
# ----------------------------
if st.button("Train and Evaluate"):

    if selected_model == "All Models":
        models_to_train = all_models
    else:
        models_to_train = {selected_model: all_models[selected_model]}

    trained_models = train_models(models_to_train, X_train, y_train)

    results_df = evaluate_models(trained_models, X_test, y_test)

    st.markdown("## 📊 Evaluation Metrics")
    styled_df = results_df.style\
        .background_gradient(cmap="viridis")\
        .format("{:.4f}")

    styled_df = results_df.style\
        .highlight_max(color="lightgreen")\
        .format("{:.4f}")


    st.dataframe(styled_df)


    # ----------------------------
    # ROC Curve
    # ----------------------------
    st.subheader("ROC Curve")

    fig1, ax1 = plt.subplots(figsize=(6, 5))

    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, label=name)

    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()

    st.pyplot(fig1)

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    st.subheader("Confusion Matrix")

    for name, model in trained_models.items():

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        st.markdown(f"### {name}")

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im = ax2.imshow(cm, cmap="Blues")

        plt.colorbar(im)

        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(["<=50K", ">50K"])
        ax2.set_yticklabels(["<=50K", ">50K"])

        threshold = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > threshold else "black"
                ax2.text(j, i, cm[i, j],
                         ha="center", va="center",
                         color=color)

        st.pyplot(fig2)
