import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Title & Description
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">❤️ Heart Disease Prediction App</h1>
    <p style="text-align:center; font-size:16px;">
    Interactive ML application using pre-trained models for heart disease prediction.
    </p>
    """,
    unsafe_allow_html=True
)

st.info(
    "Models are trained offline using **train_models.py** and loaded as `.pkl` files. "
    "By default, evaluation results are shown immediately. You may optionally switch "
    "to the official test dataset from the repository to reproduce the same results."
)

# -------------------------------------------------
# Sidebar: Model selection
# -------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")

models = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl",
}

model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))

# -------------------------------------------------
# Load scaler and model
# -------------------------------------------------
scaler = joblib.load("model/scaler.pkl")
model = joblib.load(models[model_name])

# -------------------------------------------------
# Dataset selection (DEFAULT FIRST)
# -------------------------------------------------
st.subheader("📂 Evaluation Dataset")

dataset_choice = st.selectbox(
    "Select dataset for evaluation:",
    ["Default dataset", "Official test dataset (data/test.csv)"]
)

if dataset_choice == "Official test dataset (data/test.csv)":
    if not os.path.exists("data/test.csv"):
        st.error("data/test.csv not found in the repository.")
        st.stop()

    df = pd.read_csv("data/test.csv")
    st.success("📌 Evaluating on official test dataset (data/test.csv)")
else:
    df = pd.read_csv("data/heart.csv")
    st.info("📌 Evaluating on default dataset")

X_test = df.drop("target", axis=1)
y_test = df["target"]

# -------------------------------------------------
# Feature scaling (only where required)
# -------------------------------------------------
if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
    X_eval = scaler.transform(X_test)
else:
    X_eval = X_test

# -------------------------------------------------
# Prediction
# -------------------------------------------------
y_pred = model.predict(X_eval)
y_prob = model.predict_proba(X_eval)[:, 1]

# -------------------------------------------------
# Metrics
# -------------------------------------------------
st.subheader("📊 Model Evaluation Metrics")

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

cols = st.columns(6)
for col, (metric, value) in zip(cols, metrics.items()):
    col.metric(metric, f"{value:.3f}")

# -------------------------------------------------
# Confusion Matrix (compact)
# -------------------------------------------------
st.subheader("🔍 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(3.8, 3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    annot_kws={"size": 11},
    linewidths=0.5,
    ax=ax
)

ax.set_xlabel("Predicted", fontsize=10)
ax.set_ylabel("Actual", fontsize=10)
ax.set_title("Confusion Matrix", fontsize=11)

st.pyplot(fig, use_container_width=False)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:13px;">
    Academic ML Project • Streamlit Community Cloud Deployment
    </p>
    """,
    unsafe_allow_html=True
)
