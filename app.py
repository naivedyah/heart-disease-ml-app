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
    "Models are trained offline using **train_models.py** and loaded here as `.pkl` files. "
    "You can evaluate using the official test dataset from the repository or upload your own labeled test CSV."
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
# Load scaler and model (NO TRAINING)
# -------------------------------------------------
scaler = joblib.load("model/scaler.pkl")
model = joblib.load(models[model_name])

# -------------------------------------------------
# Test dataset selection
# -------------------------------------------------
st.subheader("📤 Test Dataset Selection")

test_mode = st.radio(
    "Choose test data source:",
    ["Use official test dataset (from repo)", "Upload custom test dataset"]
)

if test_mode == "Use official test dataset (from repo)":
    if not os.path.exists("data/test.csv"):
        st.error("Official test.csv not found in data/ directory.")
        st.stop()

    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]
    st.success("📌 Evaluating on official test dataset (data/test.csv)")

else:
    uploaded_file = st.file_uploader(
        "Upload labeled test CSV (same features + target)",
        type=["csv"]
    )

    if uploaded_file is None:
        st.warning("Please upload a labeled test CSV to proceed.")
        st.stop()

    test_df = pd.read_csv(uploaded_file)

    if "target" not in test_df.columns:
        st.error("Uploaded CSV must contain a `target` column.")
        st.stop()

    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]
    st.success("📌 Evaluating on uploaded test dataset")

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
# Confusion Matrix (compact & clean)
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
