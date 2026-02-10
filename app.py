import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>❤️ Heart Disease Prediction App</h1>",
    unsafe_allow_html=True
)

st.info(
    "Models are trained offline and loaded as `.pkl` files. "
    "By default, models are evaluated on a 20% test split of the dataset. "
    "You may optionally select the saved test dataset to reproduce results."
)

# -------------------------------------------------
# Sidebar: Model selection
# -------------------------------------------------
st.sidebar.header("⚙️ Model Selection")

model_paths = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl",
}

model_name = st.sidebar.selectbox("Choose model", list(model_paths.keys()))

# -------------------------------------------------
# Load model & scaler
# -------------------------------------------------
model = joblib.load(model_paths[model_name])
scaler = joblib.load("model/scaler.pkl")

# -------------------------------------------------
# Dataset selection (ONLY test.csv here)
# -------------------------------------------------
st.subheader("📂 Evaluation Dataset")

use_saved_test = st.checkbox(
    "Use saved test dataset (data/test.csv)"
)

if use_saved_test:
    df = pd.read_csv("data/test.csv")
    st.success("📌 Evaluating on saved test dataset (data/test.csv)")
else:
    df_full = pd.read_csv("data/heart.csv")

    X = df_full.drop("target", axis=1)
    y = df_full["target"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    df = X_test.copy()
    df["target"] = y_test.values

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
st.subheader("📊 Evaluation Metrics")

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.3f}")

# -------------------------------------------------
# Confusion Matrix (compact)
# -------------------------------------------------
st.subheader("🔍 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(3.6, 3))
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
