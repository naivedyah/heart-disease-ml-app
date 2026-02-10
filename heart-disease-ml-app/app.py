import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")

st.info(
    "Models are trained offline using train_models.py and loaded here as .pkl files. "
    "Default evaluation uses the internal dataset. Uploading the same labeled test.csv "
    "will reproduce identical results."
)

# -------------------------------------------------
# Load scaler and models (NO TRAINING HERE)
# -------------------------------------------------
scaler = joblib.load("model/scaler.pkl")

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl"),
}

# -------------------------------------------------
# Model selection
# -------------------------------------------------
model_name = st.selectbox("Select ML Model", list(models.keys()))
model = models[model_name]

# -------------------------------------------------
# Load default test data
# -------------------------------------------------
df = pd.read_csv("data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# -------------------------------------------------
# Upload test CSV (HYBRID evaluation)
# -------------------------------------------------
st.subheader("📤 Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload labeled test CSV (same features + target)",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    if "target" not in test_df.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
        st.stop()

    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]
    st.info("📌 Evaluating on uploaded test dataset")
else:
    X_test = X
    y_test = y
    st.info("📌 Evaluating on internal dataset")

# -------------------------------------------------
# Feature scaling (ONLY where required)
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
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

st.subheader("📊 Evaluation Metrics")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, round(v, 3))

# -------------------------------------------------
# Confusion Matrix (FIXED SIZE)
# -------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    annot_kws={"size": 12},
    ax=ax
)

ax.set_xlabel("Predicted", fontsize=10)
ax.set_ylabel("Actual", fontsize=10)
ax.set_title("Confusion Matrix", fontsize=12)

st.pyplot(fig, use_container_width=False)
