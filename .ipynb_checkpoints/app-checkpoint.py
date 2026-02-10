import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")

st.info(
    "Hybrid Evaluation Mode: Default evaluation uses an internal 80–20 split. "
    "Optionally upload the labeled test.csv to reproduce the same results."
)

@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

df = load_data()

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_internal_test, y_train, y_internal_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

st.subheader("📤 Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload labeled test CSV (same features + target)",
    type=["csv"]
)

use_uploaded = False

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    if "target" in uploaded_df.columns:
        X_test = uploaded_df.drop("target", axis=1)
        y_test = uploaded_df["target"]
        use_uploaded = True
    else:
        st.warning("No target column found. Falling back to internal test split.")
        X_test = X_internal_test
        y_test = y_internal_test
else:
    X_test = X_internal_test
    y_test = y_internal_test

X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
}

model_name = st.selectbox("Select ML Model", list(models.keys()))
model = models[model_name]

if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
    model.fit(X_train_scaled, y_train)
    X_eval = X_test_scaled
else:
    model.fit(X_train, y_train)
    X_eval = X_test

y_pred = model.predict(X_eval)
y_prob = model.predict_proba(X_eval)[:, 1]

if use_uploaded:
    st.info("📌 Evaluating on uploaded test dataset")
else:
    st.info("📌 Evaluating on internal 20% test split")

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

st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
