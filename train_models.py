
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from model.logistic_regression import build_model as lr_model
from model.decision_tree import build_model as dt_model
from model.knn import build_model as knn_model
from model.naive_bayes import build_model as nb_model
from model.random_forest import build_model as rf_model
from model.xgboost_model import build_model as xgb_model

df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# -------------------------------------------------
# Save the official test split for Streamlit app
# -------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values

test_df.to_csv("data/test.csv", index=False)
print("Saved official test split to data/test.csv")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Path("model").mkdir(exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

models = {
    "logistic_regression": (lr_model(), X_train_scaled, X_test_scaled),
    "decision_tree": (dt_model(), X_train, X_test),
    "knn": (knn_model(), X_train_scaled, X_test_scaled),
    "naive_bayes": (nb_model(), X_train_scaled, X_test_scaled),
    "random_forest": (rf_model(), X_train, X_test),
    "xgboost": (xgb_model(), X_train, X_test),
}

for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    print(name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
    print("-" * 30)

    joblib.dump(model, f"model/{name}.pkl")
