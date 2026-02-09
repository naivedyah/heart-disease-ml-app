import pandas as pd
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
    recall_score, f1_score, matthews_corrcoef
)

# Load dataset
df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), X_train_scaled, X_test_scaled),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), X_train, X_test),
    "KNN": (KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled),
    "Naive Bayes": (GaussianNB(), X_train_scaled, X_test_scaled),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
    "XGBoost": (XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42), X_train, X_test),
}

print("\nModel Performance (Internal 20% Test Split)\n")

for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    print(f"{name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("-" * 40)
