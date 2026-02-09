
from xgboost import XGBClassifier

def build_model():
    return XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
