import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

_model = None
_preprocessor = None

def load_artifacts():
    global _model, _preprocessor
    if _model is None:
        _model = joblib.load(BASE_DIR / "model.joblib")
    if _preprocessor is None:
        _preprocessor = joblib.load(BASE_DIR / "preprocessor.joblib")

def predict(data: dict):
    load_artifacts()

    df = pd.DataFrame([data])
    X = _preprocessor.transform(df)

    pred = _model.predict(X)[0]
    prob = _model.predict_proba(X)[0][1]

    return {
        "label": "Churn" if pred == 1 else "Not Churn",
        "churn": bool(pred),
        "probability": round(float(prob), 3)
    }
