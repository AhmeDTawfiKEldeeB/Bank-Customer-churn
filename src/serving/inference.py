import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

BASE_DIR = Path(__file__).resolve().parent
_model = None
_preprocessor = None
_use_local_preprocessing = False


def load_artifacts():
    global _model, _preprocessor, _use_local_preprocessing

    # Primary expected locations
    model_path = BASE_DIR / "model.joblib"
    preproc_path = BASE_DIR / "preprocessor.joblib"

    # Fallback to repo-level models/
    if not model_path.exists() or not preproc_path.exists():
        repo_root = Path(__file__).resolve().parents[2]
        alt_models_dir = repo_root / "models"
        alt_model = alt_models_dir / "model.joblib"
        alt_preproc = alt_models_dir / "preprocessor.joblib"

        if alt_model.exists():
            model_path = alt_model
        if alt_preproc.exists():
            preproc_path = alt_preproc

    # Also allow using the notebook JSON model to match notebook predictions
    repo_root = Path(__file__).resolve().parents[2]
    notebook_json = repo_root / "notebooks" / "models" / "final_xgboost_model.json"

    if _model is None:
        try:
            if notebook_json.exists():
                m = XGBClassifier()
                m.load_model(str(notebook_json))
                _model = m
            else:
                _model = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed loading model from {model_path} (or notebook JSON): {e}")

    if _preprocessor is None:
        try:
            _preprocessor = joblib.load(preproc_path)
        except ModuleNotFoundError:
            # Fall back to using local preprocessing functions instead of
            # failing on unpickling a custom transformer class that doesn't
            # exist in this repo.
            _preprocessor = None
            _use_local_preprocessing = True
        except Exception as e:
            # If preprocessor file is missing or corrupted, use local pipeline
            _preprocessor = None
            _use_local_preprocessing = True


def _align_and_prepare(df: pd.DataFrame):
    """Apply local preprocessing and align columns to model feature names."""
    # local preprocessing
    df_p = preprocess_data(df)
    df_f = build_features(df_p)

    # small fixes for common object columns
    if 'Gender' in df_f.columns and df_f['Gender'].dtype == object:
        df_f['Gender'] = df_f['Gender'].map({'Female': 0, 'Male': 1}).fillna(0).astype(int)

    # Align to model's expected features
    try:
        model_feature_names = list(_model.feature_names_in_)
    except Exception:
        try:
            model_feature_names = _model.get_booster().feature_names
        except Exception:
            model_feature_names = list(df_f.columns)

    for col in model_feature_names:
        if col not in df_f.columns:
            df_f[col] = 0

    df_f = df_f.reindex(columns=model_feature_names, fill_value=0)

    # Ensure numeric types
    for c in df_f.columns:
        if df_f[c].dtype == object:
            df_f[c] = pd.to_numeric(df_f[c], errors='coerce').fillna(0)

    return df_f


def predict(data: dict):
    """Return prediction dict with Arabic label, churn bool and probability."""
    load_artifacts()

    df = pd.DataFrame([data])

    # Use pickled preprocessor if available and valid
    if _preprocessor is not None and not _use_local_preprocessing:
        try:
            X = _preprocessor.transform(df)
        except Exception:
            # fallback to local pipeline on transform error
            X = _align_and_prepare(df)
    else:
        X = _align_and_prepare(df)

    prob = float(_model.predict_proba(X)[0][1])
    pred = 1 if prob > 0.5 else 0

    return {
        "label": "Churn" if pred == 1 else "Not Churn",
        "churn": bool(pred),
        "probability": round(prob, 3),
    }

