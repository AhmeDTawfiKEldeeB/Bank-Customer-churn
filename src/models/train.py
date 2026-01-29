import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score


def train_model(
    df: pd.DataFrame,
    target_col: str = "Exited",
    threshold: float = 0.3
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=682,
        learning_rate=0.15,
        max_depth=8,
        subsample=0.67,
        colsample_bytree=0.78,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    mlflow.set_experiment("churn-xgboost")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= threshold).astype(int)

        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        mlflow.log_params(model.get_params())
        mlflow.log_param("threshold", threshold)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.xgboost.log_model(model, name="xgboost-model")

        print(f"Accuracy={acc:.3f} | Recall={rec:.3f}")

    return model
