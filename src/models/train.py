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
    """
    Train final XGBoost model and log results to MLflow.
    """

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle class imbalance
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Final tuned model
    model = XGBClassifier(
        n_estimators=702,
        learning_rate=0.08710825958109349,
        max_depth=7,
        subsample=0.8982473714059436,
        colsample_bytree=0.6694400666869094,
        min_child_weight=8,
        gamma=0.514080137124345,
        reg_alpha=0.44403106721449837,
        reg_lambda=1.6250927290660933,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    mlflow.set_experiment("churn-xgboost")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        # Threshold-based predictions
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= threshold).astype(int)

        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        # Log experiment data
        mlflow.log_params(model.get_params())
        mlflow.log_param("threshold", threshold)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.xgboost.log_model(model, name="xgboost-model")

        print(f"Accuracy={acc:.3f} | Recall={rec:.3f}")

    return model
