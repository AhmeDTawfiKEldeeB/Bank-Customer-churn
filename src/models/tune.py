import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, accuracy_score


def tune_model(
    X,
    y,
    threshold: float = 0.3,
    min_accuracy: float = 0.75,
    n_trials: int = 30
):
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0, 3),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        recalls, accs = [], []

        for tr, val in cv.split(X, y):
            model = XGBClassifier(**params)
            model.fit(X.iloc[tr], y.iloc[tr])

            proba = model.predict_proba(X.iloc[val])[:, 1]
            preds = (proba >= threshold).astype(int)

            recalls.append(recall_score(y.iloc[val], preds))
            accs.append(accuracy_score(y.iloc[val], preds))

        mean_recall = np.mean(recalls)
        mean_acc = np.mean(accs)

        if mean_acc < min_accuracy:
            return mean_recall - (min_accuracy - mean_acc) * 2

        return mean_recall

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best Recall:", study.best_value)
    print("Best Params:", study.best_params)

    return study.best_params
