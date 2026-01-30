"""
End-to-End Churn Prediction Pipeline
------------------------------------
Steps:
1. Load data
2. Preprocess data
3. Feature engineering
4. (Optional) Hyperparameter tuning
5. Train final model
6. Evaluate model
"""

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.tune import tune_model
from src.models.train import train_model
from src.models.evaluate import evaluate_model

from sklearn.model_selection import train_test_split


# ==========================
# CONFIGURATION
# ==========================
DATA_PATH = "data/train.csv"
TARGET_COL = "Exited"
THRESHOLD = 0.3
RUN_TUNING = False        
OUTLIER_COLS = ["Age", "CreditScore"]


def run_pipeline():
    print("🚀 Starting churn prediction pipeline...")
    
    #  Load data

    df = load_data(DATA_PATH)
    print(f"✅ Data loaded: {df.shape}")

    # Preprocess

    df = preprocess_data(df, target_col=TARGET_COL)
    print(f"✅ Preprocessing done: {df.shape}")

    # Feature Engineering
  
    df = build_features(
        df,
        target_col=TARGET_COL,
        outlier_cols=OUTLIER_COLS
    )
    print(f"✅ Feature engineering done: {df.shape}")

    # (Optional) Hyperparameter tuning

    if RUN_TUNING:
        print("🔍 Running Optuna hyperparameter tuning...")
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        best_params = tune_model(
            X,
            y,
            threshold=THRESHOLD,
            min_accuracy=0.75,
            n_trials=30
        )

        print("🏆 Best parameters found:")
        print(best_params)

    # Train final model

    model = train_model(
        df,
        target_col=TARGET_COL,
        threshold=THRESHOLD
    )

    # Evaluation (holdout)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("📊 Final Evaluation:")
    evaluate_model(model, X_test, y_test, threshold=THRESHOLD)

    print("🎉 Pipeline finished successfully!")


if __name__ == "__main__":
    run_pipeline()
