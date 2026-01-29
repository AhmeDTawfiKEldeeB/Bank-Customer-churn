from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, threshold=0.3):
    """
    Evaluate model performance using threshold-based predictions.
    """

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
