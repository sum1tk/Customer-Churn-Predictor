"""
train.py
Trains a customer churn classifier and saves it as model.joblib.
Exits with a non-zero code if accuracy is below threshold — this is
what lets the Test stage of the pipeline fail the build automatically.
"""
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ACCURACY_THRESHOLD = 0.80

def load_data():
    # Replace this with: pd.read_csv("data/churn.csv") in a real project
    X, y = make_classification(
        n_samples=2000, n_features=12, n_informative=8,
        weights=[0.7, 0.3], random_state=42
    )
    return X, y

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")

    if acc < ACCURACY_THRESHOLD:
        print(f"FAILED: accuracy {acc:.4f} below threshold {ACCURACY_THRESHOLD}")
        sys.exit(1)   # <-- this failure stops the CodeBuild/CodePipeline stage

    print("Model validation PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()
