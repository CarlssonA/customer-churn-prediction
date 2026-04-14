import argparse
import pandas as pd
import mlflow
import joblib
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score, classification_report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    # Daten laden
    X_test = pd.read_csv(f"{args.test_data}/X_test.csv")
    y_test = pd.read_csv(f"{args.test_data}/y_test.csv").squeeze()

    # Modell laden
    model = joblib.load(f"{args.model_path}/model.pkl")

      with mlflow.start_run():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= args.threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        auc      = roc_auc_score(y_test, y_prob)
        recall   = recall_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("churn_recall", recall)

        print(f"Accuracy:     {accuracy:.4f}")
        print(f"AUC:          {auc:.4f}")
        print(f"Churn Recall: {recall:.4f}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()