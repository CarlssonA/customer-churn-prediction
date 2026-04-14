import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score, classification_report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()

    X_train = pd.read_csv(f"{args.data_path}/X_train.csv")
    X_test  = pd.read_csv(f"{args.data_path}/X_test.csv")
    y_train = pd.read_csv(f"{args.data_path}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{args.data_path}/y_test.csv").squeeze()

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

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