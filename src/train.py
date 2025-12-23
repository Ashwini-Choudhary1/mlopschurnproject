from src.datapreprocessing import preprocess
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "churn_model.pkl"

def train():

    mlflow.set_experiment("churn_baseline_model")

    with mlflow.start_run():

        #preprocess()

        X_train = pd.read_csv(PROCESSED_DIR / "Xtrain.csv")
        X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

        y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()
        y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()

        n_estimators = 100

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            class_weight="balanced"
        )

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("class_weight", "balanced")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        print("📊 Classification Report")
        print(classification_report(y_test, y_pred))

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        mlflow.log_artifact(MODEL_PATH)

        print("Training completed and logged to MLflow")

if __name__ == "__main__":
    train()
