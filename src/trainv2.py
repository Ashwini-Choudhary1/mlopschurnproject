import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn

from src.datapreprocessingv2 import preprocess

PROCESSED_DIR = Path("data/processed_v2")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "churn_model_v2.pkl"

def train():
    mlflow.set_experiment("churn_model_v2_smote")

    with mlflow.start_run():
        preprocess()

        X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
        X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
        y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()
        y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()

        # ---- SMOTE ----
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        mlflow.log_param("smote", True)

        # ---- Model ----
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        model.fit(X_train_res, y_train_res)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.4).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("roc_auc", roc_auc)

        print(classification_report(y_test, y_pred))

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        print("Training v2 completed")

if __name__ == "__main__":
    train()
