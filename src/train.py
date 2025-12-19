from src.datapreprocessing import preprocess
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import (classification_report,roc_auc_score,confusion_matrix,precision_score,recall_score)
from pathlib import Path
import mlflow
import mlflow.sklearn

processed_dir = Path("data/processed")
model_dir = Path("models")
model_path = model_dir/"churnmodel.pkl"

def train():

    mlflow.set_experiment("basic churn line model")

    with mlflow.start_run():
        preprocess()

        X_train = pd.read_csv(processed_dir/"Xtrain.csv")
        X_test = pd.read_csv(processed_dir/"X_test.csv")
        y_train = pd.read_csv(processed_dir/"y_train.csv")
        y_test = pd.read_csv(processed_dir/"y_test.csv")
        n_estimators =100
        random_state =42

        model = RandomForestClassifier(class_weight="balanced",n_estimators= n_estimators,random_state=random_state)
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("class_weight", "balanced")

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:,1]

        # Metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)


        print("Classification Report:")
        print(classification_report(y_test, y_pred))


        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        # Log model artifact to MLflow
        mlflow.log_artifact(model_path)

        print("Training completed and logged to MLflow")

if __name__ == "__main__":
    train()

