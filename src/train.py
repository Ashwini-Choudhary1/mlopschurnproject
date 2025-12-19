import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import (classification_report,roc_auc_score,confusion_matrix)
from pathlib import Path

processed_dir = Path("/Users/ashwinichoudhary/mlopschurnproject/data/processed")
model_dir = Path("models")
model_path = model_dir/"churnmodel.pkl"

def train():

    X_train = pd.read_csv(processed_dir/"Xtrain.csv")
    X_test = pd.read_csv(processed_dir/"X_test.csv")
    y_train = pd.read_csv(processed_dir/"y_train.csv")
    y_test = pd.read_csv(processed_dir/"y_test.csv")

    model = RandomForestClassifier(class_weight="balanced",n_estimators=100,random_state=42)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:,1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"💾 Model saved to {model_path}")

if __name__ == "__main__":
    train()

