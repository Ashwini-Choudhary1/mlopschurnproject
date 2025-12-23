from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(title="Churn Prediction API")

# --------------------------------------------------
# Paths
# --------------------------------------------------
MODEL_PATH = Path("models/churn_model.pkl")
ENCODER_DIR = Path("models/encoders")

# --------------------------------------------------
# Load model & encoders ONCE at startup
# --------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    contract_encoder = joblib.load(ENCODER_DIR / "Contract_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Model or encoder loading failed: {e}")

# --------------------------------------------------
# Request schema (input validation)
# --------------------------------------------------
class ChurnRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(request: ChurnRequest):
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([request.dict()])

        # Encode categorical feature
        df["Contract"] = contract_encoder.transform(df["Contract"])

        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
