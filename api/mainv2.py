from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------
# App
# ---------------------------------------
app = FastAPI(title="Churn Prediction API - v2")

# ---------------------------------------
# Paths (v2 artifacts ONLY)
# ---------------------------------------
MODEL_PATH = Path("models/churn_model_v2.pkl")
ENCODER_DIR = Path("models/encoders_v2")

# ---------------------------------------
# Load model & encoders
# ---------------------------------------
try:
    model = joblib.load(MODEL_PATH)

    encoders = {
        "Contract": joblib.load(ENCODER_DIR / "Contract_encoder.pkl"),
        "PhoneService": joblib.load(ENCODER_DIR / "PhoneService_encoder.pkl"),
        "Dependents": joblib.load(ENCODER_DIR / "Dependents_encoder.pkl"),
        "Partner": joblib.load(ENCODER_DIR / "Partner_encoder.pkl"),
    }

except Exception as e:
    raise RuntimeError(f"Failed to load v2 artifacts: {e}")

# ---------------------------------------
# Request schema
# ---------------------------------------
class ChurnRequestV2(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PhoneService: str
    Dependents: str
    Partner: str

# ---------------------------------------
# Health
# ---------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "model": "v2"}

# ---------------------------------------
# Prediction
# ---------------------------------------
@app.post("/predict")
def predict(request: ChurnRequestV2):
    try:
        df = pd.DataFrame([request.dict()])

        # Encode categoricals
        for col, encoder in encoders.items():
            df[col] = encoder.transform(df[col])

        # Predict
        prob = model.predict_proba(df)[0][1]
        pred = int(prob > 0.4)

        return {
            "churn_prediction": pred,
            "churn_probability": round(prob, 3),
            "model_version": "v2"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
