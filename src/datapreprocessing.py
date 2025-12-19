import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

RAW_DATA_PATH = Path("data/raw/customerchurn.csv")
PROCESSED_DIR = Path("data/processed")
ENCODER_DIR = Path("models/encoders")

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = ["Contract"]
TARGET_COL = "Churn"

def preprocess():
    df = pd.read_csv(RAW_DATA_PATH)

    df = df[NUMERIC_COLS + CAT_COLS + [TARGET_COL]]

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)

    df.dropna(inplace=True)

    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    encoders = {}

    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        joblib.dump(le, ENCODER_DIR / f"{col}_encoder.pkl")

    target_encoder = LabelEncoder()
    df[TARGET_COL] = target_encoder.fit_transform(df[TARGET_COL])
    joblib.dump(target_encoder, ENCODER_DIR / "target_encoder.pkl")

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DIR / "Xtrain.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("✅ Data preprocessing completed")

if __name__ == "__main__":
    preprocess()
