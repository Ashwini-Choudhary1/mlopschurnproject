# Customer Churn Prediction – MLOps Project

End-to-end MLOps project for predicting customer churn using machine learning.
The project includes data preprocessing, model training, experiment tracking with MLflow,
and model serving using FastAPI.

--------------------------------------------------

PROJECT STRUCTURE

mlopschurnproject/
├── api/
│   ├── main.py        (Baseline FastAPI)
│   └── mainv2.py      (Improved FastAPI)
├── src/
│   ├── datapreprocessing.py
│   ├── datapreprocessingv2.py
│   ├── train.py
│   ├── trainv2.py
│   └── __init__.py
├── data/
│   ├── raw/customerchurn.csv
│   └── processed_v2/
├── models/
│   ├── encoders/
│   ├── encoders_v2/
│   ├── churn_model.pkl
│   └── churn_model_v2.pkl
├── notebooks/
│   ├── eda01.ipynb
│   └── eda_second.ipynb
├── mlruns/
├── requirements.txt
├── .gitignore
└── README.md

--------------------------------------------------

MODELS

Baseline Model (v1)
- Algorithm: Random Forest
- Features: tenure, MonthlyCharges, TotalCharges, Contract

Improved Model (v2)
- Algorithm: Logistic Regression
- Added features: Contract, PhoneService, Dependents, Partner
- Class imbalance handled using SMOTE
- Experiment tracking with MLflow

--------------------------------------------------

TRAINING

Activate virtual environment:
source .venv/bin/activate

Train improved model:
python src/trainv2.py

--------------------------------------------------

MLFLOW

Start MLflow UI:
mlflow ui

Open in browser:
http://127.0.0.1:5000

--------------------------------------------------

FASTAPI

Run baseline API:
uvicorn api.main:app --reload

Run improved model API:
uvicorn api.mainv2:app --reload

Swagger UI:
http://127.0.0.1:8000/docs

--------------------------------------------------

SAMPLE REQUEST

{
  "tenure": 12,
  "MonthlyCharges": 75.5,
  "TotalCharges": 900.0,
  "Contract": "Month-to-month",
  "PhoneService": "Yes",
  "Dependents": "No",
  "Partner": "Yes"
}

--------------------------------------------------

TECH STACK

Python
pandas
scikit-learn
imbalanced-learn
MLflow
FastAPI
Uvicorn
Joblib

--------------------------------------------------

AUTHOR

Ashwini Choudhary
