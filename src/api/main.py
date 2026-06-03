# =========================
# FastAPI Credit Risk Service
# =========================

from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np

# Import Pydantic models for validation
from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse
)

# =========================
# Initialize FastAPI app
# =========================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting customer credit risk using MLflow model",
)

# =========================
# Load Best Model from MLflow Registry
# =========================

# Name must match what you registered in MLflow
MODEL_NAME = "CreditRiskModel"
MODEL_VERSION = "1"

# Load model once at startup (important for performance)
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)

# =========================
# Health Check Endpoint
# =========================

@app.get("/")
def home():
    return {
        "message": "Credit Risk API is running"
    }

# =========================
# Prediction Endpoint
# =========================

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    X = np.array([[
        request.recency,
        request.frequency,
        request.monetary
    ]])

    prediction = model.predict(X)

    risk_probability = float(prediction[0])
    risk_label = 1 if risk_probability >= 0.5 else 0

    return PredictionResponse(
        risk_probability=risk_probability,
        risk_label=risk_label
    )