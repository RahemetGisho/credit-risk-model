# =========================
# Batch / Local Prediction Script
# =========================

# This file is used for testing the trained MLflow model locally
# before deploying it into the FastAPI service.

import mlflow.pyfunc
import pandas as pd


# =========================
# Load Model from MLflow Registry
# =========================
def load_model(model_name="CreditRiskModel", stage="Production"):
    """
    Load trained model from MLflow Model Registry.

    model_name → name you registered in MLflow
    stage → version stage (e.g. Production, Staging, or version number)
    """

    # Construct MLflow model URI
    model_uri = f"models:/{model_name}/{stage}"

    # Load model using MLflow pyfunc format
    model = mlflow.pyfunc.load_model(model_uri)

    return model


# =========================
# Run Prediction
# =========================
def predict(model, input_data: pd.DataFrame):
    """
    Run inference using loaded MLflow model.
    """

    # Model expects dataframe input
    predictions = model.predict(input_data)

    return predictions


# =========================
# Main Execution Block
# =========================
def main():

    # -------------------------
    # Example input data
    # -------------------------
    # This MUST match training feature order
    sample_data = pd.DataFrame([{
        "Recency": 10,
        "Frequency": 5,
        "Monetary": 20000,
        "Age": 45,
        "Income": 120000
    }])

    # -------------------------
    # Load trained model
    # -------------------------
    model = load_model()

    # -------------------------
    # Make prediction
    # -------------------------
    predictions = predict(model, sample_data)

    # -------------------------
    # Output results
    # -------------------------
    print("\n=========================")
    print("Credit Risk Prediction")
    print("=========================")
    print(predictions)


# =========================
# Run script
# =========================
if __name__ == "__main__":
    main()