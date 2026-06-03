# =========================
# Imports
# =========================

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# =========================
# Data Loading
# =========================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """

    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully: {df.shape}")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    except Exception as e:
        raise Exception(f"Error loading data: {e}")


# =========================
# Validation
# =========================

def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate dataset structure before processing.
    """

    if df is None or df.empty:
        raise ValueError("Dataset is empty")

    required_cols = ["CustomerId", "Amount", "TransactionStartTime"]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return True




# =========================
# Column Dropper
# =========================

class DropColumns(BaseEstimator, TransformerMixin):
    """
    Drop identifier columns that are not useful for modeling.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        drop_cols = [
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId"
        ]

        df = df.drop(
            columns=[c for c in drop_cols if c in df.columns],
            errors="ignore"
        )

        return df


# =========================
# Aggregate Features
# =========================

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Create customer-level aggregate transaction features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        agg = df.groupby("CustomerId")["Amount"].agg(
            TotalTransactionAmount="sum",
            AverageTransactionAmount="mean",
            TransactionCount="count",
            StdTransactionAmount="std"
        ).reset_index()

        agg["StdTransactionAmount"] = agg["StdTransactionAmount"].fillna(0)

        df = df.merge(agg, on="CustomerId", how="left")

        return df


# =========================
# Date Features
# =========================

class DateFeatures(BaseEstimator, TransformerMixin):
    """
    Extract time-based features from transaction timestamp.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        df["TransactionHour"] = df["TransactionStartTime"].dt.hour
        df["TransactionDay"] = df["TransactionStartTime"].dt.day
        df["TransactionMonth"] = df["TransactionStartTime"].dt.month
        df["TransactionYear"] = df["TransactionStartTime"].dt.year

        return df


# =========================
# Pipeline Builder
# =========================

def build_pipeline(df: pd.DataFrame):
    """
    Build full preprocessing pipeline.
    """

    numeric_features = [
        "Amount",
        "Value",
        "CountryCode",
        "TotalTransactionAmount",
        "AverageTransactionAmount",
        "TransactionCount",
        "StdTransactionAmount",
        "TransactionHour",
        "TransactionDay",
        "TransactionMonth",
        "TransactionYear"
    ]

    categorical_features = [
        "CurrencyCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy"
    ]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    full_pipeline = Pipeline(steps=[
        ("drop_columns", DropColumns()),
        ("aggregate_features", AggregateFeatures()),
        ("date_features", DateFeatures()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline


# =========================
# Main Processing Function
# =========================

def process_data(df: pd.DataFrame):
    """
    Execute full feature engineering pipeline.
    """

    validate_dataset(df)

    pipeline = build_pipeline(df)

    processed_data = pipeline.fit_transform(df)

    return processed_data