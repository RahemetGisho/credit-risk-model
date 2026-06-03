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
from sklearn.cluster import KMeans


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

def validate_dataset(df: pd.DataFrame):
    """
    Validate dataset structure before processing.
    Returns None if valid, raises error if invalid.
    """

    if df is None or df.empty:
        raise ValueError("Dataset is empty")

    required_cols = [
        "CustomerId",
        "Amount",
        "TransactionStartTime"
    ]

    missing = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )
    


# =========================
# Task 4 - RFM Metrics
# =========================

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Recency, Frequency,
    and Monetary values.
    """

    rfm_df = df.copy()
    

    rfm_df["TransactionStartTime"] = pd.to_datetime(
        rfm_df["TransactionStartTime"]
    )
    rfm_df["Amount"] = rfm_df["Amount"].abs()

    snapshot_date = (
        rfm_df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        rfm_df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=(
                "TransactionId",
                "count"
            ),
            Monetary=(
                "Amount",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm


# =========================
# Task 4 - Customer Clustering
# =========================

def cluster_customers(
    rfm: pd.DataFrame
) -> pd.DataFrame:
    """
    Cluster customers using RFM metrics.
    """

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    return rfm


# =========================
# Task 4 - Risk Label
# =========================

def assign_high_risk_label(
    rfm: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify least engaged cluster
    and create target variable.
    """

    cluster_summary = (
        rfm.groupby("Cluster")
        [
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
        .mean()
    )

    print("\nCluster Summary")
    print(cluster_summary)

    high_risk_cluster = (
        cluster_summary
        .sort_values(
            by=[
                "Frequency",
                "Monetary"
            ],
            ascending=True
        )
        .index[0]
    )

    print(
        f"\nHigh-risk cluster: "
        f"{high_risk_cluster}"
    )

    rfm["is_high_risk"] = (
        rfm["Cluster"]
        == high_risk_cluster
    ).astype(int)

    return rfm


# =========================
# Task 4 - Merge Target
# =========================

def merge_risk_label(
    df: pd.DataFrame,
    rfm: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge target variable into
    transaction dataset.
    """

    return df.merge(
        rfm[
            [
                "CustomerId",
                "is_high_risk"
            ]
        ],
        on="CustomerId",
        how="left"
    )


# =========================
# Column Dropper
# =========================

class DropColumns(
    BaseEstimator,
    TransformerMixin
):
    """
    Drop identifier columns.
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
            columns=[
                c for c in drop_cols
                if c in df.columns
            ],
            errors="ignore"
        )

        return df


# =========================
# Aggregate Features
# =========================

class AggregateFeatures(
    BaseEstimator,
    TransformerMixin
):
    """
    Create aggregate transaction features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        agg = (
            df.groupby("CustomerId")["Amount"]
            .agg(
                TotalTransactionAmount="sum",
                AverageTransactionAmount="mean",
                TransactionCount="count",
                StdTransactionAmount="std"
            )
            .reset_index()
        )

        agg["StdTransactionAmount"] = (
            agg["StdTransactionAmount"]
            .fillna(0)
        )

        df = df.merge(
            agg,
            on="CustomerId",
            how="left"
        )

        return df


# =========================
# Date Features
# =========================

class DateFeatures(
    BaseEstimator,
    TransformerMixin
):
    """
    Extract datetime features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"]
        )

        df["TransactionHour"] = (
            df["TransactionStartTime"].dt.hour
        )

        df["TransactionDay"] = (
            df["TransactionStartTime"].dt.day
        )

        df["TransactionMonth"] = (
            df["TransactionStartTime"].dt.month
        )

        df["TransactionYear"] = (
            df["TransactionStartTime"].dt.year
        )

        return df


# =========================
# Pipeline Builder
# =========================

def build_pipeline(df: pd.DataFrame):

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

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    full_pipeline = Pipeline(
        steps=[
            (
                "drop_columns",
                DropColumns()
            ),
            (
                "aggregate_features",
                AggregateFeatures()
            ),
            (
                "date_features",
                DateFeatures()
            ),
            (
                "preprocessor",
                preprocessor
            )
        ]
    )

    return full_pipeline


# =========================
# Main Processing Function
# =========================

def process_data(df: pd.DataFrame):
    """
    Execute Task 3 + Task 4 pipeline.
    """

    validate_dataset(df)

    # Task 4
    rfm = calculate_rfm(df)

    rfm = cluster_customers(rfm)

    rfm = assign_high_risk_label(rfm)

    df = merge_risk_label(
        df,
        rfm
    )

    # Task 3
    pipeline = build_pipeline(df)

    processed_data = (
        pipeline.fit_transform(df)
    )

    return processed_data, df