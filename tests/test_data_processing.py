import pandas as pd
import pytest

from src.data_processing import (
    validate_dataset
)


def test_validate_dataset_valid():
    df = pd.DataFrame({
        "CustomerId": [1],
        "Amount": [100],
        "TransactionStartTime": ["2024-01-01"]
    })
    assert validate_dataset(df) is None




def test_empty_dataframe():

    df = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_dataset(df)

def test_rfm_output_columns():
    from src.data_processing import calculate_rfm

    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionId": [10, 11, 12],
        "Amount": [100, 200, 300],
        "TransactionStartTime": pd.date_range("2024-01-01", periods=3)
    })

    rfm = calculate_rfm(df)

    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns


def test_cluster_assignment():
    from src.data_processing import cluster_customers

    df = pd.DataFrame({
        "CustomerId": [1, 2, 3],
        "Recency": [10, 20, 30],
        "Frequency": [5, 10, 15],
        "Monetary": [100, 200, 300]
    })

    result = cluster_customers(df)

    assert "Cluster" in result.columns
    