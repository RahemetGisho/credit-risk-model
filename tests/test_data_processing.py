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