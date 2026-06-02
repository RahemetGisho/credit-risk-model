import pandas as pd
import pytest

from src.data_processing import (
    validate_dataset
)


def test_validate_dataset():

    df = pd.DataFrame({
        "A": [1, 2, 3]
    })

    assert validate_dataset(df) is None


def test_empty_dataframe():

    df = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_dataset(df)