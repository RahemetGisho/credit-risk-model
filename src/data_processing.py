import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Load dataset from CSV file.
    """

    try:
        df = pd.read_csv(filepath)
        print("Dataset loaded successfully.")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    except Exception as e:
        raise Exception(
            f"Unexpected error: {e}"
        )


def validate_dataset(df):
    """
    Validate dataframe before analysis.
    """

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if len(df.columns) == 0:
        raise ValueError(
            "Dataset contains no columns."
        )

    print("Dataset validation passed.")


def get_numerical_columns(df):
    """
    Return numerical columns.
    """

    return df.select_dtypes(
        include=np.number
    ).columns


def get_categorical_columns(df):
    """
    Return categorical columns.
    """

    return df.select_dtypes(
        include="object"
    ).columns


def summarize_data(df):
    """
    Generate summary statistics.
    """

    return df.describe().T


def missing_values(df):
    """
    Return missing value summary.
    """

    return (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )