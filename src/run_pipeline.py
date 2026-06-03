import os
import pandas as pd

from src.data_processing import (
    load_data,
    process_data
)

df = load_data(
    "data/raw/data.csv"
)

processed_matrix, processed_df = process_data(df)

processed_df.to_csv(
    "data/processed/processed_data.csv",
    index=False
)

print("Processed dataset saved successfully.")


print("Raw columns:", df.columns)
print("Target column exists:", "is_high_risk" in df.columns)
print(df[["CustomerId", "is_high_risk"]].head())