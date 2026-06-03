import os
import pandas as pd

from src.data_processing import (
    load_data,
    process_data
)

# Load raw data
df = load_data("data/raw/data.csv")

# Run pipeline
processed_data = process_data(df)

# Convert to DataFrame
processed_df = pd.DataFrame(processed_data)

# Save output
processed_df.to_csv(
    "data/processed/processed_data.csv",
    index=False
)

print("Processing complete!")
print("Saved to data/processed/processed_data.csv")
print(processed_df.shape)