import os
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    Returns:
        pd.DataFrame: Loaded dataset
    """

    # Check if file exists before loading
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Read CSV file
    return pd.read_csv(path)
