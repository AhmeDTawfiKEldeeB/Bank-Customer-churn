import pandas as pd

def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Exited"
) -> pd.DataFrame:
    """
    Perform basic preprocessing steps:
    - Clean column names
    - Remove duplicates
    - Handle missing values
    - Drop non-informative columns
    - Encode target column if categorical
    """

    df = df.copy()

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Identify numeric and categorical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Fill missing numeric values with median
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with mode
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Drop ID-like columns (not useful for ML)
    for col in ["id", "CustomerId", "Surname"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Encode target column if it's categorical
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = (
            df[target_col]
            .str.strip()
            .map({"Yes": 1, "No": 0})
            .astype(int)
        )

    return df
