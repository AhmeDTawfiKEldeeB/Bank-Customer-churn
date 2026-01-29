import pandas as pd

def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Exited"
) -> pd.DataFrame:
    """
    Basic preprocessing:
    - Clean column names
    - Drop duplicates
    - Handle missing values
    - Drop ID-like columns
    - Encode target if categorical
    """

    df = df.copy()
    df.columns = df.columns.str.strip()
    df.drop_duplicates(inplace=True)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in ["id", "CustomerId", "Surname"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = (
            df[target_col]
            .str.strip()
            .map({"Yes": 1, "No": 0})
            .astype(int)
        )

    return df
