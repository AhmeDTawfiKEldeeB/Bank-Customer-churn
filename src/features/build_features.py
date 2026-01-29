import pandas as pd


def map_binary_series(s: pd.Series) -> pd.Series:
    vals = set(s.dropna().astype(str).unique())

    if vals == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1})

    if vals == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1})

    if vals == {"True", "False"}:
        return s.map({"False": 0, "True": 1})

    if len(vals) == 2:
        sorted_vals = sorted(vals)
        return s.astype(str).map({sorted_vals[0]: 0, sorted_vals[1]: 1})

    return s


def _handle_outliers_iqr(s: pd.Series, factor: float = 1.5) -> pd.Series:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return s.clip(lower, upper)


def build_features(
    df: pd.DataFrame,
    target_col: str = "Exited",
    outlier_cols: list | None = None
) -> pd.DataFrame:

    df = df.copy()

    cat_cols = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c != target_col
    ]
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if outlier_cols:
        for c in outlier_cols:
            if c in df.columns:
                df[c] = _handle_outliers_iqr(df[c])

    binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
    multi_cols = [c for c in cat_cols if df[c].nunique() > 2]

    for c in binary_cols:
        df[c] = map_binary_series(df[c].astype(str)).fillna(0).astype(int)

    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    return df
