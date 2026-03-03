# src/data_loader.py
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Get project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "GenerationProdTypeExchange.csv"
INTERIM_PATH = (
    PROJECT_ROOT / "data" / "interim" / "GenerationProdTypeExchange_clean.parquet"
)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    def to_snake(s: str) -> str:
        s = s.strip()
        s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        s = s.replace(" ", "_").replace("-", "_")
        s = re.sub(r"__+", "_", s)
        return s.lower()

    df = df.copy()
    df.columns = [to_snake(c) for c in df.columns]
    return df


def parse_european_numbers(series: pd.Series) -> pd.Series:
    """
    Parse numbers like '3.646,3' to 3646.3
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    series = series.astype(str)

    series = series.str.replace(r"\.(?=\d{3})", "", regex=True)

    series = series.str.replace(",", ".", regex=False)

    return pd.to_numeric(series, errors="coerce")


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # normalize column names
    df = normalize_column_names(df)

    # parse timestamps
    if "time_utc" in df.columns:
        df["time_utc"] = df["time_utc"].str.replace("Z", "", regex=False)
        df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)

    if "time_dk" in df.columns:
        df["time_dk"] = pd.to_datetime(df["time_dk"], errors="coerce")

    # parse european numbers in all columns except timestamps and categorical ones
    exclude_cols = ["time_utc", "time_dk", "price_area", "version"]

    for col in df.columns:
        if col not in exclude_cols:
            df[col] = parse_european_numbers(df[col])

    # drop rows with missing timestamps (if any)
    df = df.dropna(subset=["time_utc"])

    # filter to only final version
    df = df[df["version"] == "Final"]

    # sort by price area and time
    df = df.sort_values(["price_area", "time_utc"])

    return df.reset_index(drop=True)


def save_interim(df: pd.DataFrame, path: Path = INTERIM_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run():
    print("Loading raw data...")
    df_raw = load_raw_data()
    print(df_raw.head())
    print(df_raw.columns)
    print("Cleaning data...")
    df_clean = clean_data(df_raw)

    print("Saving cleaned dataset...")
    save_interim(df_clean)

    print("Done!")
    print(f"Rows: {len(df_clean):,}")
    print(f"Columns: {df_clean.shape[1]}")


if __name__ == "__main__":
    run()
