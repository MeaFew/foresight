"""Data preprocessing for multivariate time series forecasting.

Handles:
- Date parsing and indexing
- Missing value interpolation
- Log-transform for sales (handle skewness)
- Train/validation split
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CLEANED_TRAIN_CSV,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RANDOM_STATE,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
)


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load all raw CSV files."""
    files = {
        "train": RAW_DATA_DIR / "train.csv",
        "stores": RAW_DATA_DIR / "stores.csv",
        "transactions": RAW_DATA_DIR / "transactions.csv",
        "oil": RAW_DATA_DIR / "oil.csv",
        "holidays": RAW_DATA_DIR / "holidays_events.csv",
        "test": RAW_DATA_DIR / "test.csv",
    }
    data = {}
    for name, path in files.items():
        if path.exists():
            data[name] = pd.read_csv(path)
            print(f"  Loaded {name}: {data[name].shape}")
        else:
            print(f"  Warning: {path} not found")
    return data


def preprocess_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform sales data."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Sort
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Log transform sales (add 1 to handle zeros)
    if "sales" in df.columns:
        df["sales_log"] = np.log1p(df["sales"])

    # Ensure onpromotion is int
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].astype(int)

    # Add time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    return df


def merge_external(df: pd.DataFrame, oil: pd.DataFrame | None, holidays: pd.DataFrame | None) -> pd.DataFrame:
    """Merge oil prices and holiday indicators."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if oil is not None:
        oil = oil.copy()
        oil["date"] = pd.to_datetime(oil["date"])
        oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="linear")
        df = df.merge(oil[["date", "dcoilwtico"]], on="date", how="left")
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    if holidays is not None:
        holidays = holidays.copy()
        holidays["date"] = pd.to_datetime(holidays["date"])
        # Create holiday flag
        holiday_dates = holidays[~holidays["transferred"]]["date"].unique()
        df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    else:
        df["is_holiday"] = 0

    return df


def preprocess(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    print("Preprocessing sales data ...")
    sales = preprocess_sales(data["train"])

    print("Merging external data ...")
    sales = merge_external(sales, data.get("oil"), data.get("holidays"))

    # Merge store metadata
    if "stores" in data:
        sales = sales.merge(data["stores"], on="store_nbr", how="left")

    # Merge transactions
    if "transactions" in data:
        trans = data["transactions"].copy()
        trans["date"] = pd.to_datetime(trans["date"])
        sales = sales.merge(trans[["date", "store_nbr", "transactions"]], on=["date", "store_nbr"], how="left")

    return sales


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = load_raw_data()
    cleaned = preprocess(data)

    cleaned.to_csv(CLEANED_TRAIN_CSV, index=False)
    print(f"\nSaved: {CLEANED_TRAIN_CSV} ({cleaned.shape})")

    # Print date range info
    print(f"Date range: {cleaned['date'].min()} to {cleaned['date'].max()}")
    print(f"Unique stores: {cleaned['store_nbr'].nunique()}")
    print(f"Unique families: {cleaned['family'].nunique()}")
    print(f"Total rows: {len(cleaned):,}")


if __name__ == "__main__":
    main()
