"""Feature engineering for time series forecasting.

Creates:
- Lag features (1d, 7d, 14d, 28d, 364d)
- Rolling statistics (7d/14d/28d/60d mean, std, min, max)
- Expanding mean
- Seasonal encodings (month, week, dayofweek)
- Promo-related features
- Oil price features
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLEANED_TRAIN_CSV, FEATURES_TRAIN_CSV, PROCESSED_DATA_DIR


def create_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Create lag features for sales."""
    df = df.copy()
    for lag in lags:
        df[f"sales_lag_{lag}"] = df.groupby(["store_nbr", "item_nbr"])["sales_log"].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Create rolling mean and std features."""
    df = df.copy()
    for window in windows:
        df[f"sales_roll_mean_{window}"] = (
            df.groupby(["store_nbr", "item_nbr"])["sales_log"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"sales_roll_std_{window}"] = (
            df.groupby(["store_nbr", "item_nbr"])["sales_log"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )
    return df


def create_expanding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create expanding mean feature."""
    df = df.copy()
    df["sales_expanding_mean"] = (
        df.groupby(["store_nbr", "item_nbr"])["sales_log"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    return df


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical seasonal encodings."""
    df = df.copy()
    # Month cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Day of week cyclical encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    # Day of year cyclical encoding
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    return df


def create_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create promotion-related features."""
    df = df.copy()
    # Cumulative promo days in past 7/14/30 days
    for window in [7, 14, 30]:
        df[f"promo_roll_sum_{window}"] = (
            df.groupby(["store_nbr", "item_nbr"])["onpromotion"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
        )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    print("Building time series features ...")

    df = df.copy()
    df = df.sort_values(["store_nbr", "item_nbr", "date"])

    print("  Lag features ...")
    df = create_lag_features(df, lags=[1, 7, 14, 28, 364])

    print("  Rolling features ...")
    df = create_rolling_features(df, windows=[7, 14, 28, 60])

    print("  Expanding features ...")
    df = create_expanding_features(df)

    print("  Seasonal features ...")
    df = create_seasonal_features(df)

    print("  Promo features ...")
    df = create_promo_features(df)

    # Oil price lag
    if "dcoilwtico" in df.columns:
        df["oil_lag_1"] = df["dcoilwtico"].shift(1)

    # Drop rows with too many NaNs (mainly from lags)
    print("  Dropping rows with missing features ...")
    before = len(df)
    df = df.dropna(subset=[c for c in df.columns if "lag" in c or "roll" in c])
    print(f"  Kept {len(df):,} / {before:,} rows")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEANED_TRAIN_CSV, parse_dates=["date"])
    featured = build_features(df)

    featured.to_csv(FEATURES_TRAIN_CSV, index=False)
    print(f"\nSaved: {FEATURES_TRAIN_CSV} ({featured.shape})")


if __name__ == "__main__":
    main()
