"""Pure-numpy metric helpers shared across forecasting models.

This module is intentionally torch-free so it can be imported in lightweight
contexts (unit tests, CI matrices without the deep-learning stack) and reused
by baseline models that don't need PyTorch.

``scripts.metrics`` re-exports these for backwards compatibility — existing
``from foresight.metrics import mape, smape`` statements keep working.
"""

import numpy as np
import pandas as pd


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error.

    The denominator ``|y| + |ŷ|`` can approach zero when both are tiny (common
    in log1p-space sales near zero), which inflates the error toward 200% under
    the raw formula. We clip the denominator to a small positive floor rather
    than just adding an epsilon to the exact-zero case, so near-zero pairs do
    not dominate the average.
    """
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.clip(denom, 1e-8, None)
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / denom)


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (skips zeros in true values).

    Returns 0.0 when no true values are nonzero (rather than NaN), so the value
    serializes cleanly to JSON and never silently propagates.
    """
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def time_train_val_split(df: pd.DataFrame, val_days: int):
    """Split a time-sorted frame into train / validation by trailing days.

    Centralized here so the baseline trainer, the DL trainers, and evaluate.py
    all agree on the split. Previously each call site re-implemented this and
    one of them used an off-by-one literal (15 vs 16 days).
    """
    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(days=val_days - 1)
    val_df = df[df["date"] >= val_start].copy()
    train_df = df[df["date"] < val_start].copy()
    return train_df, val_df


def compute_metrics(y_true, y_pred, name: str) -> dict:
    """Build the standard {mae, rmse, mape, smape, model} metrics dict.

    Used by the baseline and DL trainers (and predict.py) so the metric set and
    rounding stay consistent across models.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "mape": float(mape(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
    }
