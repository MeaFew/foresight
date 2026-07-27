"""Pure-numpy metric helpers shared across forecasting models.

This module is intentionally torch-free so it can be imported in lightweight
contexts (unit tests, CI matrices without the deep-learning stack) and reused
by baseline models that don't need PyTorch.

``foresight.metrics`` re-exports these for backwards compatibility — existing
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


def seasonal_naive_scale(y, seasonality: int = 7) -> float:
    """In-sample MAE of the seasonal-naive forecast on a single series.

    This is the MASE denominator (Hyndman & Koehler 2006): the mean absolute
    error of predicting ``y[t] = y[t - seasonality]`` over the TRAINING
    (in-sample) portion of the series. Dividing a model's holdout MAE by this
    scale yields a unit-free error where 1.0 means "as good as repeating the
    last observed week".

    Returns NaN when the series has ``seasonality`` or fewer observations
    (no seasonal difference exists to average over).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size <= seasonality:
        return float("nan")
    return float(np.mean(np.abs(y[seasonality:] - y[:-seasonality])))


def pooled_seasonal_naive_scale(
    df: pd.DataFrame,
    target_col: str = "sales_log",
    group_cols: tuple = ("store_nbr", "family"),
    seasonality: int = 7,
) -> float:
    """Pooled in-sample seasonal-naive MAE across a panel of series.

    Computes the within-group seasonal difference ``y[t] - y[t-seasonality]``
    for every (store, family) series over the given frame (typically the
    training split) and returns the pooled mean absolute difference. This is
    the panel-data MASE denominator: it lets a single aggregate MASE be
    derived for ANY model from its aggregate holdout MAE alone
    (``MASE = MAE_model / pooled_scale``), so deep-learning models whose
    per-row predictions are not persisted can still be compared on the same
    scale as the tabular baselines.
    """
    order = [c for c in (*group_cols, "date") if c in df.columns]
    sorted_df = df.sort_values(order)
    diffs = sorted_df.groupby(list(group_cols), sort=False)[target_col].diff(seasonality).dropna()
    if diffs.empty:
        return float("nan")
    return float(np.abs(diffs).mean())


def mase(y_true, y_pred, scale: float) -> float:
    """Mean Absolute Scaled Error: holdout MAE divided by *scale*.

    *scale* is the in-sample seasonal-naive MAE (see ``seasonal_naive_scale``
    for a single series or ``pooled_seasonal_naive_scale`` for a panel).
    MASE < 1.0 means the model beats the seasonal-naive benchmark; > 1.0
    means the naive benchmark is better. Returns NaN when the scale is not
    a positive finite number (division by zero would be meaningless).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def rmsse(y_true, y_pred, y_train, seasonality: int = 7) -> float:
    """Root Mean Squared Scaled Error (M5 competition metric).

    RMSE of the model divided by the RMSE of the in-sample seasonal-naive
    forecast on ``y_train``. Like MASE, 1.0 is the naive-benchmark level, but
    squaring penalizes large errors more strongly. Returns NaN when the
    training series is too short or the naive denominator is zero (a
    perfectly periodic training series).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.size <= seasonality:
        return float("nan")
    denom = float(np.mean((y_train[seasonality:] - y_train[:-seasonality]) ** 2))
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return float(rmse / np.sqrt(denom))


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


# Columns excluded from the feature matrix (identifiers / targets / raw values).
FEATURE_EXCLUDE_COLS = ["date", "sales", "sales_log", "id", "store_nbr", "family"]


def prepare_xy(df: pd.DataFrame, target_col: str = "sales_log") -> tuple:
    """Prepare feature matrix and target vector for tabular models.

    Shared by train_baseline.py, evaluate.py, and predict.py so the column
    exclusion list and fillna strategy stay consistent across all call sites.

    Returns ``(X, y, feature_cols)`` where *X* is a DataFrame (NaN-filled),
    *y* is a 1-D numpy array, and *feature_cols* is the list of used columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in FEATURE_EXCLUDE_COLS]
    X = df[feature_cols].fillna(0)
    y = df[target_col].values
    return X, y, feature_cols


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
