"""Baseline models for time series forecasting.

Implements:
- XGBoost Regressor (tree-based benchmark)
- Prophet (Facebook's additive regression model)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from foresight.config import (
    FEATURES_TRAIN_CSV,
    MODEL_RESULTS_JSON,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    VAL_DAYS,
    XGBOOST_MODEL_PATH,
)
from foresight.logging_setup import get_logger, setup_logging
from foresight.metrics_utils import (
    compute_metrics,
    mase,
    pooled_seasonal_naive_scale,
    prepare_xy,
    time_train_val_split,
)

logger = get_logger(__name__)

# Weekly seasonality: daily retail data repeats with period 7.
SEASONALITY = 7


def _pooled_rmsse_denom(train_df: pd.DataFrame, seasonality: int = SEASONALITY) -> float:
    """Pooled RMS of the in-sample seasonal difference (RMSSE denominator)."""
    order = ["store_nbr", "family", "date"]
    diffs = (
        train_df.sort_values(order)
        .groupby(["store_nbr", "family"], sort=False)["sales_log"]
        .diff(seasonality)
        .dropna()
    )
    if diffs.empty:
        return float("nan")
    return float(np.sqrt(np.mean(diffs.to_numpy(dtype=np.float64) ** 2)))


def eval_seasonal_naive(
    train_df: pd.DataFrame, val_df: pd.DataFrame, seasonality: int = SEASONALITY
) -> tuple:
    """Seasonal-naive baseline: ŷ(t) = y(t - 7) per (store, family) series.

    The prediction for each validation row is the actual observed sales_log
    exactly one week earlier in the SAME series. This mirrors the protocol
    used by the other models, whose lag features (shift(1)-based) also
    condition on actual recent history — the comparison stays apples-to-apples.

    Also computes MASE and RMSSE against the IN-SAMPLE seasonal-naive error
    pooled over all training series, and returns the MASE scale so callers
    can score any other model on the same denominator
    (MASE = model_MAE / scale).
    """
    logger.info("\nEvaluating seasonal-naive (t-7) baseline ...")

    full = pd.concat([train_df, val_df]).sort_values(["store_nbr", "family", "date"])
    full["_snaive_pred"] = full.groupby(["store_nbr", "family"], sort=False)["sales_log"].shift(
        seasonality
    )

    pred = full.loc[val_df.index, "_snaive_pred"]
    y_true = val_df["sales_log"].to_numpy(dtype=np.float64)
    y_pred = pred.to_numpy(dtype=np.float64)

    # Guard: the first `seasonality` val days look back into the training tail,
    # so a NaN here would mean a series shorter than the season length.
    n_nan = int(np.isnan(y_pred).sum())
    if n_nan:
        logger.warning(f"  seasonal-naive: {n_nan} rows without t-{seasonality} history dropped")
    mask = ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    scale = pooled_seasonal_naive_scale(train_df, seasonality=seasonality)
    rmsse_denom = _pooled_rmsse_denom(train_df, seasonality=seasonality)

    metrics = compute_metrics(y_true, y_pred, "seasonal_naive")
    metrics["mase"] = mase(y_true, y_pred, scale)
    metrics["rmsse"] = (
        float(metrics["rmse"] / rmsse_denom)
        if np.isfinite(rmsse_denom) and rmsse_denom > 0
        else float("nan")
    )
    # Persist the scale so evaluate.py / reports can derive MASE for models
    # whose per-row predictions are not stored (LSTM/Transformer).
    metrics["mase_scale"] = scale
    logger.info(
        f"  {'seasonal_naive':20s}  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
        f"MAPE={metrics['mape']:.2f}%  sMAPE={metrics['smape']:.2f}%  "
        f"MASE={metrics['mase']:.4f}  RMSSE={metrics['rmsse']:.4f}"
    )
    return metrics, scale, rmsse_denom


def train_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
    """Train XGBoost baseline."""
    logger.info("\nTraining XGBoost ...")

    X_train, y_train, feature_cols = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred, "xgboost")
    logger.info(
        f"  {'xgboost':20s}  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
        f"MAPE={metrics['mape']:.2f}%  sMAPE={metrics['smape']:.2f}%"
    )

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, XGBOOST_MODEL_PATH)
    pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).to_csv(
        REPORTS_DIR / "xgb_feature_importance.csv", index=False
    )

    return model, metrics


def train_prophet(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
    """Train Prophet baseline on aggregated data."""
    logger.info("\nTraining Prophet (aggregated) ...")

    # Prophet requires cmdstan build tools - unavailable on Windows.
    if sys.platform == "win32":
        logger.info("  Prophet unavailable (sys.platform=win32) - skipping")
        return None, {"model": "prophet", "mae": None, "rmse": None, "mape": None, "smape": None}

    try:
        from prophet import Prophet

        # Aggregate to total daily sales
        agg_train = train_df.groupby("date")["sales"].sum().reset_index()
        agg_train.columns = ["ds", "y"]

        agg_val = val_df.groupby("date")["sales"].sum().reset_index()
        agg_val.columns = ["ds", "y"]

        model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
        model.fit(agg_train)

        future = model.make_future_dataframe(periods=len(agg_val))
        forecast = model.predict(future)
        val_pred = forecast.iloc[-len(agg_val) :]["yhat"].values

        metrics = compute_metrics(agg_val["y"].values, val_pred, "prophet")
        logger.info(
            f"  {'prophet':20s}  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"MAPE={metrics['mape']:.2f}%  sMAPE={metrics['smape']:.2f}%"
        )

        # Save
        joblib.dump(model, MODELS_DIR / "prophet_baseline.joblib")
        forecast.to_csv(REPORTS_DIR / "prophet_forecast.csv", index=False)

        return model, metrics
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.info(f"  Prophet unavailable ({type(e).__name__}) - skipping")
        return None, {"model": "prophet", "mae": None, "rmse": None, "mape": None, "smape": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FEATURES_TRAIN_CSV)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading features from {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["date"])

    train, val = time_train_val_split(df, VAL_DAYS)
    logger.info(
        f"Train: {len(train):,} rows ({train['date'].min().date()} ~ {train['date'].max().date()})"
    )
    logger.info(
        f"Val:   {len(val):,} rows ({val['date'].min().date()} ~ {val['date'].max().date()})"
    )

    results = []

    # Seasonal-naive first: it defines the MASE/RMSSE denominators used to
    # scale every other model's error on this holdout.
    snaive_metrics, mase_scale, rmsse_denom = eval_seasonal_naive(train, val)
    results.append(snaive_metrics)

    _, xgb_metrics = train_xgboost(train, val)
    if np.isfinite(mase_scale) and mase_scale > 0:
        xgb_metrics["mase"] = float(xgb_metrics["mae"] / mase_scale)
    if np.isfinite(rmsse_denom) and rmsse_denom > 0:
        xgb_metrics["rmsse"] = float(xgb_metrics["rmse"] / rmsse_denom)
    results.append(xgb_metrics)

    _, prophet_metrics = train_prophet(train, val)
    results.append(prophet_metrics)

    # Upsert into the shared results JSON (read-modify-write, same as the DL
    # trainers in train_common.py) so existing lstm_results / transformer_results
    # are preserved instead of being silently wiped by a full-file overwrite.
    all_results = {}
    if MODEL_RESULTS_JSON.exists():
        with open(MODEL_RESULTS_JSON) as f:
            all_results = json.load(f)
    all_results["baseline_results"] = results
    with open(MODEL_RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved: {MODEL_RESULTS_JSON}")


if __name__ == "__main__":
    setup_logging()
    main()
