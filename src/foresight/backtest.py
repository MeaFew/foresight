"""Rolling-origin (walk-forward) backtest for the XGBoost and seasonal-naive baselines.

Why not random k-fold cross-validation for time series? Random CV shuffles
observations, so training rows end up *after* validation rows in time — the
model gets to "see the future" (and autocorrelated neighbours leak across the
fold boundary), producing optimistically biased scores. Rolling-origin
evaluation respects the arrow of time: every fold trains only on the past and
validates on the immediately following horizon.

Design: EXPANDING window, ``n_origins`` folds, fixed validation horizon
(default = VAL_DAYS, the same 16-day window as the single holdout):

    date ────────────────────────────────────────────────────────►
    fold 1: [════════ train ════════][val]
    fold 2: [════════ train ════════════][val]
    fold 3: [════════ train ═════════════════][val]
    fold 4: [════════ train ═════════════════════][val]
    fold 5: [════════ train ═════════════════════════][val]
                                       ▲ each [val] is `horizon` days,
                                       ▲ train never touches it or anything after

Each fold *retrains* XGBoost from scratch on the expanded training window
(feature engineering is reused as-is — the pipeline's lag/rolling features are
causal by construction, see tests/test_pipeline.py::TestLeakagePrevention) and
re-scores the seasonal-naive (t-7) baseline. The MASE denominator (pooled
in-sample seasonal-naive MAE) is recomputed per fold on that fold's training
window, so it grows more stable as the window expands.

Outputs per-fold MAE/RMSE/MAPE/sMAPE/MASE plus mean±std across folds, and a
side-by-side comparison against the single-holdout numbers in
``reports/model_results.json`` (checking whether the single score falls inside
the rolling mean ± 1 std band). Persisted to ``reports/backtest_results.json``.

Usage:
    python -m foresight.backtest [--origins 5] [--horizon 16] [--n-jobs 8]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from foresight.config import (
    FEATURES_TRAIN_CSV,
    MODEL_RESULTS_JSON,
    RANDOM_STATE,
    REPORTS_DIR,
    VAL_DAYS,
)
from foresight.logging_setup import get_logger, setup_logging
from foresight.metrics_utils import compute_metrics, mase, pooled_seasonal_naive_scale, prepare_xy
from foresight.train_baseline import SEASONALITY, eval_seasonal_naive

logger = get_logger(__name__)

BACKTEST_RESULTS_JSON = REPORTS_DIR / "backtest_results.json"

METRIC_NAMES = ("mae", "rmse", "mape", "smape", "mase")

# Same hyperparameters as train_baseline.train_xgboost, except n_jobs: the
# single-holdout run uses n_jobs=-1 (all cores), but sustained all-core load
# can trip this machine, so the backtest caps threads (default 8, see --n-jobs).
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 30,
}


def rolling_origin_bounds(dates: pd.DatetimeIndex, n_origins: int, horizon: int) -> list[dict]:
    """Compute (train_end, val_start, val_end) date bounds for each expanding fold.

    ``dates`` is the sorted unique index of observation dates (daily
    frequency assumed — the horizon is expressed in days/steps). The most
    recent ``n_origins * horizon`` dates form the evaluation region; fold *k*
    validates on the *k*-th horizon-sized block of that region (oldest first)
    and trains on everything strictly before it.

    Returns a list of dicts with keys ``fold``, ``train_end``, ``val_start``,
    ``val_end`` (all pd.Timestamp; ``val_end`` inclusive). Raises ValueError
    when there are not enough dates for the requested folds.
    """
    dates = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(dates))).sort_values()
    n = len(dates)
    needed = n_origins * horizon
    if n <= needed:
        raise ValueError(
            f"Need more than {needed} distinct dates for {n_origins} origins "
            f"x {horizon}-day horizon, got {n}."
        )
    bounds = []
    for k in range(n_origins):
        val_start_idx = n - (n_origins - k) * horizon
        val_end_idx = val_start_idx + horizon - 1
        bounds.append(
            {
                "fold": k,
                "train_end": dates[val_start_idx - 1],
                "val_start": dates[val_start_idx],
                "val_end": dates[val_end_idx],
            }
        )
    return bounds


def split_fold(df: pd.DataFrame, fold_bounds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize (train, val) frames for one fold. Strictly no future overlap:
    every training row is dated on or before ``train_end`` < ``val_start``."""
    train_df = df[df["date"] <= fold_bounds["train_end"]].copy()
    val_df = df[
        (df["date"] >= fold_bounds["val_start"]) & (df["date"] <= fold_bounds["val_end"])
    ].copy()
    return train_df, val_df


def evaluate_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_jobs: int = 8,
    xgb_overrides: dict | None = None,
) -> dict:
    """Retrain XGBoost and re-score seasonal-naive on one fold.

    Returns a dict with per-model metric dicts (MAE/RMSE/MAPE/sMAPE/MASE) and
    the fold's MASE scale (pooled in-sample seasonal-naive MAE on THIS fold's
    training window).
    """
    # Seasonal-naive first: defines this fold's MASE denominator.
    naive_metrics, scale, _ = eval_seasonal_naive(train_df, val_df)

    X_train, y_train, _ = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df)

    params = {**XGB_PARAMS, **(xgb_overrides or {}), "n_jobs": n_jobs}
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    xgb_metrics = compute_metrics(y_val, y_pred, "xgboost")
    xgb_metrics["mase"] = mase(y_val, y_pred, scale)
    xgb_metrics["best_iteration"] = int(getattr(model, "best_iteration", params["n_estimators"]))

    return {
        "xgboost": xgb_metrics,
        "seasonal_naive": {k: v for k, v in naive_metrics.items() if k != "mase_scale"},
        "mase_scale": scale,
    }


def summarize_folds(folds: list[dict]) -> dict:
    """Aggregate per-fold metrics into mean±std for each model."""
    summary = {}
    for model in ("xgboost", "seasonal_naive"):
        model_summary = {}
        for metric in METRIC_NAMES:
            values = [
                f[model][metric]
                for f in folds
                if f.get(model) and np.isfinite(f[model].get(metric, float("nan")))
            ]
            if values:
                model_summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": [float(v) for v in values],
                }
        summary[model] = model_summary
    scales = [f["mase_scale"] for f in folds if np.isfinite(f.get("mase_scale", float("nan")))]
    if scales:
        summary["mase_scale"] = {
            "mean": float(np.mean(scales)),
            "std": float(np.std(scales, ddof=1)) if len(scales) > 1 else 0.0,
            "values": [float(v) for v in scales],
        }
    return summary


def compare_with_holdout(summary: dict, holdout: dict) -> dict:
    """Check whether each single-holdout metric falls inside rolling mean ± 1 std.

    ``holdout`` maps model name -> single-split metric dict (from
    reports/model_results.json). Returns per-model, per-metric comparison
    records with the signed z-like distance in std units.
    """
    comparison = {}
    for model, model_summary in summary.items():
        if model not in holdout or not isinstance(model_summary, dict):
            continue
        per_metric = {}
        for metric, stats in model_summary.items():
            single = holdout[model].get(metric)
            if single is None or not np.isfinite(single):
                continue
            std = stats["std"]
            distance = (single - stats["mean"]) / std if std > 0 else 0.0
            per_metric[metric] = {
                "single_holdout": float(single),
                "rolling_mean": stats["mean"],
                "rolling_std": std,
                "within_1std": bool(abs(distance) <= 1.0),
                "distance_in_std": float(distance),
            }
        if per_metric:
            comparison[model] = per_metric
    return comparison


def load_single_holdout(results_json: Path = MODEL_RESULTS_JSON) -> dict:
    """Pull the single-holdout baseline numbers from reports/model_results.json."""
    if not results_json.exists():
        logger.warning(f"{results_json} not found — skipping holdout comparison")
        return {}
    with open(results_json) as f:
        results = json.load(f)
    holdout = {}
    for record in results.get("baseline_results", []):
        name = record.get("model")
        if name in ("xgboost", "seasonal_naive"):
            holdout[name] = record
    return holdout


def run_backtest(
    df: pd.DataFrame,
    n_origins: int = 5,
    horizon: int = VAL_DAYS,
    n_jobs: int = 8,
    xgb_overrides: dict | None = None,
    holdout: dict | None = None,
) -> dict:
    """Run the full rolling-origin backtest and return the results dict."""
    bounds = rolling_origin_bounds(df["date"], n_origins, horizon)
    folds = []
    for b in bounds:
        train_df, val_df = split_fold(df, b)
        logger.info(
            f"\nFold {b['fold'] + 1}/{n_origins}: train ≤ {b['train_end'].date()} "
            f"({len(train_df):,} rows) | val {b['val_start'].date()} ~ {b['val_end'].date()} "
            f"({len(val_df):,} rows)"
        )
        fold_metrics = evaluate_fold(train_df, val_df, n_jobs=n_jobs, xgb_overrides=xgb_overrides)
        logger.info(
            f"  xgboost        MAE={fold_metrics['xgboost']['mae']:.4f}  "
            f"RMSE={fold_metrics['xgboost']['rmse']:.4f}  "
            f"MASE={fold_metrics['xgboost']['mase']:.4f}"
        )
        logger.info(
            f"  seasonal_naive MAE={fold_metrics['seasonal_naive']['mae']:.4f}  "
            f"RMSE={fold_metrics['seasonal_naive']['rmse']:.4f}  "
            f"MASE={fold_metrics['seasonal_naive']['mase']:.4f}"
        )
        folds.append(
            {
                "fold": b["fold"],
                "train_end": str(b["train_end"].date()),
                "val_start": str(b["val_start"].date()),
                "val_end": str(b["val_end"].date()),
                "n_train_rows": int(len(train_df)),
                "n_val_rows": int(len(val_df)),
                **fold_metrics,
            }
        )

    summary = summarize_folds(folds)
    comparison = compare_with_holdout(summary, holdout or {})
    return {
        "config": {
            "n_origins": n_origins,
            "horizon_days": horizon,
            "seasonality": SEASONALITY,
            "xgb_params": {k: v for k, v in {**XGB_PARAMS, **(xgb_overrides or {})}.items()},
            "n_jobs": n_jobs,
        },
        "folds": folds,
        "summary": summary,
        "holdout_comparison": comparison,
    }


def sample_series(df: pd.DataFrame, max_series: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Subsample whole (store_nbr, family) series to bound runtime/memory."""
    series = df[["store_nbr", "family"]].drop_duplicates()
    if len(series) <= max_series:
        return df
    keep = series.sample(n=max_series, random_state=seed)
    sampled = df.merge(keep, on=["store_nbr", "family"], how="inner")
    logger.info(f"Sampled {max_series}/{len(series)} series ({len(sampled):,} rows)")
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Rolling-origin backtest for baseline models")
    parser.add_argument("--input", type=Path, default=FEATURES_TRAIN_CSV)
    parser.add_argument("--output", type=Path, default=BACKTEST_RESULTS_JSON)
    parser.add_argument("--origins", type=int, default=5, help="number of rolling origins (4-6)")
    parser.add_argument("--horizon", type=int, default=VAL_DAYS, help="validation days per fold")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="XGBoost threads (capped at 8: sustained all-core load trips this machine)",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="optionally subsample N (store, family) series to bound runtime",
    )
    args = parser.parse_args()

    n_jobs = min(args.n_jobs, 8)
    logger.info(f"Loading features from {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["date"])
    if args.max_series:
        df = sample_series(df, args.max_series)

    holdout = load_single_holdout()
    results = run_backtest(
        df, n_origins=args.origins, horizon=args.horizon, n_jobs=n_jobs, holdout=holdout
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nBacktest results saved: {args.output}")

    for model, model_summary in results["summary"].items():
        if not isinstance(model_summary, dict) or "mase" not in model_summary:
            continue
        m = model_summary["mase"]
        logger.info(f"  {model:16s} MASE {m['mean']:.4f} ± {m['std']:.4f} across folds")


if __name__ == "__main__":
    setup_logging()
    main()
