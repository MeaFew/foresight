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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURES_TRAIN_CSV,
    MODEL_RESULTS_JSON,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    XGBOOST_MODEL_PATH,
)
from scripts.metrics import mape, smape


def split_train_val(df: pd.DataFrame, val_days: int = 16) -> tuple:
    """Split by time — last N days as validation."""
    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(days=val_days - 1)

    train = df[df["date"] < val_start].copy()
    val = df[df["date"] >= val_start].copy()

    return train, val


def prepare_xy(df: pd.DataFrame, target_col: str = "sales_log") -> tuple:
    """Prepare feature matrix and target vector."""
    exclude = ["date", "sales", "sales_log", "id", "store_nbr", "family"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = df[feature_cols].fillna(0)
    y = df[target_col].values
    return X, y, feature_cols


def evaluate(y_true, y_pred, name: str) -> dict:
    """Compute regression metrics."""
    metrics = {
        "model": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
    }
    print(
        f"  {name:20s}  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
        f"MAPE={metrics['mape']:.2f}%  sMAPE={metrics['smape']:.2f}%"
    )
    return metrics


def train_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
    """Train XGBoost baseline."""
    print("\nTraining XGBoost ...")

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
    metrics = evaluate(y_val, y_pred, "xgboost")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, XGBOOST_MODEL_PATH)
    pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).to_csv(
        REPORTS_DIR / "xgb_feature_importance.csv", index=False
    )

    return model, metrics


def train_prophet(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
    """Train Prophet baseline on aggregated data."""
    print("\nTraining Prophet (aggregated) ...")

    # Prophet requires cmdstan build tools - unavailable on Windows.
    if sys.platform == "win32":
        print("  Prophet unavailable (sys.platform=win32) - skipping")
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

        metrics = evaluate(agg_val["y"].values, val_pred, "prophet")

        # Save
        joblib.dump(model, MODELS_DIR / "prophet_baseline.joblib")
        forecast.to_csv(REPORTS_DIR / "prophet_forecast.csv", index=False)

        return model, metrics
    except (ImportError, AttributeError, RuntimeError) as e:
        print(f"  Prophet unavailable ({type(e).__name__}) - skipping")
        return None, {"model": "prophet", "mae": None, "rmse": None, "mape": None, "smape": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FEATURES_TRAIN_CSV)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["date"])

    train, val = split_train_val(df)
    print(
        f"Train: {len(train):,} rows ({train['date'].min().date()} ~ {train['date'].max().date()})"
    )
    print(f"Val:   {len(val):,} rows ({val['date'].min().date()} ~ {val['date'].max().date()})")

    results = []

    _, xgb_metrics = train_xgboost(train, val)
    results.append(xgb_metrics)

    _, prophet_metrics = train_prophet(train, val)
    results.append(prophet_metrics)

    # Save results
    with open(MODEL_RESULTS_JSON, "w") as f:
        json.dump({"baseline_results": results}, f, indent=2)
    print(f"\nResults saved: {MODEL_RESULTS_JSON}")


if __name__ == "__main__":
    main()
