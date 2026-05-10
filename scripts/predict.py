"""Model loading and inference for trained forecasting models.

Loads trained models from checkpoints and runs inference on test/validation data.
Supports LSTM, Transformer, and XGBoost models.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    FEATURES_TRAIN_CSV,
    LSTM_MODEL_PATH,
    MODEL_RESULTS_JSON,
    MODELS_DIR,
    REPORTS_DIR,
    TRANSFORMER_MODEL_PATH,
    XGBOOST_MODEL_PATH,
)
from scripts.metrics import mape, smape, TimeSeriesDataset
from scripts.train_lstm import LSTMForecastModule
from scripts.train_transformer import TransformerForecastModule

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error


def find_best_model():
    """Find the best model from model_results.json (lowest MAE)."""
    if not MODEL_RESULTS_JSON.exists():
        print(f"[WARN] {MODEL_RESULTS_JSON} not found.")
        return None

    with open(MODEL_RESULTS_JSON) as f:
        results = json.load(f)

    best_model = None
    best_mae = float("inf")
    for key in ["baseline_results", "lstm_results", "transformer_results"]:
        entries = results.get(key, [])
        for entry in entries:
            mae = entry.get("mae")
            if mae is not None and mae < best_mae:
                best_mae = mae
                best_model = entry.get("model", key)

    if best_model is None:
        print("[WARN] No valid model results found.")
        return None

    print(f"Best model: {best_model} (MAE={best_mae:.4f})")
    return best_model


def load_xgboost_and_predict(df: pd.DataFrame):
    """Load XGBoost model and run inference."""
    model_path = XGBOOST_MODEL_PATH
    if not model_path.exists():
        print(f"[SKIP] XGBoost model not found: {model_path}")
        return None

    model = joblib.load(model_path)
    print(f"Loaded XGBoost from {model_path}")

    exclude = ["date", "sales", "sales_log", "id", "store_nbr", "family"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = df[feature_cols].fillna(0)
    y_pred = model.predict(X)
    return y_pred


def load_lstm_and_predict(df: pd.DataFrame):
    """Load LSTM model and run inference."""
    model_path = LSTM_MODEL_PATH
    meta_path = LSTM_MODEL_PATH.with_suffix(".meta.joblib")
    if not model_path.exists():
        print(f"[SKIP] LSTM model not found: {model_path}")
        return None

    meta = joblib.load(meta_path)
    ds = TimeSeriesDataset(
        df,
        encoder=meta["encoders"],
        scalers=meta["scalers"],
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LSTMForecastModule(
        num_stores=len(meta["encoders"]["store_nbr"].classes_),
        num_families=len(meta["encoders"]["family"].classes_),
        num_numeric=len(meta["numeric_cols"]),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            y_hat = model(batch["cat"], batch["num"])
            all_preds.extend(y_hat.cpu().numpy())

    print(f"Loaded LSTM from {model_path}")
    return np.array(all_preds)


def load_transformer_and_predict(df: pd.DataFrame):
    """Load Transformer model and run inference."""
    model_path = TRANSFORMER_MODEL_PATH
    meta_path = TRANSFORMER_MODEL_PATH.with_suffix(".meta.joblib")
    if not model_path.exists():
        print(f"[SKIP] Transformer model not found: {model_path}")
        return None

    meta = joblib.load(meta_path)
    ds = TimeSeriesDataset(
        df,
        encoder=meta["encoders"],
        scalers=meta["scalers"],
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TransformerForecastModule(
        num_stores=len(meta["encoders"]["store_nbr"].classes_),
        num_families=len(meta["encoders"]["family"].classes_),
        num_numeric=len(meta["numeric_cols"]),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            y_hat = model(batch["cat"], batch["num"])
            all_preds.extend(y_hat.cpu().numpy())

    print(f"Loaded Transformer from {model_path}")
    return np.array(all_preds)


def predict_model(df: pd.DataFrame, model_name: str):
    """Run inference for a specific model type."""
    if model_name == "xgboost":
        return load_xgboost_and_predict(df)
    elif model_name == "lstm":
        return load_lstm_and_predict(df)
    elif model_name == "transformer":
        return load_transformer_and_predict(df)
    else:
        print(f"[SKIP] Unknown model: {model_name}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Load models and run inference")
    parser.add_argument("--input", type=Path, default=FEATURES_TRAIN_CSV)
    parser.add_argument("--model", type=str, default=None,
                        choices=[None, "xgboost", "lstm", "transformer"],
                        help="Model to use (default: best from results)")
    parser.add_argument("--val_days", type=int, default=16,
                        help="Days to use as validation/holdout")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "predictions.csv")
    args = parser.parse_args()

    print(f"Loading data from {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["date"])

    # Split off validation period
    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(days=args.val_days - 1)
    val_df = df[df["date"] >= val_start].copy()
    print(f"Validation period: {val_start.date()} ~ {max_date.date()} ({len(val_df):,} rows)")

    # Determine model to use
    model_name = args.model or find_best_model()
    if model_name is None:
        print("[SKIP] No model available for prediction.")
        return

    # Run inference (DL models need full df for TimeSeriesDataset)
    predict_df = df if model_name in ("lstm", "transformer") else val_df
    y_pred = predict_model(predict_df, model_name)

    if y_pred is None:
        return

    # For DL models, predictions are for the sequence windows; align with val_df
    if model_name in ("lstm", "transformer"):
        # TimeSeriesDataset returns predictions per window; take last val_days * n_groups
        if len(y_pred) < len(val_df):
            val_df = val_df.iloc[-len(y_pred):].copy()

        val_df["sales_log_pred"] = y_pred
        if "sales_log" in val_df.columns:
            y_true = val_df["sales_log"].values[-len(y_pred):]
            y_pred_arr = np.array(y_pred[:len(y_true)])
        else:
            y_true = None
            y_pred_arr = y_pred
    else:
        val_df["sales_log_pred"] = y_pred
        if "sales_log" in val_df.columns:
            y_true = val_df["sales_log"].values
            y_pred_arr = y_pred
        else:
            y_true = None
            y_pred_arr = y_pred

    # Save predictions
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(args.output, index=False)
    print(f"Predictions saved: {args.output}")

    # Print metrics if ground truth available
    if y_true is not None and len(y_true) > 0 and len(y_pred_arr) > 0:
        min_len = min(len(y_true), len(y_pred_arr))
        y_true = y_true[:min_len]
        y_pred_arr = y_pred_arr[:min_len]
        mae = float(mean_absolute_error(y_true, y_pred_arr))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_arr)))
        mape_val = float(mape(y_true, y_pred_arr))
        smape_val = float(smape(y_true, y_pred_arr))
        print(f"\n{model_name.upper()} Prediction Metrics:")
        print(f"  MAE  = {mae:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAPE = {mape_val:.2f}%")
        print(f"  sMAPE = {smape_val:.2f}%")


if __name__ == "__main__":
    main()
