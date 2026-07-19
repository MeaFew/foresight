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
import torch
from torch.utils.data import DataLoader

from foresight.config import (
    BATCH_SIZE,
    FEATURES_TRAIN_CSV,
    LSTM_MODEL_PATH,
    MODEL_RESULTS_JSON,
    MODELS_DIR,
    REPORTS_DIR,
    TRANSFORMER_MODEL_PATH,
    VAL_DAYS,
    XGBOOST_MODEL_PATH,
)
from foresight.logging_setup import get_logger, setup_logging
from foresight.metrics import TimeSeriesDataset, mape, smape
from foresight.metrics_utils import prepare_xy
from foresight.train_lstm import LSTMForecastModule
from foresight.train_transformer import TransformerForecastModule

logger = get_logger(__name__)


def find_best_model():
    """Find the best model from model_results.json (lowest MAE)."""
    if not MODEL_RESULTS_JSON.exists():
        logger.warning(f"[WARN] {MODEL_RESULTS_JSON} not found.")
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
        logger.warning("[WARN] No valid model results found.")
        return None

    logger.info(f"Best model: {best_model} (MAE={best_mae:.4f})")
    return best_model


def load_xgboost_and_predict(df: pd.DataFrame):
    """Load XGBoost model and run inference."""
    model_path = XGBOOST_MODEL_PATH
    if not model_path.exists():
        logger.info(f"[SKIP] XGBoost model not found: {model_path}")
        return None

    model = joblib.load(model_path)
    logger.info(f"Loaded XGBoost from {model_path}")

    X, _, _ = prepare_xy(df)
    y_pred = model.predict(X)
    return y_pred


def _load_dl_and_predict(model_path: Path, model_cls, df: pd.DataFrame, min_target_date=None):
    """Shared DL inference: load checkpoint + meta, build dataset, run forward pass.

    Used by both LSTM and Transformer paths — the only differences are the
    checkpoint path and the module class, so the boilerplate (meta loading,
    dataset construction, DataLoader, eval loop) lives here once.
    """
    meta_path = model_path.with_suffix(".meta.joblib")
    if not model_path.exists():
        logger.info(f"[SKIP] Model not found: {model_path}")
        return None

    meta = joblib.load(meta_path)
    ds = TimeSeriesDataset(
        df,
        encoder=meta["encoders"],
        scalers=meta["scalers"],
        min_target_date=min_target_date,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = model_cls(
        num_stores=len(meta["encoders"]["store_nbr"].classes_),
        num_families=len(meta["encoders"]["family"].classes_),
        num_numeric=len(meta["numeric_cols"]),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            y_hat = model(batch["cat"], batch["num"])
            all_preds.extend(y_hat.cpu().numpy())

    logger.info(f"Loaded {model_cls.__name__} from {model_path}")
    return np.array(all_preds)


def load_lstm_and_predict(df: pd.DataFrame, min_target_date=None):
    """Load LSTM model and run inference."""
    return _load_dl_and_predict(LSTM_MODEL_PATH, LSTMForecastModule, df, min_target_date)


def load_transformer_and_predict(df: pd.DataFrame, min_target_date=None):
    """Load Transformer model and run inference."""
    return _load_dl_and_predict(
        TRANSFORMER_MODEL_PATH, TransformerForecastModule, df, min_target_date
    )


def predict_model(df: pd.DataFrame, model_name: str, min_target_date=None):
    """Run inference for a specific model type.

    ``min_target_date`` is forwarded to the DL loaders so predictions are
    emitted only for targets on/after that date (used for leakage-free
    validation inference).
    """
    if model_name == "xgboost":
        return load_xgboost_and_predict(df)
    elif model_name == "lstm":
        return load_lstm_and_predict(df, min_target_date=min_target_date)
    elif model_name == "transformer":
        return load_transformer_and_predict(df, min_target_date=min_target_date)
    else:
        logger.info(f"[SKIP] Unknown model: {model_name}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Load models and run inference")
    parser.add_argument("--input", type=Path, default=FEATURES_TRAIN_CSV)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[None, "xgboost", "lstm", "transformer"],
        help="Model to use (default: best from results)",
    )
    parser.add_argument(
        "--val_days",
        type=int,
        default=VAL_DAYS,
        help="Days to use as validation/holdout (default: config.VAL_DAYS)",
    )
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "predictions.csv")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["date"])

    # Split off validation period
    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(days=args.val_days - 1)
    val_df = df[df["date"] >= val_start].copy()
    logger.info(f"Validation period: {val_start.date()} ~ {max_date.date()} ({len(val_df):,} rows)")

    # Determine model to use
    model_name = args.model or find_best_model()
    if model_name is None:
        logger.info("[SKIP] No model available for prediction.")
        return

    # Run inference.
    #
    # DL models (LSTM/Transformer) need SEQ_LENGTH days of CONTEXT preceding the
    # validation period as window input. We build that context from the training
    # tail (exactly as training does) and pass ONLY that frame to
    # load_lstm/transformer_and_predict with min_target_date=val_start, so:
    #   - predictions are emitted ONLY for true-validation targets (no
    #     train-period targets leak into the inference MAE), and
    #   - the returned y_pred aligns 1:1 with val_df rows (one prediction per
    #     validation row), instead of the old path that passed the FULL frame
    #     and then sliced tails arbitrarily.
    if model_name in ("lstm", "transformer"):
        from foresight.config import SEQ_LENGTH

        context_start = val_start - pd.Timedelta(days=SEQ_LENGTH)
        dl_predict_df = df[df["date"] >= context_start].copy()
        y_pred = predict_model(dl_predict_df, model_name, min_target_date=val_start)
    else:
        y_pred = predict_model(val_df, model_name)

    if y_pred is None:
        return

    # DL predictions now align 1:1 with the true validation rows (filtered to
    # targets on/after val_start). Baseline (XGBoost/Prophet) already aligns.
    val_df["sales_log_pred"] = y_pred
    if "sales_log" in val_df.columns:
        y_true = val_df["sales_log"].values
        y_pred_arr = np.array(y_pred)
        # If lengths differ (e.g. a series too short to form an encoder window),
        # truncate to the common length rather than silently misaligning rows.
        # A mismatch on a real validation set usually signals an upstream bug,
        # so warn loudly instead of hiding it.
        if len(y_true) != len(y_pred_arr):
            logger.info(
                f"[WARN] length mismatch: y_true={len(y_true)} vs y_pred={len(y_pred_arr)}; "
                "truncating to common length. Investigate the upstream alignment."
            )
            min_len = min(len(y_true), len(y_pred_arr))
            y_true = y_true[:min_len]
            y_pred_arr = y_pred_arr[:min_len]
            val_df = val_df.iloc[:min_len].copy()
    else:
        y_true = None
        y_pred_arr = y_pred

    # Save predictions
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(args.output, index=False)
    logger.info(f"Predictions saved: {args.output}")

    # Print metrics if ground truth available
    if y_true is not None and len(y_true) > 0 and len(y_pred_arr) > 0:
        min_len = min(len(y_true), len(y_pred_arr))
        y_true = y_true[:min_len]
        y_pred_arr = y_pred_arr[:min_len]
        mae_val = float(np.mean(np.abs(y_true - y_pred_arr)))
        rmse_val = float(np.sqrt(np.mean((y_true - y_pred_arr) ** 2)))
        mape_val = float(mape(y_true, y_pred_arr))
        smape_val = float(smape(y_true, y_pred_arr))
        logger.info(f"\n{model_name.upper()} Prediction Metrics:")
        logger.info(f"  MAE  = {mae_val:.4f}")
        logger.info(f"  RMSE = {rmse_val:.4f}")
        logger.info(f"  MAPE = {mape_val:.2f}%")
        logger.info(f"  sMAPE = {smape_val:.2f}%")


if __name__ == "__main__":
    setup_logging()
    main()
