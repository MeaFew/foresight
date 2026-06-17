"""Model evaluation and comparison.

Reads model results from reports/model_results.json (written by train_baseline.py,
train_lstm.py, and train_transformer.py). Generates comparison charts and summary
metrics for the README. Computes residuals on-the-fly for diagnostic plots.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    FEATURES_TRAIN_CSV,
    FORECAST_PLOT_PNG,
    IMAGES_DIR,
    MODEL_RESULTS_JSON,
    MODELS_DIR,
    RESIDUAL_PLOT_PNG,
    VAL_DAYS,
)

# Ensure images/ exists
IMAGES_DIR.mkdir(exist_ok=True)


def load_results():
    """Load combined model results from JSON."""
    if not MODEL_RESULTS_JSON.exists():
        print(f"[WARN] {MODEL_RESULTS_JSON} not found. Run train scripts first.")
        return {}
    with open(MODEL_RESULTS_JSON) as f:
        return json.load(f)


def print_metrics_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'sMAPE':>8}")
    print("-" * 70)

    all_metrics = []
    for key in ["baseline_results", "lstm_results", "transformer_results"]:
        entries = results.get(key, [])
        for entry in entries:
            name = entry.get("model", key)
            mae = entry.get("mae")
            rmse = entry.get("rmse")
            mape = entry.get("mape_pct") if "mape_pct" in entry else entry.get("mape")
            smape = entry.get("smape")
            # Handle None/null metrics (e.g. Prophet unavailable on Windows)
            mae_str = f"{mae:>8.4f}" if mae is not None else "       --"
            rmse_str = f"{rmse:>8.4f}" if rmse is not None else "       --"
            # MAPE is already in percentage (0-100), format with f and append literal %
            mape_str = f"{mape:>7.2f}%" if mape is not None else "      --"
            smape_str = f"{smape:>7.2f}%" if smape is not None else "      --"
            print(f"{name:<20} {mae_str} {rmse_str} {mape_str} {smape_str}")
            all_metrics.append(
                {
                    "model": name,
                    "mae": mae if mae is not None else float("nan"),
                    "rmse": rmse if rmse is not None else float("nan"),
                    "mape": mape if mape is not None else float("nan"),
                    "smape": smape if smape is not None else float("nan"),
                }
            )

    print("-" * 70)
    print()
    return all_metrics


def plot_metrics_bar(results):
    """Generate a bar chart comparing MAE across models."""
    entries = []
    for key in ["baseline_results", "lstm_results", "transformer_results"]:
        for entry in results.get(key, []):
            if entry.get("mae") is None:
                continue
            name = entry.get("model", key).split("_")[0].upper()
            entries.append((name, entry))

    if not entries:
        print("[SKIP] No results to plot.")
        return

    names = [e[0] for e in entries]
    maes = [e[1].get("mae", float("nan")) for e in entries]
    rmses = [e[1].get("rmse", float("nan")) for e in entries]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, maes, width, label="MAE", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, rmses, width, label="RMSE", color="#ff7f0e")

    ax.set_ylabel("Error (log scale)")
    ax.set_title("Model Comparison -- MAE & RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(FORECAST_PLOT_PNG, dpi=150)
    plt.close()
    print(f"[OK] Saved forecast comparison: {FORECAST_PLOT_PNG}")


def compute_residuals_on_the_fly():
    """Load XGBoost model and compute residuals on validation data.

    Returns (model_name, y_true, y_pred, residuals) or None if unavailable.
    """
    import joblib
    from sklearn.metrics import mean_absolute_error

    model_path = MODELS_DIR / "xgboost_baseline.joblib"
    if not model_path.exists():
        print("[SKIP] XGBoost model not found. Cannot compute residuals.")
        return None

    feature_path = FEATURES_TRAIN_CSV
    if not feature_path.exists():
        print("[SKIP] Feature file not found. Cannot compute residuals.")
        return None

    print("Computing residuals from XGBoost model ...")
    df = pd.read_csv(feature_path, parse_dates=["date"])

    # Same split as baseline training (VAL_DAYS from config, via shared helper).
    from metrics_utils import time_train_val_split

    _train_df, val_df = time_train_val_split(df, VAL_DAYS)

    # Prepare features
    exclude = ["date", "sales", "sales_log", "id", "store_nbr", "family"]
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X_val = val_df[feature_cols].fillna(0).values
    y_true = val_df["sales_log"].values

    # Load and predict
    model = joblib.load(model_path)
    y_pred = model.predict(X_val)
    residuals = y_true - y_pred

    # Print quick summary
    mae_val = mean_absolute_error(y_true, y_pred)
    print(f"  XGBoost validation: MAE={mae_val:.4f}, N={len(residuals):,}")

    return ("XGBoost", y_true, y_pred, residuals)


def plot_residuals(results):
    """Generate residual diagnostics plots.

    Tries to use pre-computed residuals from results JSON first,
    then falls back to computing them on-the-fly from the XGBoost model.
    """
    # First, try to find pre-computed residuals in results
    best_metrics = None
    for key in ["baseline_results", "lstm_results", "transformer_results"]:
        for entry in results.get(key, []):
            if "residuals" in entry:
                residuals = np.array(entry["residuals"])
                if best_metrics is None or entry.get("mae", float("inf")) < best_metrics.get(
                    "mae", float("inf")
                ):
                    best_metrics = entry

    # If no pre-computed residuals, compute on-the-fly
    if best_metrics is None or "residuals" not in best_metrics:
        computed = compute_residuals_on_the_fly()
        if computed is None:
            print("[SKIP] No residual data available for plotting.")
            return
        model_name, y_true, y_pred, residuals = computed
        best_metrics = {
            "model": model_name,
            "residuals": residuals,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    else:
        residuals = np.array(best_metrics["residuals"])

    residuals = np.array(best_metrics["residuals"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residual histogram
    axes[0].hist(residuals, bins=50, color="#2ca02c", edgecolor="white", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Residual Distribution -- {best_metrics.get('model', 'Best')}")

    # Residual vs Predicted scatter
    if "y_true" in best_metrics and "y_pred" in best_metrics:
        y_true_arr = np.array(best_metrics["y_true"])
        y_pred_arr = np.array(best_metrics["y_pred"])
        res = y_true_arr - y_pred_arr
        axes[1].scatter(y_pred_arr, res, alpha=0.3, s=5, color="#1f77b4")
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals vs Predicted")
    else:
        axes[1].text(
            0.5, 0.5, "No prediction data", ha="center", va="center", transform=axes[1].transAxes
        )

    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT_PNG, dpi=150)
    plt.close()
    print(f"[OK] Saved residual plot: {RESIDUAL_PLOT_PNG}")


def main():
    results = load_results()
    if not results:
        print("[SKIP] No model results found. Skipping evaluation.")
        return

    print_metrics_table(results)
    plot_metrics_bar(results)
    plot_residuals(results)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
