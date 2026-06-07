"""Model evaluation and comparison.

Reads model results from reports/model_results.json (written by train_baseline.py,
train_lstm.py, and train_transformer.py). Generates comparison charts and summary
metrics for the README.
"""
import json
from pathlib import Path

import sys
repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import MODEL_RESULTS_JSON, FORECAST_PLOT_PNG, RESIDUAL_PLOT_PNG, IMAGES_DIR

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
            mae = entry.get("mae", float("nan"))
            rmse = entry.get("rmse", float("nan"))
            mape = entry.get("mape", float("nan"))
            smape = entry.get("smape", float("nan"))
            print(f"{name:<20} {mae:>8.4f} {rmse:>8.4f} {mape:>7.2%} {smape:>7.2%}")
            all_metrics.append({"model": name, "mae": mae, "rmse": rmse, "mape": mape, "smape": smape})

    print("-" * 70)
    print()
    return all_metrics


def plot_metrics_bar(results):
    """Generate a bar chart comparing MAE across models."""
    entries = []
    for key in ["baseline_results", "lstm_results", "transformer_results"]:
        for entry in results.get(key, []):
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
    ax.set_title("Model Comparison — MAE & RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FORECAST_PLOT_PNG, dpi=150)
    plt.close()
    print(f"[OK] Saved forecast comparison: {FORECAST_PLOT_PNG}")


def plot_residuals(results):
    """Generate residual diagnostics plots for the best performing model."""
    if not results:
        return

    # Find the model with lowest MAE
    best_metrics = None
    for key in ["baseline_results", "lstm_results", "transformer_results"]:
        for entry in results.get(key, []):
            if "residuals" in entry:
                residuals = np.array(entry["residuals"])
                if best_metrics is None or entry.get("mae", float("inf")) < best_metrics.get("mae", float("inf")):
                    best_metrics = entry

    if best_metrics is None or "residuals" not in best_metrics:
        print("[SKIP] No residual data available for plotting.")
        return

    residuals = np.array(best_metrics["residuals"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residual histogram
    axes[0].hist(residuals, bins=50, color="#2ca02c", edgecolor="white", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Residual Distribution — {best_metrics.get('model', 'Best')}")

    # Residual vs Predicted scatter
    if "y_true" in best_metrics and "y_pred" in best_metrics:
        y_true = np.array(best_metrics["y_true"])
        y_pred = np.array(best_metrics["y_pred"])
        res = y_true - y_pred
        axes[1].scatter(y_pred, res, alpha=0.3, s=5, color="#1f77b4")
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals vs Predicted")
    else:
        axes[1].text(0.5, 0.5, "No prediction data", ha="center", va="center", transform=axes[1].transAxes)

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
