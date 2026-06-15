"""Streamlit dashboard for multivariate time series forecasting.

Displays model comparison, forecast visualization, and residual analysis.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import streamlit as st

from config import FORECAST_PLOT_PNG, IMAGES_DIR, MODEL_RESULTS_JSON, RESIDUAL_PLOT_PNG

st.set_page_config(
    page_title="Time Series Forecasting",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Multivariate Time Series Forecasting")
st.markdown("Model comparison dashboard for Store Sales forecasting (XGBoost · LSTM · Transformer)")

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Model Comparison", "Forecast Plot", "Residual Analysis"])


# ── Load results ───────────────────────────────────────────────────
@st.cache_data
def load_results():
    if not MODEL_RESULTS_JSON.exists():
        return None
    with open(MODEL_RESULTS_JSON) as f:
        return json.load(f)


results = load_results()


# ── Shared helpers ─────────────────────────────────────────────────
def flatten_metrics(results):
    """Extract all model metrics into a flat list."""
    if results is None:
        return []
    metrics = []
    for key in ["baseline_results", "lstm_results", "transformer_results"]:
        for entry in results.get(key, []):
            entry = dict(entry)
            entry.setdefault("model", key)
            metrics.append(entry)
    return metrics


# ── Page 1: Model Comparison ──────────────────────────────────────
if page == "Model Comparison":
    st.header("Model Performance Comparison")

    if results is None:
        st.warning("No model results found. Run the pipeline first: `make train-all`")
    else:
        metrics = flatten_metrics(results)
        if not metrics:
            st.warning("No metrics found in results file.")
        else:
            rows = []
            for m in metrics:
                rows.append(
                    {
                        "Model": m.get("model", "Unknown"),
                        "MAE": m.get("mae"),
                        "RMSE": m.get("rmse"),
                        "MAPE": m.get("mape"),
                        "sMAPE": m.get("smape"),
                    }
                )
            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.highlight_min(axis=0, subset=["MAE", "RMSE", "MAPE", "sMAPE"]),
                use_container_width=True,
            )

            # Bar chart
            st.subheader("Error Comparison (log scale)")
            if "MAE" in df.columns and "RMSE" in df.columns:
                chart_df = df.set_index("Model")[["MAE", "RMSE"]]
                st.bar_chart(chart_df)

# ── Page 2: Forecast Plot ─────────────────────────────────────────
elif page == "Forecast Plot":
    st.header("Forecast Comparison")

    if FORECAST_PLOT_PNG.exists():
        st.image(str(FORECAST_PLOT_PNG), use_container_width=True)
    else:
        st.warning("Forecast plot not generated. Run `make evaluate` to create it.")

# ── Page 3: Residual Analysis ─────────────────────────────────────
elif page == "Residual Analysis":
    st.header("Residual Diagnostics")

    if RESIDUAL_PLOT_PNG.exists():
        st.image(str(RESIDUAL_PLOT_PNG), use_container_width=True)
    else:
        st.warning("Residual plot not generated. Run `make evaluate` to create it.")

# ── Footer ────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Data: Kaggle Store Sales · Models trained locally")
