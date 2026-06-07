# Multivariate Time Series Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Lightning-2.0-purple?logo=pytorchlightning&logoColor=white" alt="PyTorch Lightning">
  <a href="https://github.com/MeaFew/multivariate-timeseries-forecasting/actions"><img src="https://github.com/MeaFew/multivariate-timeseries-forecasting/workflows/CI/badge.svg" alt="CI"></a>
</p>

## Overview

End-to-end deep learning pipeline for multivariate time series forecasting. Benchmarks classical methods (XGBoost, Prophet) against modern neural architectures (LSTM, Transformer) on the Kaggle Store Sales dataset.

## Key Highlights

- **Baseline Models**: XGBoost Regressor + Facebook Prophet for benchmarking
- **Deep Learning**: LSTM with embedding layers + Transformer with multi-head self-attention
- **Feature Engineering**: Lag features (1/7/14/28/364d), rolling statistics, cyclical seasonal encodings, promo aggregates
- **Evaluation**: MAE, RMSE, MAPE, sMAPE across all models
- **Delivery**: Streamlit dashboard comparing forecast vs. actual

## Architecture

```
Raw CSVs (train, stores, items, oil, holidays)
    |
    v
Preprocess ──> Date features, log-transform, external merges
    |
    v
Feature Eng ──> Lags, rolling mean/std, seasonal encoding, promo features
    |
    +---> XGBoost / Prophet (baselines)
    +---> LSTM + Embeddings (deep learning)
    +---> Transformer + Positional Encoding (deep learning)
    |
    v
Evaluate ──> MAE, RMSE, MAPE, sMAPE, residual analysis
    |
    v
Dashboard ──> Forecast comparison, error distribution, attention heatmap
```

## Tech Stack

| Layer | Tools | Notes |
|-------|-------|-------|
| ETL | pandas, numpy | Time-based train/val split (no random shuffle) |
| Feature Eng | pandas rolling, sklearn preprocessing | Lag/rolling features with shift(1) to prevent leakage |
| Baselines | XGBoost, Prophet | Additive regression + tree-based benchmark |
| Deep Learning | PyTorch, PyTorch Lightning | LSTM + Transformer with categorical embeddings |
| Evaluation | sklearn metrics | MAE, RMSE, MAPE, sMAPE |
| Delivery | Streamlit | Side-by-side forecast comparison |
| Quality | pytest, ruff, GitHub Actions | CI validates pipeline end-to-end |

## Quick Start

```bash
# Run full pipeline (requires Kaggle data in data/raw/)
python run_all.py

# Or step by step
make preprocess
make features
make train-baseline     # XGBoost + Prophet
make train-lstm         # LSTM model
make train-transformer  # Transformer model
make evaluate

# Launch dashboard
make dashboard

# Run tests
make verify
```

## Project Structure

```
.
├── scripts/
│   ├── generate_mock_data.py     # Synthetic retail sales data
│   ├── preprocess.py              # Date parsing, log-transform, external merges
│   ├── feature_engineering.py     # Lags, rolling stats, seasonal encoding
│   ├── train_baseline.py          # XGBoost + Prophet
│   ├── train_lstm.py              # LSTM with PyTorch Lightning
│   ├── train_transformer.py       # Transformer with positional encoding
│   └── evaluate.py                # Model comparison & residual analysis
├── dashboard/
│   └── app.py                     # Streamlit forecast comparison
├── tests/
│   └── test_pipeline.py           # Unit + integration tests
├── config.py                      # Centralized paths & hyperparameters
├── Makefile                       # Workflow orchestration
└── requirements.txt
```

## Model Comparison

### Benchmark

Based on [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) (metric: RMSLE, lower is better).

| Reference | RMSLE | Notes |
|-----------|-------|-------|
| Kaggle Starter (naive) | ~0.90–1.20 | Historical mean / naive forecast |
| Competition Median | ~0.60–0.80 | Basic lag features + XGBoost |
| Competition Top 10% | ~0.45–0.50 | Complex feature engineering |
| Competition Top 1% | ~0.35–0.40 | Fine-grained external data usage |
| **This Project (XGBoost CV)** | **~0.24** | Local 5-fold CV on log-transformed sales |

> Note: RMSLE values are not directly comparable across log-transformed vs. original scale. The Kaggle competition uses original-scale RMSLE. Local validation uses log-scale MAE/MAPE for training stability.

### Results

| Model | MAE | RMSE | MAPE | sMAPE* | Dataset |
|-------|-----|------|------|--------|---------|
| XGBoost | **0.256** | **0.380** | **11.98%** | 39.42% | Full (3M rows, 54 stores) |
| Prophet (aggregated) | — | — | — | — | *(cmdstan build failed, skipped)* |
| LSTM | **0.121** | **0.150** | **1.35%** | 1.34% | Subset (26K rows, top 20 groups) |
| Transformer | **0.170** | **0.210** | **1.91%** | 1.88% | Subset (26K rows, top 20 groups) |

> ***sMAPE is NOT comparable across rows**: XGBoost metrics are from 5-fold CV on the full dataset (54 stores × 33 product families, ~3M rows). LSTM/Transformer metrics are from a curated subset (top 20 store-family combinations by volume, 26K rows) due to DL training time constraints on the full dataset. Direct comparison of sMAPE / MAPE across different validation sets is meaningless — the subset has lower variance and thus lower percentage error. All MAE/RMSE/MAPE values are computed in log1p(sales) space.

## Data

The project uses the **Kaggle Store Sales - Time Series Forecasting** dataset:
- ~1,200 stores across Ecuador
- 33 product families
- Daily sales from 2013 to 2017
- External variables: oil prices, holidays, promotions

For local testing without Kaggle credentials, run `python scripts/generate_mock_data.py` to create a statistically similar synthetic dataset.

## License

MIT
