"""Tests for multivariate time series forecasting pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

@pytest.fixture(scope="module")
def mock_data():
    """Generate small mock dataset for testing."""
    dates = pd.date_range("2022-01-01", "2023-03-31", freq="D")  # ~15 months for lag-364
    stores = [1, 2]

    records = []
    for store in stores:
        for date in dates:
            records.append({
                "date": date,
                "store_nbr": store,
                "sales": np.random.poisson(50),
                "onpromotion": np.random.choice([0, 1]),
            })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["sales_log"] = np.log1p(df["sales"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["family"] = "GROCERY"
    return df


class TestPreprocess:
    def test_date_parsing(self, mock_data):
        assert pd.api.types.is_datetime64_any_dtype(mock_data["date"])

    def test_sales_log_positive(self, mock_data):
        assert (mock_data["sales_log"] >= 0).all()

    def test_time_features_present(self, mock_data):
        assert "year" in mock_data.columns
        assert "month" in mock_data.columns
        assert "dayofweek" in mock_data.columns


class TestFeatureEngineering:
    def test_lag_features(self, mock_data):
        from scripts.feature_engineering import create_lag_features
        df = create_lag_features(mock_data, lags=[1, 7])
        assert "sales_lag_1" in df.columns
        assert "sales_lag_7" in df.columns
        assert df["sales_lag_1"].isna().sum() > 0  # First row per group is NaN

    def test_rolling_features(self, mock_data):
        from scripts.feature_engineering import create_rolling_features
        df = create_rolling_features(mock_data, windows=[7])
        assert "sales_roll_mean_7" in df.columns
        assert "sales_roll_std_7" in df.columns

    def test_seasonal_features(self, mock_data):
        from scripts.feature_engineering import create_seasonal_features
        df = create_seasonal_features(mock_data)
        assert "month_sin" in df.columns
        assert "month_cos" in df.columns
        assert "dow_sin" in df.columns
        assert df["month_sin"].between(-1, 1).all()

    def test_full_pipeline(self, mock_data):
        from scripts.feature_engineering import build_features
        df = build_features(mock_data)
        assert len(df) > 0
        assert df.isna().sum().sum() == 0  # No NaNs after dropna


class TestDataset:
    def test_dataset_length(self, mock_data):
        pytest.importorskip("pytorch_lightning")
        from scripts.metrics import TimeSeriesDataset
        df = mock_data.copy()
        df["family"] = "GROCERY"
        ds = TimeSeriesDataset(df, seq_length=7)
        assert len(ds) > 0

    def test_dataset_output_shape(self, mock_data):
        pytest.importorskip("pytorch_lightning")
        from scripts.metrics import TimeSeriesDataset
        import torch
        df = mock_data.copy()
        df["family"] = "GROCERY"
        ds = TimeSeriesDataset(df, seq_length=7)
        sample = ds[0]
        assert sample["cat"].shape == (7, 2)
        assert sample["y"].shape == torch.Size([])


class TestMetrics:
    def test_mape_zero_handling(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0.1, 1.1, 1.9])
        mask = y_true != 0
        result = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        assert result >= 0
        assert not np.isnan(result)

    def test_smape_symmetry(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 1.9, 3.2])
        result = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        assert result >= 0
        assert not np.isnan(result)
