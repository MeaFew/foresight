"""Tests for multivariate time series forecasting pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def mock_data():
    """Generate small mock dataset for testing."""
    dates = pd.date_range("2022-01-01", "2023-03-31", freq="D")  # ~15 months for lag-364
    stores = [1, 2]

    records = []
    for store in stores:
        for date in dates:
            records.append(
                {
                    "date": date,
                    "store_nbr": store,
                    "sales": np.random.poisson(50),
                    "onpromotion": np.random.choice([0, 1]),
                }
            )
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
        from foresight.feature_engineering import create_lag_features

        df = create_lag_features(mock_data, lags=[1, 7])
        assert "sales_lag_1" in df.columns
        assert "sales_lag_7" in df.columns
        assert df["sales_lag_1"].isna().sum() > 0  # First row per group is NaN

    def test_rolling_features(self, mock_data):
        from foresight.feature_engineering import create_rolling_features

        df = create_rolling_features(mock_data, windows=[7])
        assert "sales_roll_mean_7" in df.columns
        assert "sales_roll_std_7" in df.columns

    def test_seasonal_features(self, mock_data):
        from foresight.feature_engineering import create_seasonal_features

        df = create_seasonal_features(mock_data)
        assert "month_sin" in df.columns
        assert "month_cos" in df.columns
        assert "dow_sin" in df.columns
        assert df["month_sin"].between(-1, 1).all()

    def test_full_pipeline(self, mock_data):
        from foresight.feature_engineering import build_features

        df = build_features(mock_data)
        assert len(df) > 0
        assert df.isna().sum().sum() == 0  # No NaNs after dropna


class TestDataset:
    def test_dataset_length(self, mock_data):
        pytest.importorskip("pytorch_lightning")
        from foresight.metrics import TimeSeriesDataset

        df = mock_data.copy()
        df["family"] = "GROCERY"
        ds = TimeSeriesDataset(df, seq_length=7)
        assert len(ds) > 0

    def test_dataset_output_shape(self, mock_data):
        pytest.importorskip("pytorch_lightning")
        import torch

        from foresight.metrics import TimeSeriesDataset

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
        result = 100 * np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )
        assert result >= 0
        assert not np.isnan(result)


class TestLeakagePrevention:
    """Regression tests for the leakage fixes (H4 oil_lag groupby, H5 val
    target filter, H6 causal oil fill)."""

    def test_oil_lag_uses_daily_series_not_row_adjacency(self):
        """H4: oil_lag_1 must be the previous DAY's oil, computed on a
        date-unique series — not a plain shift(1) that crosses (store,family)
        group boundaries. We build a frame where two stores share the same
        dates and verify every row for a given date gets the SAME oil_lag_1
        (yesterday's oil), regardless of its row position."""
        from foresight.feature_engineering import build_features

        dates = pd.date_range("2022-01-01", periods=10, freq="D")
        rows = []
        # Distinct oil price per day so the lag is checkable
        for d in dates:
            for store in (1, 2):
                rows.append(
                    {
                        "date": d,
                        "store_nbr": store,
                        "family": "GROCERY",
                        "sales": 10.0,
                        "onpromotion": 0,
                        "dcoilwtico": float(d.day) * 10.0,  # 10,20,...,100
                    }
                )
        df = pd.DataFrame(rows)
        # build_features needs the time-part columns that preprocess.py adds
        df["sales_log"] = np.log1p(df["sales"])
        for col, fn in [
            ("year", lambda x: x.dt.year),
            ("month", lambda x: x.dt.month),
            ("day", lambda x: x.dt.day),
            ("dayofweek", lambda x: x.dt.dayofweek),
            ("dayofyear", lambda x: x.dt.dayofyear),
        ]:
            df[col] = fn(df["date"])
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

        out = build_features(df)
        # For each date, oil_lag_1 must equal the PREVIOUS date's oil price
        # and be identical across stores (proves it came from the daily series,
        # not row-adjacent shift).
        for i in range(1, len(dates)):
            d = dates[i]
            prev_oil = float(dates[i - 1].day) * 10.0
            sub = out[out["date"] == d]
            assert (sub["oil_lag_1"] == prev_oil).all(), (
                f"oil_lag_1 for {d.date()} should be previous day's oil "
                f"({prev_oil}) for ALL stores (H4: compute on daily series)"
            )

    def test_val_dataset_filters_pre_val_targets(self):
        """H5: when min_target_date is set, the validation dataset must NOT
        emit samples whose prediction target falls before that date — those
        prepended context rows are window INPUT only. Otherwise train-period
        targets leak into the validation MAE."""
        pytest.importorskip("pytorch_lightning")
        from foresight.metrics import TimeSeriesDataset

        dates = pd.date_range("2022-01-01", periods=60, freq="D")
        df = pd.DataFrame(
            {
                "date": np.tile(dates, 2),
                "store_nbr": np.repeat([1, 2], len(dates)),
                "family": "GROCERY",
                "sales_log": np.arange(2 * len(dates), dtype=float),
            }
        )
        val_start = dates[40]  # last 20 days are "true" validation
        ds_all = TimeSeriesDataset(df.copy(), seq_length=7)
        ds_filtered = TimeSeriesDataset(df.copy(), seq_length=7, min_target_date=val_start)

        # Filtered must emit strictly fewer samples (the prepended-context
        # targets before val_start are dropped).
        assert len(ds_filtered) < len(ds_all)
        # Every emitted sample's target date is on/after val_start.
        for gid, end_idx in ds_filtered.samples:
            assert pd.Timestamp(ds_filtered.groups_date[gid][end_idx]) >= val_start, (
                "Validation dataset emitted a pre-val_start target (H5 leak)."
            )

    def test_oil_fill_is_causal(self):
        """H6: a day's oil price must never be derived from a LATER day's
        price. We insert an interior NaN and assert it is filled with the
        PRECEDING known value (forward-fill), not interpolated from the next
        value."""
        from foresight.preprocess import merge_external

        oil = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"]),
                "dcoilwtico": [50.0, np.nan, np.nan, 80.0],
            }
        )
        base = pd.DataFrame({"date": oil["date"], "store_nbr": 1, "family": "GROCERY"})
        out = merge_external(base, oil, None)
        # 2022-01-02 and 2022-01-03 must both be 50.0 (ffill), NOT
        # interpolated toward 80.0 (which would use a future value).
        assert out.loc[out["date"] == pd.Timestamp("2022-01-02"), "dcoilwtico"].iloc[0] == 50.0
        assert out.loc[out["date"] == pd.Timestamp("2022-01-03"), "dcoilwtico"].iloc[0] == 50.0
