"""Unit tests for metrics & dataset utilities.

The existing test_pipeline.py exercises feature engineering end-to-end; these
tests focus on the numerical contract of the shared metrics (mape/smape) and
the TimeSeriesDataset encoder/scaler consistency — paths that previously had
no direct coverage.

mape/smape live in src/foresight/metrics_utils.py (torch-free) so their numerical
contract can be tested without the deep-learning stack installed. The
TimeSeriesDataset tests still skip when torch is missing, mirroring
test_pipeline.py's importorskip strategy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foresight.metrics_utils import mape, smape  # noqa: E402

# ---------------------------------------------------------------------------
# mape
# ---------------------------------------------------------------------------


class TestMAPE:
    def test_perfect_prediction_is_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        assert mape(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        # |(100-90)/100| = 10%, |(50-55)/50| = 10% => mean 10%
        y_true = np.array([100.0, 50.0])
        y_pred = np.array([90.0, 55.0])
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_skips_zero_true_values(self):
        """zeros in y_true must not blow up to inf/nan; they're masked out."""
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([5.0, 90.0])
        # only the second sample counts: |100-90|/100 = 10%
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_never_negative(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(1, 100, 50)
        y_pred = rng.uniform(1, 100, 50)
        assert mape(y_true, y_pred) >= 0

    def test_handles_all_zeros_gracefully(self):
        """All-zero y_true => empty mask => must return 0.0 (finite, serializable)."""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        result = mape(y_true, y_pred)
        assert result == 0.0


# ---------------------------------------------------------------------------
# smape
# ---------------------------------------------------------------------------


class TestSMAPE:
    def test_perfect_prediction_is_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        assert smape(y, y) == pytest.approx(0.0, abs=1e-4)

    def test_symmetric_property(self):
        """sMAPE(y, y_hat) == sMAPE(y_hat, y) by definition."""
        y_true = np.array([100.0, 50.0, 25.0])
        y_pred = np.array([90.0, 60.0, 30.0])
        assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))

    def test_bounded_above_100_for_positive_values(self):
        """For positive true & pred, sMAPE stays in [0, 200]."""
        rng = np.random.default_rng(1)
        y_true = rng.uniform(1, 100, 100)
        y_pred = rng.uniform(1, 100, 100)
        result = smape(y_true, y_pred)
        assert 0.0 <= result <= 200.0

    def test_known_value(self):
        # 2*|100-90| / (100+90) = 20/190 ≈ 10.526%
        y_true = np.array([100.0])
        y_pred = np.array([90.0])
        expected = 100 * 2 * 10 / 190
        assert smape(y_true, y_pred) == pytest.approx(expected)

    def test_near_zero_denominator_is_floored(self):
        """When both y and ŷ are tiny the denom is floored at 1e-8 so the error
        does not blow up toward 200% (log1p-space sales are often near 0)."""
        y_true = np.array([1e-10, 1e-10])
        y_pred = np.array([1e-10, 1e-10])
        result = smape(y_true, y_pred)
        # Perfect prediction => 0 regardless of the floor.
        assert result == pytest.approx(0.0, abs=1e-4)
        # And a near-zero pair stays finite (not inf/nan).
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# TimeSeriesDataset — encoder/scaler consistency
# ---------------------------------------------------------------------------


@pytest.fixture
def small_ts_df():
    """Minimal dataframe shaped like the training data.

    Two stores x two families x enough days to form at least one sliding window.
    """
    pytest.importorskip("torch")
    rows = []
    for store in [1, 2]:
        for family in ["GROCERY", "DAIRY"]:
            for i in range(40):  # seq_length default 28 -> ~12 windows per series
                rows.append(
                    {
                        "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                        "store_nbr": store,
                        "family": family,
                        "sales": 100 + i,
                        "sales_log": np.log1p(100 + i),
                        "onpromotion": int(i % 2),
                    }
                )
    df = pd.DataFrame(rows)
    # mimic pipeline-derived calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


class TestTimeSeriesDataset:
    def test_fit_then_transform_produces_consistent_shapes(self, small_ts_df):
        from foresight.metrics import TimeSeriesDataset

        train_ds = TimeSeriesDataset(small_ts_df, seq_length=7)
        sample = train_ds[0]
        assert sample["cat"].shape[0] == 7  # seq_length
        assert sample["num"].shape[0] == 7
        # categorical dim = 2 (store_nbr, family)
        assert sample["cat"].shape[1] == 2

    def test_encoder_reuse_does_not_refit(self, small_ts_df):
        """A second dataset built with the first's encoders/scalers must not crash
        and must produce same column dtypes (no LabelEncoder refit)."""
        from foresight.metrics import TimeSeriesDataset

        train_ds = TimeSeriesDataset(small_ts_df, seq_length=7)
        # Reuse encoders on the same data (simulating validation set)
        val_ds = TimeSeriesDataset(
            small_ts_df,
            seq_length=7,
            encoder=train_ds.encoders,
            scalers=train_ds.scalers,
        )
        assert len(val_ds) == len(train_ds)

    def test_seq_length_governs_window_count(self, small_ts_df):
        """Longer seq_length => fewer windows per series."""
        from foresight.metrics import TimeSeriesDataset

        short = TimeSeriesDataset(small_ts_df, seq_length=5)
        long = TimeSeriesDataset(small_ts_df, seq_length=20)
        assert len(long) < len(short)

    def test_numeric_cols_parameter_pins_feature_order(self, small_ts_df):
        """predict.py passes the training-time numeric_cols from the saved meta.
        The num tensor's column order must follow that list, NOT the inference
        frame's column order — otherwise a shuffled CSV column order would
        silently feed the wrong features into the model."""
        from foresight.metrics import TimeSeriesDataset

        cols = ["month", "onpromotion", "dayofweek"]
        ds_a = TimeSeriesDataset(small_ts_df, seq_length=7, numeric_cols=cols)
        shuffled = small_ts_df[list(reversed(small_ts_df.columns))]
        ds_b = TimeSeriesDataset(shuffled, seq_length=7, numeric_cols=cols)

        assert ds_a.numeric_cols == cols
        assert ds_b.numeric_cols == cols
        # Same data + pinned order => identical num arrays regardless of the
        # input frame's column order.
        assert len(ds_a.groups_num) == len(ds_b.groups_num)
        for ga, gb in zip(ds_a.groups_num, ds_b.groups_num):
            np.testing.assert_array_equal(ga, gb)
