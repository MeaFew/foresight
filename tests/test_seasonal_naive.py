"""Tests for the seasonal-naive baseline and scaled-error metrics (MASE/RMSSE).

The pure-numpy helpers live in ``foresight.metrics_utils`` (torch-free) and the
panel-level evaluation lives in ``foresight.train_baseline.eval_seasonal_naive``.
These tests pin the numerical contract: MASE = holdout MAE / in-sample
seasonal-naive MAE, RMSSE likewise with squared errors, and the naive baseline
predicts exactly y(t-7) within each (store, family) series.
"""

import numpy as np
import pandas as pd
import pytest

from foresight.metrics_utils import (
    mase,
    pooled_seasonal_naive_scale,
    rmsse,
    seasonal_naive_scale,
    time_train_val_split,
)

# ---------------------------------------------------------------------------
# seasonal_naive_scale (single series, in-sample naive MAE)
# ---------------------------------------------------------------------------


class TestSeasonalNaiveScale:
    def test_known_value(self):
        # diffs: |10-1|, |20-2|, |30-3| with seasonality=1 -> mean = 9+18+27 /3 = 18
        y = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        assert seasonal_naive_scale(y, seasonality=3) == pytest.approx(18.0)

    def test_constant_series_scale_is_zero(self):
        y = np.full(20, 5.0)
        assert seasonal_naive_scale(y, seasonality=7) == pytest.approx(0.0)

    def test_too_short_series_returns_nan(self):
        y = np.arange(7, dtype=float)
        assert np.isnan(seasonal_naive_scale(y, seasonality=7))

    def test_periodic_series_scale_is_zero(self):
        """Perfectly weekly-periodic series: y(t) == y(t-7) => zero naive error."""
        week = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        y = np.tile(week, 5)
        assert seasonal_naive_scale(y, seasonality=7) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# pooled_seasonal_naive_scale (panel of series)
# ---------------------------------------------------------------------------


@pytest.fixture
def small_panel():
    """Two stores x two families, 30 days, linear ramp per series."""
    rows = []
    for store in [1, 2]:
        for family in ["A", "B"]:
            base = 100.0 if family == "A" else 10.0
            for i in range(30):
                rows.append(
                    {
                        "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                        "store_nbr": store,
                        "family": family,
                        "sales_log": base + i,  # diff(7) == 7 everywhere
                    }
                )
    return pd.DataFrame(rows)


class TestPooledSeasonalNaiveScale:
    def test_known_value(self, small_panel):
        # Every in-sample diff(7) equals exactly 7.0 -> pooled scale 7.0.
        assert pooled_seasonal_naive_scale(small_panel, seasonality=7) == pytest.approx(7.0)

    def test_does_not_cross_group_boundaries(self, small_panel):
        """If diffs crossed (store, family) boundaries the jump between series
        (base 100 vs base 10) would inflate the pooled scale far above 7."""
        scale = pooled_seasonal_naive_scale(small_panel, seasonality=7)
        assert scale == pytest.approx(7.0, abs=1e-9)

    def test_empty_diffs_returns_nan(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"] * 2),
                "store_nbr": [1, 1],
                "family": ["A", "A"],
                "sales_log": [1.0, 2.0],
            }
        )
        assert np.isnan(pooled_seasonal_naive_scale(df, seasonality=7))


# ---------------------------------------------------------------------------
# mase
# ---------------------------------------------------------------------------


class TestMASE:
    def test_perfect_prediction_is_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        assert mase(y, y, scale=2.0) == pytest.approx(0.0)

    def test_known_value(self):
        # MAE = 4.0, scale = 2.0 => MASE = 2.0 (worse than naive)
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([14.0, 16.0])
        assert mase(y_true, y_pred, scale=2.0) == pytest.approx(2.0)

    def test_naive_quality_boundary(self):
        """MASE == 1 exactly when holdout MAE equals the in-sample naive MAE."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([12.0, 18.0])  # MAE = 2.0
        assert mase(y_true, y_pred, scale=2.0) == pytest.approx(1.0)

    def test_zero_scale_returns_nan(self):
        y = np.array([1.0, 2.0])
        assert np.isnan(mase(y, y, scale=0.0))

    def test_nan_scale_returns_nan(self):
        y = np.array([1.0, 2.0])
        assert np.isnan(mase(y, y, scale=float("nan")))


# ---------------------------------------------------------------------------
# rmsse
# ---------------------------------------------------------------------------


class TestRMSSE:
    def test_perfect_prediction_is_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        assert rmsse(y, y, y_train=np.arange(20, dtype=float)) == pytest.approx(0.0)

    def test_known_value(self):
        # y_train diffs(1) all == 1 => denom = 1. RMSE of pred = 2 => RMSSE = 2.
        y_train = np.arange(10, dtype=float)
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([12.0, 18.0])  # errors 2, -2 -> RMSE 2
        assert rmsse(y_true, y_pred, y_train, seasonality=1) == pytest.approx(2.0)

    def test_short_train_returns_nan(self):
        y = np.array([1.0, 2.0])
        assert np.isnan(rmsse(y, y, y_train=np.arange(5, dtype=float), seasonality=7))

    def test_zero_denom_returns_nan(self):
        """Perfectly periodic training series => naive denominator zero => NaN."""
        y_train = np.tile([1.0, 2.0], 5)
        y = np.array([1.0, 2.0])
        assert np.isnan(rmsse(y, y, y_train, seasonality=2))


# ---------------------------------------------------------------------------
# eval_seasonal_naive (panel-level integration)
# ---------------------------------------------------------------------------


class TestEvalSeasonalNaive:
    def _make_panel(self, weekly_periodic: bool):
        rows = []
        rng = np.random.default_rng(0)
        for store in [1, 2]:
            for family in ["A", "B"]:
                for i in range(60):
                    date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
                    if weekly_periodic:
                        sales = 100.0 + 10 * (i % 7)
                    else:
                        sales = 100.0 + i + rng.normal(0, 5)
                    rows.append(
                        {
                            "date": date,
                            "store_nbr": store,
                            "family": family,
                            "sales": sales,
                            "sales_log": np.log1p(sales),
                        }
                    )
        return pd.DataFrame(rows)

    def test_prediction_is_exactly_t_minus_7(self):
        from foresight.train_baseline import eval_seasonal_naive

        df = self._make_panel(weekly_periodic=False)
        train, val = time_train_val_split(df, 16)
        metrics, scale, rmsse_denom = eval_seasonal_naive(train, val)

        # Manual t-7 reference within each series over the full frame.
        full = pd.concat([train, val]).sort_values(["store_nbr", "family", "date"])
        ref = full.groupby(["store_nbr", "family"])["sales_log"].shift(7)
        y_true = val["sales_log"].to_numpy()
        y_ref = ref.loc[val.index].to_numpy()
        manual_mae = float(np.mean(np.abs(y_true - y_ref)))

        assert metrics["mae"] == pytest.approx(manual_mae)
        assert metrics["model"] == "seasonal_naive"
        assert metrics["mase"] == pytest.approx(metrics["mae"] / scale)
        assert metrics["rmsse"] == pytest.approx(metrics["rmse"] / rmsse_denom)
        assert metrics["mase_scale"] == pytest.approx(scale)

    def test_weekly_periodic_series_scores_zero(self):
        """On a perfectly weekly-periodic series the t-7 naive forecast is exact."""
        from foresight.train_baseline import eval_seasonal_naive

        df = self._make_panel(weekly_periodic=True)
        train, val = time_train_val_split(df, 16)
        metrics, scale, _ = eval_seasonal_naive(train, val)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-12)
        # The training portion is periodic too => in-sample naive scale is 0,
        # so MASE is undefined (NaN) rather than 0/0.
        assert scale == pytest.approx(0.0)
        assert np.isnan(metrics["mase"])

    def test_metrics_are_finite_and_serializable(self):
        import json

        from foresight.train_baseline import eval_seasonal_naive

        df = self._make_panel(weekly_periodic=False)
        train, val = time_train_val_split(df, 16)
        metrics, scale, _ = eval_seasonal_naive(train, val)
        for key in ["mae", "rmse", "mape", "smape", "mase", "rmsse"]:
            assert np.isfinite(metrics[key]), f"{key} not finite"
        # Must round-trip through JSON (it is persisted to model_results.json).
        json.dumps(metrics)
        assert scale > 0
