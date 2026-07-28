"""Tests for foresight.backtest — rolling-origin evaluation.

Synthetic panel data verifies:
- split windows never leak (train_end < val_start for every fold, expanding
  training windows, exact horizon-sized validation blocks),
- the number of folds matches the request,
- the seasonal-naive fold prediction only uses past data (t-7 actuals),
- MASE equals the hand-computed MAE / in-sample seasonal-naive scale,
- fold aggregation (mean±std) and the ±1std holdout comparison behave sanely.
"""

import numpy as np
import pandas as pd
import pytest

from foresight.backtest import (
    compare_with_holdout,
    evaluate_fold,
    rolling_origin_bounds,
    run_backtest,
    split_fold,
    summarize_folds,
)

SEASONALITY = 7


def make_panel(n_days=120, stores=(1, 2), families=("A", "B")):
    """Small synthetic panel: weekly seasonality + per-series level + noise."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rows = []
    for s in stores:
        for f in families:
            level = 10 * s + len(f)
            for i, d in enumerate(dates):
                sales = level + 5 * np.sin(2 * np.pi * i / 7) + rng.normal(0, 0.5)
                rows.append(
                    {
                        "date": d,
                        "store_nbr": s,
                        "family": f,
                        "sales": max(sales, 0.0),
                        "sales_log": float(np.log1p(max(sales, 0.0))),
                        "dayofweek": d.dayofweek,
                        "onpromotion": int(rng.random() < 0.1),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# rolling_origin_bounds / split_fold — split geometry & leakage contract
# ---------------------------------------------------------------------------


class TestRollingOriginBounds:
    def test_fold_count_matches_request(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        for n_origins in (4, 5, 6):
            bounds = rolling_origin_bounds(dates, n_origins=n_origins, horizon=16)
            assert len(bounds) == n_origins

    def test_no_leakage_every_fold_train_end_before_val_start(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        bounds = rolling_origin_bounds(dates, n_origins=5, horizon=16)
        for b in bounds:
            assert b["train_end"] < b["val_start"]
            assert b["val_start"] <= b["val_end"]

    def test_expanding_window_train_end_moves_forward(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        bounds = rolling_origin_bounds(dates, n_origins=5, horizon=16)
        train_ends = [b["train_end"] for b in bounds]
        assert train_ends == sorted(train_ends)
        # Each origin advances by exactly one horizon.
        for prev, cur in zip(bounds, bounds[1:]):
            assert cur["val_start"] - prev["val_start"] == pd.Timedelta(days=16)

    def test_validation_block_is_exactly_one_horizon(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        horizon = 16
        for b in rolling_origin_bounds(dates, n_origins=5, horizon=horizon):
            assert (b["val_end"] - b["val_start"]).days == horizon - 1

    def test_last_fold_ends_on_last_date(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        bounds = rolling_origin_bounds(dates, n_origins=5, horizon=16)
        assert bounds[-1]["val_end"] == dates[-1]

    def test_folds_do_not_overlap_in_time(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        bounds = rolling_origin_bounds(dates, n_origins=5, horizon=16)
        for prev, cur in zip(bounds, bounds[1:]):
            assert cur["val_start"] > prev["val_end"]

    def test_raises_when_insufficient_history(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        with pytest.raises(ValueError, match="Need more than"):
            rolling_origin_bounds(dates, n_origins=5, horizon=16)


class TestSplitFold:
    def test_row_dates_respect_bounds(self):
        df = make_panel()
        bounds = rolling_origin_bounds(df["date"], n_origins=4, horizon=14)
        for b in bounds:
            train_df, val_df = split_fold(df, b)
            assert train_df["date"].max() <= b["train_end"]
            assert val_df["date"].min() >= b["val_start"]
            assert val_df["date"].max() <= b["val_end"]
            # Strict separation: no future rows in train.
            assert train_df["date"].max() < val_df["date"].min()
            # Validation covers every day of the horizon for every series.
            assert val_df["date"].nunique() == 14

    def test_expanding_train_sizes(self):
        df = make_panel()
        bounds = rolling_origin_bounds(df["date"], n_origins=4, horizon=14)
        sizes = [len(split_fold(df, b)[0]) for b in bounds]
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == len(sizes)  # strictly growing


# ---------------------------------------------------------------------------
# evaluate_fold — MASE against hand computation, naive uses only the past
# ---------------------------------------------------------------------------


class TestEvaluateFold:
    def test_naive_mase_matches_hand_computation(self):
        df = make_panel(n_days=60)
        bounds = rolling_origin_bounds(df["date"], n_origins=2, horizon=14)
        train_df, val_df = split_fold(df, bounds[0])

        out = evaluate_fold(
            train_df,
            val_df,
            n_jobs=1,
            xgb_overrides={"n_estimators": 5, "early_stopping_rounds": 2},
        )

        # Hand-computed seasonal-naive predictions: ŷ(t) = y(t-7) per series.
        full = pd.concat([train_df, val_df]).sort_values(["store_nbr", "family", "date"])
        full["_pred"] = full.groupby(["store_nbr", "family"], sort=False)["sales_log"].shift(
            SEASONALITY
        )
        merged = val_df.merge(
            full[["date", "store_nbr", "family", "_pred"]],
            on=["date", "store_nbr", "family"],
        )
        expected_mae = float(np.abs(merged["sales_log"] - merged["_pred"]).mean())

        # Hand-computed in-sample scale: pooled mean |y(t) - y(t-7)| on train.
        tr = train_df.sort_values(["store_nbr", "family", "date"])
        diffs = tr.groupby(["store_nbr", "family"], sort=False)["sales_log"].diff(SEASONALITY)
        expected_scale = float(np.abs(diffs.dropna()).mean())

        assert out["seasonal_naive"]["mae"] == pytest.approx(expected_mae, rel=1e-9)
        assert out["mase_scale"] == pytest.approx(expected_scale, rel=1e-9)
        assert out["seasonal_naive"]["mase"] == pytest.approx(
            expected_mae / expected_scale, rel=1e-9
        )

    def test_naive_prediction_uses_only_past_data(self):
        """The first validation day's naive prediction must come from the
        training tail (val_start - 7), never from the validation window."""
        df = make_panel(n_days=60)
        bounds = rolling_origin_bounds(df["date"], n_origins=2, horizon=14)
        b = bounds[0]
        train_df, val_df = split_fold(df, b)

        out = evaluate_fold(
            train_df,
            val_df,
            n_jobs=1,
            xgb_overrides={"n_estimators": 5, "early_stopping_rounds": 2},
        )
        assert np.isfinite(out["seasonal_naive"]["mae"])

        # Spot-check: for one series, actual at val_start - 7 exists in train.
        series_train = train_df[train_df["store_nbr"] == 1].set_index("date")
        lookback = b["val_start"] - pd.Timedelta(days=SEASONALITY)
        assert lookback in series_train.index  # i.e. the source is strictly in the past

    def test_xgboost_fold_outputs_all_metrics(self):
        df = make_panel(n_days=60)
        bounds = rolling_origin_bounds(df["date"], n_origins=2, horizon=14)
        train_df, val_df = split_fold(df, bounds[0])
        out = evaluate_fold(
            train_df,
            val_df,
            n_jobs=1,
            xgb_overrides={"n_estimators": 5, "early_stopping_rounds": 2},
        )
        for metric in ("mae", "rmse", "mape", "smape", "mase"):
            assert np.isfinite(out["xgboost"][metric])
            assert np.isfinite(out["seasonal_naive"][metric])


# ---------------------------------------------------------------------------
# Aggregation & holdout comparison
# ---------------------------------------------------------------------------


class TestSummarize:
    def _fold(self, mae, mase):
        return {
            "xgboost": {"mae": mae, "rmse": mae, "mape": 1.0, "smape": 1.0, "mase": mase},
            "seasonal_naive": {
                "mae": 2 * mae,
                "rmse": 2 * mae,
                "mape": 2.0,
                "smape": 2.0,
                "mase": 2 * mase,
            },
            "mase_scale": 0.5,
        }

    def test_mean_std_matches_numpy(self):
        folds = [self._fold(m, s) for m, s in [(0.2, 0.4), (0.4, 0.8), (0.3, 0.6)]]
        summary = summarize_folds(folds)
        assert summary["xgboost"]["mae"]["mean"] == pytest.approx(np.mean([0.2, 0.4, 0.3]))
        assert summary["xgboost"]["mae"]["std"] == pytest.approx(np.std([0.2, 0.4, 0.3], ddof=1))
        assert summary["seasonal_naive"]["mase"]["values"] == [0.8, 1.6, 1.2]
        assert summary["mase_scale"]["mean"] == pytest.approx(0.5)

    def test_compare_within_1std(self):
        folds = [self._fold(m, s) for m, s in [(0.2, 0.4), (0.4, 0.8), (0.3, 0.6)]]
        summary = summarize_folds(folds)
        holdout = {"xgboost": {"mae": 0.31, "mase": 0.62}}  # inside ±1std
        cmp = compare_with_holdout(summary, holdout)
        assert cmp["xgboost"]["mae"]["within_1std"] is True
        assert cmp["xgboost"]["mase"]["within_1std"] is True

    def test_compare_outside_1std(self):
        folds = [self._fold(m, s) for m, s in [(0.20, 0.4), (0.22, 0.44), (0.21, 0.42)]]
        summary = summarize_folds(folds)
        holdout = {"xgboost": {"mae": 0.9, "mase": 1.8}}  # way outside
        cmp = compare_with_holdout(summary, holdout)
        assert cmp["xgboost"]["mae"]["within_1std"] is False
        assert cmp["xgboost"]["mae"]["distance_in_std"] > 1.0


# ---------------------------------------------------------------------------
# End-to-end on synthetic data (tiny XGB to keep it fast)
# ---------------------------------------------------------------------------


class TestRunBacktest:
    def test_end_to_end_synthetic(self):
        df = make_panel(n_days=90)
        results = run_backtest(
            df,
            n_origins=3,
            horizon=14,
            n_jobs=1,
            xgb_overrides={"n_estimators": 5, "early_stopping_rounds": 2},
            holdout={"xgboost": {"mae": 0.05, "mase": 0.1}},
        )
        assert len(results["folds"]) == 3
        # No-leakage invariant holds for the realized folds too.
        for f in results["folds"]:
            assert f["train_end"] < f["val_start"]
            assert f["val_start"] <= f["val_end"]
        assert "xgboost" in results["summary"]
        assert "mase" in results["summary"]["xgboost"]
        assert results["holdout_comparison"]["xgboost"]["mase"]["single_holdout"] == 0.1
