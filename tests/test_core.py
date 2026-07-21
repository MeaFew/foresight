"""Tests for core utility modules: config, logging_setup, metrics_utils, evaluate.

Covers paths that previously had no direct test coverage:
- config.py: path constants resolve correctly
- logging_setup.py: setup_logging / get_logger contract
- metrics_utils.py: prepare_xy, time_train_val_split, compute_metrics
- evaluate.py: load_results, print_metrics_table
- preprocess.py: preprocess_sales, merge_external
- predict.py: find_best_model
- generate_mock_data.py: synthetic data generators
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_base_dir_is_project_root(self):
        from foresight.config import BASE_DIR

        # BASE_DIR should contain pyproject.toml (project root marker)
        assert (BASE_DIR / "pyproject.toml").exists()

    def test_data_dirs_are_under_base(self):
        from foresight.config import BASE_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR

        assert DATA_DIR.parent == BASE_DIR
        assert MODELS_DIR.parent == BASE_DIR
        assert REPORTS_DIR.parent == BASE_DIR

    def test_model_paths_are_under_models_dir(self):
        from foresight.config import (
            LSTM_MODEL_PATH,
            MODELS_DIR,
            TRANSFORMER_MODEL_PATH,
            XGBOOST_MODEL_PATH,
        )

        assert LSTM_MODEL_PATH.parent == MODELS_DIR
        assert TRANSFORMER_MODEL_PATH.parent == MODELS_DIR
        assert XGBOOST_MODEL_PATH.parent == MODELS_DIR

    def test_val_days_is_positive(self):
        from foresight.config import VAL_DAYS

        assert VAL_DAYS > 0

    def test_seq_length_is_positive(self):
        from foresight.config import SEQ_LENGTH

        assert SEQ_LENGTH > 0


# ---------------------------------------------------------------------------
# logging_setup
# ---------------------------------------------------------------------------


class TestLoggingSetup:
    def test_get_logger_returns_logger(self):
        import logging

        from foresight.logging_setup import get_logger

        lg = get_logger("test.module")
        assert isinstance(lg, logging.Logger)
        assert lg.name == "test.module"

    def test_setup_logging_is_idempotent(self):
        from foresight.logging_setup import setup_logging

        # Calling twice should not raise or add duplicate handlers
        setup_logging()
        setup_logging()


# ---------------------------------------------------------------------------
# metrics_utils — prepare_xy
# ---------------------------------------------------------------------------


class TestPrepareXY:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "store_nbr": [1, 1, 2, 2, 3],
                "family": ["A", "B", "A", "B", "A"],
                "sales": [10.0, 20.0, 30.0, 40.0, 50.0],
                "sales_log": np.log1p([10.0, 20.0, 30.0, 40.0, 50.0]),
                "onpromotion": [0, 1, 0, 1, 0],
                "extra_feat": [1.0, np.nan, 3.0, 4.0, 5.0],
            }
        )

    def test_excludes_non_feature_columns(self, sample_df):
        from foresight.metrics_utils import prepare_xy

        X, y, feature_cols = prepare_xy(sample_df)
        for col in ["date", "sales", "sales_log", "id", "store_nbr", "family"]:
            assert col not in feature_cols

    def test_includes_numeric_features(self, sample_df):
        from foresight.metrics_utils import prepare_xy

        X, y, feature_cols = prepare_xy(sample_df)
        assert "onpromotion" in feature_cols
        assert "extra_feat" in feature_cols

    def test_fills_nan_with_zero(self, sample_df):
        from foresight.metrics_utils import prepare_xy

        X, y, feature_cols = prepare_xy(sample_df)
        assert X.isna().sum().sum() == 0

    def test_target_is_numpy_array(self, sample_df):
        from foresight.metrics_utils import prepare_xy

        X, y, feature_cols = prepare_xy(sample_df)
        assert isinstance(y, np.ndarray)
        assert len(y) == len(sample_df)

    def test_custom_target_col(self, sample_df):
        from foresight.metrics_utils import prepare_xy

        X, y, feature_cols = prepare_xy(sample_df, target_col="sales")
        np.testing.assert_array_equal(y, sample_df["sales"].values)


# ---------------------------------------------------------------------------
# metrics_utils — time_train_val_split
# ---------------------------------------------------------------------------


class TestTimeTrainValSplit:
    @pytest.fixture
    def time_df(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame({"date": dates, "value": range(30)})

    def test_split_covers_all_rows(self, time_df):
        from foresight.metrics_utils import time_train_val_split

        train, val = time_train_val_split(time_df, val_days=7)
        assert len(train) + len(val) == len(time_df)

    def test_val_has_correct_day_count(self, time_df):
        from foresight.metrics_utils import time_train_val_split

        train, val = time_train_val_split(time_df, val_days=7)
        assert val["date"].nunique() == 7

    def test_train_dates_before_val_dates(self, time_df):
        from foresight.metrics_utils import time_train_val_split

        train, val = time_train_val_split(time_df, val_days=7)
        assert train["date"].max() < val["date"].min()

    def test_no_overlap(self, time_df):
        from foresight.metrics_utils import time_train_val_split

        train, val = time_train_val_split(time_df, val_days=7)
        overlap = set(train["date"]) & set(val["date"])
        assert len(overlap) == 0


# ---------------------------------------------------------------------------
# metrics_utils — compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_prediction(self):
        from foresight.metrics_utils import compute_metrics

        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y, "test")
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["model"] == "test"

    def test_known_mae(self):
        from foresight.metrics_utils import compute_metrics

        y_true = np.array([10.0, 20.0])
        y_pred = np.array([12.0, 18.0])
        m = compute_metrics(y_true, y_pred, "test")
        assert m["mae"] == pytest.approx(2.0)

    def test_returns_all_keys(self):
        from foresight.metrics_utils import compute_metrics

        m = compute_metrics(np.array([1.0]), np.array([1.5]), "m")
        assert set(m.keys()) == {"model", "mae", "rmse", "mape", "smape"}


# ---------------------------------------------------------------------------
# evaluate — load_results / print_metrics_table
# ---------------------------------------------------------------------------


class TestEvaluateHelpers:
    """evaluate.py imports matplotlib at module level, which is not part of the
    lightweight CI test deps — skip these tests when it is missing."""

    def test_load_results_missing_file(self, tmp_path, monkeypatch):
        """load_results returns {} when the JSON file does not exist."""
        pytest.importorskip("matplotlib")
        import foresight.evaluate as ev

        monkeypatch.setattr(ev, "MODEL_RESULTS_JSON", tmp_path / "nonexistent.json")
        result = ev.load_results()
        assert result == {}

    def test_load_results_valid_json(self, tmp_path, monkeypatch):
        """load_results reads a valid JSON file."""
        pytest.importorskip("matplotlib")
        import foresight.evaluate as ev

        results_file = tmp_path / "model_results.json"
        data = {"baseline_results": [{"model": "xgboost", "mae": 0.5}]}
        results_file.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(ev, "MODEL_RESULTS_JSON", results_file)
        result = ev.load_results()
        assert "baseline_results" in result

    def test_print_metrics_table_handles_empty(self):
        """print_metrics_table should not crash on empty results."""
        pytest.importorskip("matplotlib")
        from foresight.evaluate import print_metrics_table

        all_metrics = print_metrics_table({})
        assert all_metrics == []

    def test_print_metrics_table_handles_none_mae(self):
        """print_metrics_table gracefully handles None metrics (e.g. Prophet on Windows)."""
        pytest.importorskip("matplotlib")
        from foresight.evaluate import print_metrics_table

        results = {
            "baseline_results": [
                {"model": "prophet", "mae": None, "rmse": None, "mape": None, "smape": None}
            ]
        }
        all_metrics = print_metrics_table(results)
        assert len(all_metrics) == 1
        assert np.isnan(all_metrics[0]["mae"])


# ---------------------------------------------------------------------------
# preprocess — preprocess_sales / merge_external
# ---------------------------------------------------------------------------


class TestPreprocessSales:
    @pytest.fixture
    def raw_sales(self):
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "store_nbr": [1] * 10,
                "family": ["GROCERY"] * 10,
                "sales": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                "onpromotion": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

    def test_adds_sales_log(self, raw_sales):
        from foresight.preprocess import preprocess_sales

        out = preprocess_sales(raw_sales)
        assert "sales_log" in out.columns
        assert (out["sales_log"] >= 0).all()

    def test_adds_time_features(self, raw_sales):
        from foresight.preprocess import preprocess_sales

        out = preprocess_sales(raw_sales)
        for col in ["year", "month", "day", "dayofweek", "dayofyear", "weekofyear"]:
            assert col in out.columns

    def test_does_not_mutate_input(self, raw_sales):
        from foresight.preprocess import preprocess_sales

        original_cols = set(raw_sales.columns)
        preprocess_sales(raw_sales)
        assert set(raw_sales.columns) == original_cols


class TestMergeExternal:
    def test_holiday_flag_set(self):
        from foresight.preprocess import merge_external

        base = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "store_nbr": [1, 1, 1],
            }
        )
        holidays = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "type": ["National"],
                "locale": ["National"],
                "transferred": [False],
            }
        )
        out = merge_external(base, None, holidays)
        assert out.loc[out["date"] == pd.Timestamp("2024-01-01"), "is_holiday"].iloc[0] == 1
        assert out.loc[out["date"] == pd.Timestamp("2024-01-02"), "is_holiday"].iloc[0] == 0

    def test_no_holidays_gives_zero_flag(self):
        from foresight.preprocess import merge_external

        base = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "store_nbr": [1]})
        out = merge_external(base, None, None)
        assert (out["is_holiday"] == 0).all()

    def test_oil_merge_adds_column(self):
        from foresight.preprocess import merge_external

        base = pd.DataFrame(
            {"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "store_nbr": [1, 1]}
        )
        oil = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "dcoilwtico": [70.0, 72.0],
            }
        )
        out = merge_external(base, oil, None)
        assert "dcoilwtico" in out.columns
        assert out["dcoilwtico"].isna().sum() == 0


# ---------------------------------------------------------------------------
# predict — find_best_model
# ---------------------------------------------------------------------------


class TestFindBestModel:
    """predict.py imports torch at module level, which is deliberately absent
    from the lightweight CI test deps — skip these tests when it is missing."""

    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        import foresight.predict as pred

        monkeypatch.setattr(pred, "MODEL_RESULTS_JSON", tmp_path / "missing.json")
        assert pred.find_best_model() is None

    def test_returns_lowest_mae_model(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        import foresight.predict as pred

        results_file = tmp_path / "model_results.json"
        data = {
            "baseline_results": [{"model": "xgboost", "mae": 0.5}],
            "lstm_results": [{"model": "lstm", "mae": 0.3}],
            "transformer_results": [{"model": "transformer", "mae": 0.4}],
        }
        results_file.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(pred, "MODEL_RESULTS_JSON", results_file)
        assert pred.find_best_model() == "lstm"

    def test_skips_none_mae(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        import foresight.predict as pred

        results_file = tmp_path / "model_results.json"
        data = {
            "baseline_results": [
                {"model": "prophet", "mae": None},
                {"model": "xgboost", "mae": 0.5},
            ]
        }
        results_file.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(pred, "MODEL_RESULTS_JSON", results_file)
        assert pred.find_best_model() == "xgboost"


class TestPredictMainLengthMismatch:
    """Regression test: when the model returns fewer predictions than there are
    validation rows (e.g. a series too short to form an encoder window),
    predict.main() must truncate instead of crashing — the length check has to
    run BEFORE the `val_df["sales_log_pred"] = y_pred` assignment, because
    pandas raises ValueError on a length mismatch at assignment time."""

    def test_mismatched_prediction_length_is_truncated(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        import foresight.predict as pred

        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "store_nbr": 1,
                "family": "GROCERY",
                "sales_log": np.log1p(np.arange(20, dtype=float) + 10),
            }
        )
        input_csv = tmp_path / "features.csv"
        df.to_csv(input_csv, index=False)
        output_csv = tmp_path / "predictions.csv"

        # Simulate a model that emits fewer predictions than validation rows.
        monkeypatch.setattr(
            pred, "predict_model", lambda d, name, min_target_date=None: np.array([0.5])
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict",
                "--input",
                str(input_csv),
                "--model",
                "xgboost",
                "--val_days",
                "4",
                "--output",
                str(output_csv),
            ],
        )
        pred.main()  # must not raise ValueError

        out = pd.read_csv(output_csv)
        assert len(out) == 1
        assert out["sales_log_pred"].tolist() == [0.5]


# ---------------------------------------------------------------------------
# train_baseline — model_results.json upsert
# ---------------------------------------------------------------------------


class TestTrainBaselineResultsUpsert:
    """Regression test: train_baseline.main() must merge baseline_results into
    the existing model_results.json (read-modify-write) instead of overwriting
    the whole file and silently deleting lstm_results / transformer_results."""

    def test_main_preserves_existing_dl_results(self, tmp_path, monkeypatch):
        pytest.importorskip("xgboost")
        import foresight.train_baseline as tb

        # Feature CSV just needs enough rows for the time-based split.
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "store_nbr": 1,
                "family": "GROCERY",
                "sales_log": np.log1p(np.arange(40, dtype=float) + 10),
                "onpromotion": 0,
            }
        )
        input_csv = tmp_path / "features.csv"
        df.to_csv(input_csv, index=False)

        results_json = tmp_path / "model_results.json"
        results_json.write_text(
            json.dumps(
                {
                    "lstm_results": [{"model": "lstm", "mae": 0.3}],
                    "transformer_results": [{"model": "transformer", "mae": 0.4}],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(tb, "MODEL_RESULTS_JSON", results_json)

        # Skip real training — only the results-JSON write is under test.
        fake_xgb = {"model": "xgboost", "mae": 0.5, "rmse": 0.6, "mape": 1.0, "smape": 1.0}
        fake_prophet = {"model": "prophet", "mae": None, "rmse": None, "mape": None, "smape": None}
        monkeypatch.setattr(tb, "train_xgboost", lambda train, val: (None, fake_xgb))
        monkeypatch.setattr(tb, "train_prophet", lambda train, val: (None, fake_prophet))
        monkeypatch.setattr(sys, "argv", ["train_baseline", "--input", str(input_csv)])

        tb.main()

        saved = json.loads(results_json.read_text(encoding="utf-8"))
        assert saved["lstm_results"] == [{"model": "lstm", "mae": 0.3}]
        assert saved["transformer_results"] == [{"model": "transformer", "mae": 0.4}]
        assert saved["baseline_results"] == [fake_xgb, fake_prophet]


# ---------------------------------------------------------------------------
# generate_mock_data — synthetic generators
# ---------------------------------------------------------------------------


class TestMockDataGenerators:
    def test_generate_oil_prices_shape(self):
        from foresight.generate_mock_data import generate_oil_prices

        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        oil = generate_oil_prices(dates)
        assert len(oil) == 30
        assert "dcoilwtico" in oil.columns
        assert (oil["dcoilwtico"] > 0).all()

    def test_generate_stores_shape(self):
        from foresight.generate_mock_data import generate_stores

        stores = generate_stores(n_stores=5)
        assert len(stores) == 5
        assert "store_nbr" in stores.columns
        assert "type" in stores.columns

    def test_generate_items_shape(self):
        from foresight.generate_mock_data import generate_items

        items = generate_items(n_items=10)
        assert len(items) == 10
        assert "family" in items.columns

    def test_generate_holidays_returns_df(self):
        from foresight.generate_mock_data import generate_holidays

        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        holidays = generate_holidays(dates)
        assert isinstance(holidays, pd.DataFrame)
        assert "date" in holidays.columns
        assert "transferred" in holidays.columns
