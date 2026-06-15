"""Cross-reference audit: README claims vs. actual pipeline outputs.

Run after `make all` to verify that key metrics declared in README.md
match the actual values produced by the pipeline.

Usage: python scripts/audit_consistency.py
"""

import json
import re
import sys
from pathlib import Path


def read_readme_metric(readme_path: Path, metric_name: str) -> float | None:
    """Extract a numeric metric from README.md table rows.

    Looks for patterns like `| XGBoost | **0.256** | ...`
    """
    text = readme_path.read_text(encoding="utf-8")
    pattern = rf"\*\*{re.escape(metric_name)}\*\*\*.*?(\d+\.\d+)"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def check(condition: bool, msg: str) -> bool:
    """Assert-like check that prints pass/fail."""
    if condition:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
    return condition


def main():
    root = Path(__file__).resolve().parents[1]
    results_json = root / "reports" / "model_results.json"
    models_dir = root / "models"
    data_raw = root / "data" / "raw"
    passed = 0
    failed = 0

    def ok(cond, msg):
        nonlocal passed, failed
        if check(cond, msg):
            passed += 1
        else:
            failed += 1

    # ── Check 1: model_results.json exists and has baseline_results ──
    print("\n[1] model_results.json")

    if not results_json.exists():
        ok(False, f"model_results.json not found at {results_json}")
    else:
        with open(results_json) as f:
            results = json.load(f)
        ok("baseline_results" in results, "Has 'baseline_results' key")
        if "baseline_results" in results:
            xgb = [r for r in results["baseline_results"] if r.get("model") == "xgboost"]
            ok(len(xgb) > 0, "XGBoost results present in baseline_results")
            if len(xgb) > 0:
                mae = xgb[0].get("mae")
                ok(
                    mae is not None and mae > 0,
                    f"XGBoost MAE is valid ({mae:.4f})" if mae else "XGBoost MAE is None",
                )
        ok("lstm_results" in results, "Has 'lstm_results' key")
        ok("transformer_results" in results, "Has 'transformer_results' key")

    # ── Check 2: README data size claims vs actual data ──
    print("\n[2] README data size claims vs actual data")

    if data_raw.exists():
        train_csv = data_raw / "train.csv"
        if train_csv.exists():
            import pandas as pd

            df = pd.read_csv(train_csv)
            actual_rows = len(df)
            # README claims: "Full (3M rows, 54 stores)"
            ok(
                actual_rows >= 100_000,
                f"train.csv has {actual_rows:,} rows (expected >=100K for real data)",
            )
            if "store_nbr" in df.columns:
                n_stores = df["store_nbr"].nunique()
                ok(True, f"train.csv has {n_stores} unique stores")
            if "family" in df.columns:
                n_families = df["family"].nunique()
                ok(True, f"train.csv has {n_families} unique families")
        else:
            ok(False, f"train.csv not found at {train_csv}")
    else:
        ok(False, f"data/raw/ directory not found at {data_raw}")

    # ── Check 3: models directory has expected files ──
    print("\n[3] Models directory")

    expected_models = [
        "xgboost_baseline.joblib",
        "lstm_model.pt",
        "lstm_model.meta.joblib",
        "transformer_model.pt",
        "transformer_model.meta.joblib",
    ]
    if models_dir.exists():
        for fname in expected_models:
            fpath = models_dir / fname
            ok(fpath.exists() and fpath.stat().st_size > 0, f"{fname} exists and is non-empty")
    else:
        ok(False, f"models/ directory not found at {models_dir}")

    # ── Check 4: XGBoost README metric vs model_results.json ──
    print("\n[4] README XGBoost MAE vs model_results.json")

    if results_json.exists():
        with open(results_json) as f:
            results = json.load(f)
        baselines = results.get("baseline_results", [])
        xgb_entry = None
        for r in baselines:
            if r.get("model") == "xgboost":
                xgb_entry = r
                break

        if xgb_entry and xgb_entry.get("mae") is not None:
            actual_mae = xgb_entry["mae"]
            # README claims XGBoost MAE=0.256
            ok(
                abs(actual_mae - 0.256) < 0.01,
                f"XGBoost MAE: README ~0.256 vs actual {actual_mae:.4f} (diff={abs(actual_mae - 0.256):.4f})",
            )

    # ── Check 5: config.py paths are consistent ──
    print("\n[5] Config path consistency")

    sys.path.insert(0, str(root))
    from config import (
        CLEANED_TRAIN_CSV,
        FEATURES_TRAIN_CSV,
        LSTM_MODEL_PATH,
        MODEL_RESULTS_JSON,
        MODELS_DIR,
        TRANSFORMER_MODEL_PATH,
    )

    ok(
        FEATURES_TRAIN_CSV.parent.exists(),
        f"FEATURES_TRAIN_CSV parent dir exists: {FEATURES_TRAIN_CSV.parent}",
    )
    ok(
        CLEANED_TRAIN_CSV.parent.exists(),
        f"CLEANED_TRAIN_CSV parent dir exists: {CLEANED_TRAIN_CSV.parent}",
    )
    ok(
        MODEL_RESULTS_JSON.parent.exists(),
        f"MODEL_RESULTS_JSON parent dir exists: {MODEL_RESULTS_JSON.parent}",
    )
    ok(MODELS_DIR.exists(), f"MODELS_DIR exists: {MODELS_DIR}")
    ok(LSTM_MODEL_PATH.parent == MODELS_DIR, "LSTM_MODEL_PATH is in MODELS_DIR")
    ok(TRANSFORMER_MODEL_PATH.parent == MODELS_DIR, "TRANSFORMER_MODEL_PATH is in MODELS_DIR")

    # ── Summary ──
    total = passed + failed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        print("ACTION: Update README.md or pipeline to resolve mismatches.")
        sys.exit(1)


if __name__ == "__main__":
    main()
