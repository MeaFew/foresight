"""Windows-compatible one-shot pipeline runner.

Replaces `make all` on systems without GNU Make (e.g., Windows).
Usage: python run_all.py
"""

import subprocess
import sys
from pathlib import Path


def run(cmd: str, cwd: Path | None = None):
    print(f"\n{'=' * 60}")
    print(f">>> {cmd}")
    print("=" * 60)
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"WARNING: Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    here = Path(__file__).resolve().parent

    steps = [
        ("Preprocessing", "python scripts/preprocess.py"),
        ("Feature Engineering", "python scripts/feature_engineering.py"),
        ("Baseline Training (XGBoost)", "python scripts/train_baseline.py"),
        ("LSTM Training", "python scripts/train_lstm.py"),
        ("Transformer Training", "python scripts/train_transformer.py"),
        ("Evaluation", "python scripts/evaluate.py"),
    ]

    print("Multivariate Time Series Forecasting — Full Pipeline")
    print("=" * 60)

    for name, cmd in steps:
        if not run(cmd, cwd=here):
            print(f"\nPipeline stopped at step: {name}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
