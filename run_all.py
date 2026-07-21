"""Windows-compatible one-shot pipeline runner.

Replaces `make all` on systems without GNU Make (e.g., Windows).
Usage:
    python run_all.py
    python run_all.py --quick          # fast CPU verification (1 epoch)
    python run_all.py --max-epochs 10  # custom DL training budget
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 mode for the whole subprocess tree on Windows before any heavy
# imports (e.g. PyTorch Lightning's Rich progress bar) are loaded.
os.environ.setdefault("PYTHONUTF8", "1")


def run(cmd: list[str], cwd: Path | None = None):
    print(f"\n{'=' * 60}")
    print(f">>> {' '.join(cmd)}")
    print("=" * 60)
    # cmd is a list; no shell=True — avoids shell-injection surface and
    # correctly handles paths with spaces.
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"WARNING: Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the full multivariate time-series forecasting pipeline."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Max epochs for LSTM/Transformer (default: config.MAX_EPOCHS).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stopping patience for LSTM/Transformer (default: config.PATIENCE).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast verification mode: 1 epoch, patience=1.",
    )
    args = parser.parse_args()

    if args.quick:
        max_epochs = 1
        patience = 1
    else:
        max_epochs = args.max_epochs
        patience = args.patience

    dl_flags: list[str] = []
    if max_epochs is not None:
        dl_flags += ["--max_epochs", str(max_epochs)]
    if patience is not None:
        dl_flags += ["--patience", str(patience)]

    here = Path(__file__).resolve().parent

    steps = [
        ("Preprocessing", [sys.executable, "-m", "foresight.preprocess"]),
        ("Feature Engineering", [sys.executable, "-m", "foresight.feature_engineering"]),
        ("Baseline Training (XGBoost)", [sys.executable, "-m", "foresight.train_baseline"]),
        ("LSTM Training", [sys.executable, "-m", "foresight.train_lstm", *dl_flags]),
        ("Transformer Training", [sys.executable, "-m", "foresight.train_transformer", *dl_flags]),
        ("Evaluation", [sys.executable, "-m", "foresight.evaluate"]),
    ]

    print("Multivariate Time Series Forecasting - Full Pipeline")
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
