"""Project-wide configuration for foresight.

All paths are resolved relative to this file's location.
"""

from pathlib import Path

# ── Base directories ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
IMAGES_DIR = BASE_DIR / "images"

# ── Output files ──────────────────────────────────────────────────
CLEANED_TRAIN_CSV = PROCESSED_DATA_DIR / "train_cleaned.csv"
FEATURES_TRAIN_CSV = PROCESSED_DATA_DIR / "features_train.csv"

# Model checkpoints
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.pt"
TRANSFORMER_MODEL_PATH = MODELS_DIR / "transformer_model.pt"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_baseline.joblib"

# Reports
MODEL_RESULTS_JSON = REPORTS_DIR / "model_results.json"
FORECAST_PLOT_PNG = IMAGES_DIR / "forecast_comparison.png"
RESIDUAL_PLOT_PNG = IMAGES_DIR / "residual_distribution.png"

# ── Modeling constants ────────────────────────────────────────────
RANDOM_STATE = 42
# Number of days reserved for the validation split (held out from the tail of
# the training series). Used by every trainer and by evaluate.py so the split
# is defined in one place — a previous version hard-coded 16 in trainers and
# 15 (an off-by-window-size literal) in evaluate.py.
VAL_DAYS = 16

# ── Deep learning constants ───────────────────────────────────────
# batch_size raised to 1024: on a 4060 8GB the model uses <1GB even at this
# size, and larger batches cut the per-epoch step count (17.7k → 2.2k) which
# removes the kernel-launch overhead that dominated training time at bs=128
# (benchmark: bs=128 → 72s/epoch, bs=1024 → 38s/epoch). lr scaled up to match
# (1e-3 was tuned for bs=128; larger batches tolerate a larger step).
BATCH_SIZE = 1024
MAX_EPOCHS = 100
LEARNING_RATE = 3e-3
PATIENCE = 10  # early stopping patience
SEQ_LENGTH = 28  # lookback window for LSTM/Transformer
D_MODEL = 64  # Transformer embedding dimension
N_HEADS = 4  # Transformer attention heads
N_LAYERS = 2  # Transformer encoder layers
DROPOUT = 0.1
