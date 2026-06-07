"""Project-wide configuration for multivariate-timeseries-forecasting.

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

# Reports
MODEL_RESULTS_JSON = REPORTS_DIR / "model_results.json"
FORECAST_PLOT_PNG = IMAGES_DIR / "forecast_comparison.png"
RESIDUAL_PLOT_PNG = IMAGES_DIR / "residual_distribution.png"

# ── Modeling constants ────────────────────────────────────────────
RANDOM_STATE = 42

# ── Deep learning constants ───────────────────────────────────────
BATCH_SIZE = 128
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 10  # early stopping patience
SEQ_LENGTH = 28  # lookback window for LSTM/Transformer
D_MODEL = 64  # Transformer embedding dimension
N_HEADS = 4  # Transformer attention heads
N_LAYERS = 2  # Transformer encoder layers
DROPOUT = 0.1
