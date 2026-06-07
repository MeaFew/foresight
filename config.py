"""Project-wide configuration for multivariate-timeseries-forecasting.

All paths are resolved relative to this file's location.
"""

from pathlib import Path

# ── Base directories ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
IMAGES_DIR = BASE_DIR / "images"
DASHBOARD_DIR = BASE_DIR / "dashboard"
DOCS_DIR = BASE_DIR / "docs"

# ── Input files ───────────────────────────────────────────────────
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
STORES_CSV = RAW_DATA_DIR / "stores.csv"
ITEMS_CSV = RAW_DATA_DIR / "items.csv"
TRANSACTIONS_CSV = RAW_DATA_DIR / "transactions.csv"
HOLIDAYS_CSV = RAW_DATA_DIR / "holidays_events.csv"
OIL_CSV = RAW_DATA_DIR / "oil.csv"

# ── Output files ──────────────────────────────────────────────────
CLEANED_TRAIN_CSV = PROCESSED_DATA_DIR / "train_cleaned.csv"
FEATURES_TRAIN_CSV = PROCESSED_DATA_DIR / "features_train.csv"
FEATURES_TEST_CSV = PROCESSED_DATA_DIR / "features_test.csv"

# Model checkpoints
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.pt"
TRANSFORMER_MODEL_PATH = MODELS_DIR / "transformer_model.pt"
PROPHET_MODEL_PATH = MODELS_DIR / "prophet_model.json"

# Reports
MODEL_RESULTS_JSON = REPORTS_DIR / "model_results.json"
FORECAST_PLOT_PNG = IMAGES_DIR / "forecast_comparison.png"
ATTENTION_HEATMAP_PNG = IMAGES_DIR / "attention_heatmap.png"
RESIDUAL_PLOT_PNG = IMAGES_DIR / "residual_distribution.png"

# ── Modeling constants ────────────────────────────────────────────
RANDOM_STATE = 42
FORECAST_HORIZON = 16  # days to forecast (Kaggle competition format)

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

# ── Time series-specific ──────────────────────────────────────────
# Date range (Store Sales dataset)
TRAIN_START = "2013-01-01"
TRAIN_END = "2017-08-15"
TEST_START = "2017-08-16"
TEST_END = "2017-08-31"

# External regressors
USE_PROMO = True
USE_OIL = True
USE_HOLIDAYS = True
