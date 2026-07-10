"""foresight — multivariate time-series forecasting pipeline.

XGBoost / Prophet / LSTM / Transformer comparison with leakage prevention.
"""

from __future__ import annotations

import logging

__version__ = "1.0.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
