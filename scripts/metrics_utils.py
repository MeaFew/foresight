"""Pure-numpy metric helpers shared across forecasting models.

This module is intentionally torch-free so it can be imported in lightweight
contexts (unit tests, CI matrices without the deep-learning stack) and reused
by baseline models that don't need PyTorch.

``scripts.metrics`` re-exports these for backwards compatibility — existing
``from scripts.metrics import mape, smape`` statements keep working.
"""

import numpy as np


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (skips zeros in true values)."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
