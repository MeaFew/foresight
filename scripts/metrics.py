"""Shared metrics and dataset utilities for time series forecasting.

Provides:
- TimeSeriesDataset: PyTorch sliding-window dataset (shared by LSTM & Transformer)
- mape/smape: evaluation metrics (shared by baseline, LSTM, Transformer)
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

from config import SEQ_LENGTH


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (skips zeros in true values)."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for sliding-window time series."""

    def __init__(self, df: pd.DataFrame, seq_length: int = SEQ_LENGTH, encoder: dict | None = None, scalers: dict | None = None):
        self.seq_length = seq_length
        self.df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

        # Encode categoricals
        cat_cols = ["store_nbr", "family"]
        self.encoders = encoder or {}
        for col in cat_cols:
            if col not in self.encoders:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
            else:
                self.df[col] = self.encoders[col].transform(self.df[col].astype(str))

        # Scale numeric features
        numeric_cols = [c for c in self.df.columns
                        if c not in ["date", "sales", "sales_log", "id", "store_nbr", "family"]
                        and self.df[c].dtype in [np.float64, np.int64]]
        self.numeric_cols = numeric_cols
        self.scalers = scalers or {}
        for col in numeric_cols:
            if col not in self.scalers:
                sc = StandardScaler()
                self.df[col] = sc.fit_transform(self.df[[col]].fillna(0))
                self.scalers[col] = sc
            else:
                self.df[col] = self.scalers[col].transform(self.df[[col]].fillna(0))

        # Group by series
        self.series = list(self.df.groupby(["store_nbr", "family"]))
        self.samples = []
        for (store, family), group in self.series:
            group = group.reset_index(drop=True)
            for i in range(seq_length, len(group)):
                self.samples.append((store, family, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        store, family, end_idx = self.samples[idx]
        group = self.df[(self.df["store_nbr"] == store) & (self.df["family"] == family)].reset_index(drop=True)

        seq = group.iloc[end_idx - self.seq_length:end_idx]
        target = group.iloc[end_idx]["sales_log"]

        x_cat = torch.tensor(seq[["store_nbr", "family"]].values, dtype=torch.long)
        x_num = torch.tensor(seq[self.numeric_cols].values, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)

        return {"cat": x_cat, "num": x_num, "y": y}
