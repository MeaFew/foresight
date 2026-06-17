"""Shared metrics and dataset utilities for time series forecasting.

Provides:
- TimeSeriesDataset: PyTorch sliding-window dataset (shared by LSTM & Transformer)
- mape/smape: evaluation metrics (shared by baseline, LSTM, Transformer)

``mape``/``smape`` are pure-numpy and live in ``scripts.metrics_utils``; they
are re-exported here so existing ``from scripts.metrics import mape, smape``
imports keep working. Importing the torch-free helpers directly from
``metrics_utils`` avoids pulling in the PyTorch stack (useful in tests).
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset

from config import SEQ_LENGTH
from scripts.metrics_utils import mape, smape

__all__ = ["mape", "smape", "TimeSeriesDataset"]


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for sliding-window time series."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_length: int = SEQ_LENGTH,
        encoder: dict | None = None,
        scalers: dict | None = None,
    ):
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

        # Scale numeric features. Use is_numeric_dtype (not a fixed dtype list)
        # so float32 / Int32 / nullable dtypes are not silently dropped — this
        # keeps the DL feature set consistent with the XGBoost path, which uses
        # select_dtypes(include=[np.number]).
        _exclude = {"date", "sales", "sales_log", "id", "store_nbr", "family"}
        numeric_cols = [
            c for c in self.df.columns
            if c not in _exclude and pd.api.types.is_numeric_dtype(self.df[c])
        ]
        self.numeric_cols = numeric_cols
        self.scalers = scalers or {}
        for col in numeric_cols:
            if col not in self.scalers:
                sc = StandardScaler()
                self.df[col] = sc.fit_transform(self.df[[col]].fillna(0))
                self.scalers[col] = sc
            else:
                self.df[col] = self.scalers[col].transform(self.df[[col]].fillna(0))

        # Group by series and cache the reset-indexed groups so __getitem__ is
        # O(1) (a slice) instead of re-scanning the whole frame per sample
        # (which made each epoch O(N^2) over the dataset size).
        self.groups = [
            g.reset_index(drop=True)
            for _, g in self.df.groupby(["store_nbr", "family"], sort=False)
        ]
        self.samples = []
        for gid, group in enumerate(self.groups):
            for i in range(seq_length, len(group)):
                self.samples.append((gid, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gid, end_idx = self.samples[idx]
        group = self.groups[gid]

        seq = group.iloc[end_idx - self.seq_length : end_idx]
        target = group.iloc[end_idx]["sales_log"]

        x_cat = torch.tensor(seq[["store_nbr", "family"]].values, dtype=torch.long)
        x_num = torch.tensor(seq[self.numeric_cols].values, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)

        return {"cat": x_cat, "num": x_num, "y": y}
