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
    """PyTorch Dataset for sliding-window time series.

    ``min_target_date`` (optional): if set, only samples whose PREDICTION TARGET
    (the window's last row, ``end_idx``) falls on or after this date are emitted.
    This is used by the validation dataset to drop samples whose target date is
    actually in the training period — those rows are present only as window
    CONTEXT (so the first true-validation targets have enough lookback), not as
    prediction targets. Without this filter the validation loss/metrics mix in
    train-period targets and overstate DL performance.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_length: int = SEQ_LENGTH,
        encoder: dict | None = None,
        scalers: dict | None = None,
        min_target_date: pd.Timestamp | None = None,
    ):
        self.seq_length = seq_length
        self.min_target_date = min_target_date
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
            c
            for c in self.df.columns
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

        # Group by series and pre-convert each group to contiguous numpy arrays
        # so __getitem__ is a cheap array slice instead of a pandas iloc call.
        # The previous version stored DataFrame groups and called .iloc per
        # sample, which on a 2.3M-row / 2.27M-sample training set made each
        # epoch take many minutes purely on indexing. With numpy, __getitem__
        # is ~50× faster and the single-threaded loader keeps up with the GPU.
        self.n_stores = self.df["store_nbr"].nunique()
        self.n_families = self.df["family"].nunique()
        self.groups_cat = []  # list of (len_i, 2) int64 arrays per series
        self.groups_num = []  # list of (len_i, num_numeric) float32 arrays
        self.groups_y = []  # list of (len_i,) float32 arrays (sales_log) per series
        self.groups_date = []  # list of (len_i,) datetime64 arrays per series
        for _, g in self.df.groupby(["store_nbr", "family"], sort=False):
            g = g.reset_index(drop=True)
            self.groups_cat.append(g[["store_nbr", "family"]].to_numpy(dtype=np.int64))
            self.groups_num.append(g[self.numeric_cols].to_numpy(dtype=np.float32))
            self.groups_y.append(g["sales_log"].to_numpy(dtype=np.float32))
            self.groups_date.append(g["date"].to_numpy())
        # Free the raw frame — all data now lives in the per-series arrays.
        self.df = None

        self.samples = []
        for gid in range(len(self.groups_cat)):
            n = len(self.groups_cat[gid])
            dates = self.groups_date[gid]
            for i in range(seq_length, n):
                # If a min_target_date is set (validation set with prepended
                # context), drop samples whose target date is before the true
                # validation start — those are train-period targets included
                # only to give the first validation windows enough lookback.
                if min_target_date is not None and pd.Timestamp(dates[i]) < min_target_date:
                    continue
                self.samples.append((gid, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gid, end_idx = self.samples[idx]
        start = end_idx - self.seq_length
        x_cat = torch.from_numpy(self.groups_cat[gid][start:end_idx])
        x_num = torch.from_numpy(self.groups_num[gid][start:end_idx])
        y = torch.from_numpy(self.groups_y[gid][end_idx : end_idx + 1]).squeeze()
        return {"cat": x_cat, "num": x_num, "y": y}
