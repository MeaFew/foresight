"""Transformer (Informer-style) model for time series forecasting.

Architecture: positional encoding + embedding for categorical features ->
Transformer encoder -> FC head. Training/evaluation plumbing is shared via
scripts/train_common.py.
"""

import math
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_common import BaseForecastModule, build_arg_parser, load_and_split, train_and_evaluate

from config import (
    D_MODEL,
    DROPOUT,
    FEATURES_TRAIN_CSV,
    LEARNING_RATE,
    N_HEADS,
    N_LAYERS,
    RANDOM_STATE,
    TRANSFORMER_MODEL_PATH,
)
from scripts.metrics import TimeSeriesDataset


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerForecastModule(BaseForecastModule):
    """PyTorch Lightning module for Transformer forecasting."""

    def __init__(
        self,
        num_stores: int,
        num_families: int,
        num_numeric: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        lr: float = LEARNING_RATE,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.store_embed = nn.Embedding(num_stores, d_model // 4)
        self.family_embed = nn.Embedding(num_families, d_model // 4)
        self.num_proj = nn.Linear(num_numeric, d_model // 2)

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.criterion = nn.MSELoss()

    def forward(self, cat, num):
        store_emb = self.store_embed(cat[:, :, 0])
        family_emb = self.family_embed(cat[:, :, 1])
        num_emb = self.num_proj(num)
        x = torch.cat([store_emb, family_emb, num_emb], dim=-1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out.squeeze(-1)


def main():
    pl.seed_everything(RANDOM_STATE, workers=True)
    args = build_arg_parser().parse_args()
    if args.input is None:
        args.input = FEATURES_TRAIN_CSV

    print(f"Loading data from {args.input} ...")
    train_df, val_df = load_and_split(args.input)

    train_ds = TimeSeriesDataset(train_df)
    val_ds = TimeSeriesDataset(val_df, encoder=train_ds.encoders, scalers=train_ds.scalers)

    num_stores = train_ds.n_stores
    num_families = train_ds.n_families
    num_numeric = len(train_ds.numeric_cols)

    model = TransformerForecastModule(
        num_stores=num_stores,
        num_families=num_families,
        num_numeric=num_numeric,
    )

    train_and_evaluate(
        model,
        train_ds,
        val_ds,
        args,
        name="Transformer",
        checkpoint_filename="transformer-{epoch:02d}-{val_loss:.4f}",
        state_dict_path=TRANSFORMER_MODEL_PATH,
        results_key="transformer_results",
    )


if __name__ == "__main__":
    main()
