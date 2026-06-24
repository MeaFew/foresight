"""LSTM model for time series forecasting.

Architecture: embedding for categorical features -> 2-layer LSTM -> FC head.
Training/evaluation plumbing is shared via scripts/train_common.py.
"""

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_common import BaseForecastModule, build_arg_parser, load_and_split, train_and_evaluate

from config import FEATURES_TRAIN_CSV, LEARNING_RATE, LSTM_MODEL_PATH, RANDOM_STATE
from scripts.metrics import TimeSeriesDataset


class LSTMForecastModule(BaseForecastModule):
    """PyTorch Lightning module for LSTM forecasting."""

    def __init__(
        self,
        num_stores: int,
        num_families: int,
        num_numeric: int,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        lr: float = LEARNING_RATE,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.store_embed = nn.Embedding(num_stores, embed_dim)
        self.family_embed = nn.Embedding(num_families, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim * 2 + num_numeric,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.criterion = nn.MSELoss()

    def forward(self, cat, num):
        store_emb = self.store_embed(cat[:, :, 0])
        family_emb = self.family_embed(cat[:, :, 1])
        x = torch.cat([store_emb, family_emb, num], dim=-1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


def main():
    pl.seed_everything(RANDOM_STATE, workers=True)
    args = build_arg_parser().parse_args()
    if args.input is None:
        args.input = FEATURES_TRAIN_CSV

    print(f"Loading data from {args.input} ...")
    train_df, val_df, val_target_start = load_and_split(args.input)

    train_ds = TimeSeriesDataset(train_df)
    # min_target_date drops samples whose prediction target is in the training
    # period (the prepended context rows are window INPUT only) — otherwise the
    # validation loss/MAE would mix in train-period targets and overstate DL.
    val_ds = TimeSeriesDataset(
        val_df,
        encoder=train_ds.encoders,
        scalers=train_ds.scalers,
        min_target_date=val_target_start,
    )

    num_stores = train_ds.n_stores
    num_families = train_ds.n_families
    num_numeric = len(train_ds.numeric_cols)

    model = LSTMForecastModule(
        num_stores=num_stores, num_families=num_families, num_numeric=num_numeric
    )

    train_and_evaluate(
        model,
        train_ds,
        val_ds,
        args,
        name="LSTM",
        checkpoint_filename="lstm-{epoch:02d}-{val_loss:.4f}",
        state_dict_path=LSTM_MODEL_PATH,
        results_key="lstm_results",
    )


if __name__ == "__main__":
    main()
