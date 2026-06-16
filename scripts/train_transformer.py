"""Transformer (Informer-style) model for time series forecasting.

Architecture:
- Positional encoding + embedding for categorical features
- Transformer encoder for temporal attention
- Fully-connected output head
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Rich progress bar uses Windows-console APIs that fail on GBK code pages.
# Disable it early so PyTorch Lightning falls back to a plain progress bar.
if os.name == "nt":
    os.environ.setdefault("PYTORCH_ENABLE_RICH", "0")

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    D_MODEL,
    DROPOUT,
    FEATURES_TRAIN_CSV,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_RESULTS_JSON,
    N_HEADS,
    N_LAYERS,
    PATIENCE,
    RANDOM_STATE,
    REPORTS_DIR,
    TRANSFORMER_MODEL_PATH,
)
from scripts.metrics import TimeSeriesDataset, mape, smape

pl.seed_everything(RANDOM_STATE, workers=True)


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


class TransformerForecastModule(pl.LightningModule):
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

    def training_step(self, batch, batch_idx):
        y_hat = self(batch["cat"], batch["num"])
        loss = self.criterion(y_hat, batch["y"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch["cat"], batch["num"])
        loss = self.criterion(y_hat, batch["y"])
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FEATURES_TRAIN_CSV)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    print(f"Loading data from {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["date"])

    # Split by time — last N days as validation
    val_days = 16
    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(days=val_days - 1)
    train_df = df[df["date"] < val_start].copy()
    val_df = df[df["date"] >= val_start].copy()

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    train_ds = TimeSeriesDataset(train_df)
    val_ds = TimeSeriesDataset(val_df, encoder=train_ds.encoders, scalers=train_ds.scalers)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_stores = train_ds.df["store_nbr"].nunique()
    num_families = train_ds.df["family"].nunique()
    num_numeric = len(train_ds.numeric_cols)

    model = TransformerForecastModule(
        num_stores=num_stores,
        num_families=num_families,
        num_numeric=num_numeric,
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="transformer-{epoch:02d}-{val_loss:.4f}",
        dirpath=REPORTS_DIR / "checkpoints",
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint, early_stop],
        accelerator="auto",
        enable_progress_bar=sys.platform != "win32",
        deterministic=True,
    )

    print("\nTraining Transformer ...")
    trainer.fit(model, train_loader, val_loader)

    # Evaluate
    print("\nEvaluating on validation set ...")
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in val_loader:
            y_hat = model(batch["cat"], batch["num"])
            all_preds.extend(y_hat.cpu().numpy())
            all_true.extend(batch["y"].cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    metrics = {
        "model": "transformer",
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
    }
    print(
        f"  Transformer  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
        f"MAPE={metrics['mape']:.2f}%  sMAPE={metrics['smape']:.2f}%"
    )

    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), TRANSFORMER_MODEL_PATH)
    joblib.dump(
        {
            "encoders": train_ds.encoders,
            "scalers": train_ds.scalers,
            "numeric_cols": train_ds.numeric_cols,
        },
        TRANSFORMER_MODEL_PATH.with_suffix(".meta.joblib"),
    )

    results_path = MODEL_RESULTS_JSON
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results["transformer_results"] = [metrics]
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nModel saved: {TRANSFORMER_MODEL_PATH}")


if __name__ == "__main__":
    main()
