"""LSTM model for time series forecasting using PyTorch Lightning.

Architecture:
- Embedding layers for categorical features (store, family)
- LSTM encoder for temporal patterns
- Fully-connected output head

Evaluation: MAE, RMSE, MAPE, sMAPE on validation set.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BATCH_SIZE,
    CLEANED_TRAIN_CSV,
    DROPOUT,
    FEATURES_TRAIN_CSV,
    LSTM_MODEL_PATH,
    MAX_EPOCHS,
    MODEL_RESULTS_JSON,
    PATIENCE,
    RANDOM_STATE,
    REPORTS_DIR,
    SEQ_LENGTH,
)

pl.seed_everything(RANDOM_STATE, workers=True)


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for sliding-window time series."""

    def __init__(self, df: pd.DataFrame, seq_length: int = SEQ_LENGTH, encoder: dict | None = None, scaler: dict | None = None):
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
        self.scalers = scaler or {}
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


class LSTMForecastModule(pl.LightningModule):
    """PyTorch Lightning module for LSTM forecasting."""

    def __init__(
        self,
        num_stores: int,
        num_families: int,
        num_numeric: int,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = DROPOUT,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.store_embed = nn.Embedding(num_stores, embed_dim)
        self.family_embed = nn.Embedding(num_families, embed_dim)

        input_dim = embed_dim * 2 + num_numeric
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.criterion = nn.MSELoss()

    def forward(self, cat, num):
        store_emb = self.store_embed(cat[:, :, 0])
        family_emb = self.family_embed(cat[:, :, 1])
        x = torch.cat([store_emb, family_emb, num], dim=-1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
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

    # Split by time
    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(days=60)
    train_df = df[df["date"] < val_start].copy()
    val_df = df[df["date"] >= val_start].copy()

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    # Datasets
    train_ds = TimeSeriesDataset(train_df)
    val_ds = TimeSeriesDataset(val_df, encoder=train_ds.encoders, scaler=train_ds.scalers)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    num_stores = train_ds.df["store_nbr"].nunique()
    num_families = train_ds.df["family"].nunique()
    num_numeric = len(train_ds.numeric_cols)

    model = LSTMForecastModule(
        num_stores=num_stores,
        num_families=num_families,
        num_numeric=num_numeric,
    )

    # Trainer
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="lstm-{epoch:02d}-{val_loss:.4f}",
        dirpath=REPORTS_DIR / "checkpoints",
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint, early_stop],
        accelerator="auto",
        enable_progress_bar=True,
        deterministic=True,
    )

    print("\nTraining LSTM ...")
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
        "model": "lstm",
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
    }
    print(f"  LSTM  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
          f"MAPE={metrics['mape']:.2f}%  sMAPE={metrics['smape']:.2f}%")

    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LSTM_MODEL_PATH)
    joblib.dump({"encoders": train_ds.encoders, "scalers": train_ds.scalers, "numeric_cols": train_ds.numeric_cols},
                LSTM_MODEL_PATH.with_suffix(".meta.joblib"))

    # Update results
    results_path = MODEL_RESULTS_JSON
    if results_path.exists():
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results["lstm_results"] = metrics
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nModel saved: {LSTM_MODEL_PATH}")


if __name__ == "__main__":
    main()
