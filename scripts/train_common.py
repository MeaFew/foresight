"""Shared training/evaluation plumbing for the DL forecast models.

train_lstm.py and train_transformer.py previously duplicated ~80% of their
code: the Lightning training_step/validation_step/configure_optimizers, the
data loading + split + DataLoader setup, the train loop, the eval loop, the
artifact save, and the model_results.json read-modify-write. This module
holds the shared parts so each trainer is a thin wrapper that only defines
its model architecture.
"""

import json
import os
import sys
from pathlib import Path

# Rich progress bar uses Windows-console APIs that fail on GBK code pages.
if os.name == "nt":
    os.environ.setdefault("PYTORCH_ENABLE_RICH", "0")

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_utils import compute_metrics, time_train_val_split

from config import BATCH_SIZE, MAX_EPOCHS, MODEL_RESULTS_JSON, PATIENCE, REPORTS_DIR, VAL_DAYS


class BaseForecastModule(pl.LightningModule):
    """Shared Lightning boilerplate: MSE loss + Adam + train/val logging.

    Subclasses (LSTMForecastModule, TransformerForecastModule) only need to
    implement ``forward(cat, num)`` and set ``self.criterion`` / ``self.lr``.
    """

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


def load_and_split(input_path: Path):
    """Read the feature CSV and apply the shared time-based train/val split."""
    df = pd.read_csv(input_path, parse_dates=["date"])
    train_df, val_df = time_train_val_split(df, VAL_DAYS)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    return train_df, val_df


def make_loaders(train_ds, val_ds, batch_size: int = BATCH_SIZE):
    """Build shuffling train / non-shuffling val DataLoaders."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_and_evaluate(
    model,
    train_ds,
    val_ds,
    args,
    name: str,
    checkpoint_filename: str,
    state_dict_path: Path,
    results_key: str,
):
    """Run fit + eval + save artifacts + upsert results JSON.

    Shared by train_lstm.py and train_transformer.py. ``name`` is the display
    label ("LSTM" / "Transformer"); ``results_key`` is the JSON key
    ("lstm_results" / "transformer_results").
    """
    train_loader, val_loader = make_loaders(train_ds, val_ds, args.batch_size)

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=checkpoint_filename,
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

    print(f"\nTraining {name} ...")
    trainer.fit(model, train_loader, val_loader)

    # Restore the best checkpoint (lowest val_loss) selected by ModelCheckpoint,
    # so the weights saved/evaluated below are the best ones — not the final
    # epoch's. Without this, EarlyStopping + checkpoint selection have no effect
    # on the model actually used for inference.
    if checkpoint.best_model_path:
        best = torch.load(checkpoint.best_model_path, map_location="cpu")
        model.load_state_dict(best["state_dict"] if "state_dict" in best else best)

    # Evaluate on the validation split.
    print(f"\nEvaluating {name} on validation set ...")
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            y_hat = model(batch["cat"], batch["num"])
            all_preds.extend(y_hat.cpu().numpy())
            all_true.extend(batch["y"].cpu().numpy())

    metrics = compute_metrics(np.array(all_true), np.array(all_preds), name.lower())
    print(
        f"  {name:<11s} MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
        f"MAPE={metrics['mape']:.2f}%  sMAPE={metrics['smape']:.2f}%"
    )

    # Save model state_dict + preprocessing metadata.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state_dict_path)
    joblib.dump(
        {
            "encoders": train_ds.encoders,
            "scalers": train_ds.scalers,
            "numeric_cols": train_ds.numeric_cols,
        },
        state_dict_path.with_suffix(".meta.joblib"),
    )

    # Upsert into the shared results JSON.
    all_results = {}
    if MODEL_RESULTS_JSON.exists():
        with open(MODEL_RESULTS_JSON) as f:
            all_results = json.load(f)
    all_results[results_key] = [metrics]
    with open(MODEL_RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nModel saved: {state_dict_path}")
    return metrics


def build_arg_parser():
    """Common CLI args shared by both DL trainers."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    return parser
