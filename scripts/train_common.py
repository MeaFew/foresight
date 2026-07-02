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

# Enable Tensor Core matmul on Ampere/Ada GPUs (4060+) — PyTorch recommends
# this whenever CUDA is available. Trades a tiny amount of precision for a
# large throughput gain on tensor-core hardware.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")
    # Our inputs are all fixed-length (SEQ_LENGTH) windows, so cudnn can safely
    # benchmark kernel variants and cache the fastest. This is a free speedup
    # for the LSTM's recurrent kernels in particular.
    torch.backends.cudnn.benchmark = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_utils import compute_metrics, time_train_val_split

from config import BATCH_SIZE, MAX_EPOCHS, MODEL_RESULTS_JSON, PATIENCE, REPORTS_DIR, VAL_DAYS


class BaseForecastModule(pl.LightningModule):
    """Shared Lightning boilerplate: MSE loss + Adam + train/val logging.

    Subclasses (LSTMForecastModule, TransformerForecastModule) only need to
    implement ``forward(cat, num)`` and set ``self.criterion`` / ``self.lr``.
    """

    def training_step(self, batch, batch_idx):
        # non_blocking overlap: the loader pins host memory, so the host→device
        # copy can run alongside the previous step's compute (keeps the GPU fed).
        cat = batch["cat"].to(self.device, non_blocking=True)
        num = batch["num"].to(self.device, non_blocking=True)
        y = batch["y"].to(self.device, non_blocking=True)
        y_hat = self(cat, num)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cat = batch["cat"].to(self.device, non_blocking=True)
        num = batch["num"].to(self.device, non_blocking=True)
        y = batch["y"].to(self.device, non_blocking=True)
        y_hat = self(cat, num)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def load_and_split(input_path: Path):
    """Read the feature CSV and apply the shared time-based train/val split.

    The validation frame is given ``SEQ_LENGTH`` extra days of history from
    the tail of the training period so that sliding-window samples can form in
    the validation set: a window predicting day *t* needs the prior
    ``SEQ_LENGTH`` days as input, and without this overlap the first
    ``SEQ_LENGTH`` validation days would yield zero samples. The training
    frame still excludes the held-out validation *targets* — only the
    context (features) is shared, which is the standard walk-forward setup.

    Returns ``(train_df, val_df, val_target_start)``. ``val_target_start`` is
    the first true-validation date; the validation dataset should only emit
    samples whose PREDICTION TARGET is on or after it (the prepended context
    rows are window input only, never prediction targets — see
    ``TimeSeriesDataset(min_target_date=...)``).
    """
    from config import SEQ_LENGTH

    df = pd.read_csv(input_path, parse_dates=["date"])
    train_df, val_df = time_train_val_split(df, VAL_DAYS)
    val_target_start = val_df["date"].min()
    # Prepend SEQ_LENGTH days of training tail as window context for val.
    context_start = val_target_start - pd.Timedelta(days=SEQ_LENGTH)
    val_context = train_df[train_df["date"] >= context_start]
    val_df = pd.concat([val_context, val_df], ignore_index=True)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} (incl. {SEQ_LENGTH}-day context)")
    print(f"Val target window starts at {val_target_start.date()} (context-only before then)")
    return train_df, val_df, val_target_start


def make_loaders(train_ds, val_ds, batch_size: int = BATCH_SIZE, num_workers: int = 4):
    """Build shuffling train / non-shuffling val DataLoaders.

    Speed notes (the old default was ``num_workers=0``, which starved the GPU):

    - ``num_workers>0`` parallelizes ``__getitem__`` across processes. On
      Windows this uses *spawn*, so each worker copies the dataset object — but
      ``TimeSeriesDataset.__init__`` already converts its data to compact
      per-series numpy arrays (~300-500 MB for the 2.3M-row training set), so a
      handful of workers fits comfortably in RAM. ``persistent_workers=True``
      keeps those workers alive for the whole run, so the copy cost is paid
      *once* rather than every epoch (the previous per-epoch spawn was the real
      reason workers felt expensive).
    - ``prefetch_factor`` keeps a small queue of pre-built batches so the GPU
      never waits on the loader between steps.
    - ``pin_memory=True`` + ``non_blocking`` transfers overlap host→device copy
      with compute.
    - ``drop_last=True`` on train gives uniform batch shapes so cudnn can use
      its benchmarked kernels on every step (avoids a slow fallback on the
      short final batch).
    """
    # num_workers=0 can't use prefetch_factor on some platforms; fall back to
    # the synchronous loader in that case (mainly for tiny CI datasets).
    if num_workers and num_workers > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
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
    train_loader, val_loader = make_loaders(
        train_ds, val_ds, args.batch_size, num_workers=args.num_workers
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=checkpoint_filename,
        dirpath=REPORTS_DIR / "checkpoints",
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")
    # bf16-mixed (not 16-mixed): the 4060 is Ada (bf16-capable), and bf16 has
    # the same dynamic range as fp32 so it avoids the overflow/underflow that
    # 16-mixed can hit on loss values — safer for MSE on log-space sales with
    # no accuracy penalty, while still engaging the Tensor Cores for speed.
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint, early_stop],
        accelerator="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        enable_progress_bar=sys.platform != "win32",
    )

    print(f"\nTraining {name} ...")
    # If --resume was passed, continue from the most recent checkpoint in the
    # checkpoint dir (glob match on the filename prefix). This restores the
    # model weights, optimizer state, epoch counter, and LR-scheduler state.
    ckpt_path = None
    if getattr(args, "resume", False):
        import glob

        ckpts = sorted(
            (REPORTS_DIR / "checkpoints").glob(f"{name.lower()}-*.ckpt"),
            key=os.path.getmtime,
        )
        if ckpts:
            ckpt_path = str(ckpts[-1])
            print(f"Resuming from {ckpt_path}")
        else:
            print("No checkpoint found to resume from; starting fresh.")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Restore the best checkpoint (lowest val_loss) selected by ModelCheckpoint,
    # so the weights saved/evaluated below are the best ones — not the final
    # epoch's. Without this, EarlyStopping + checkpoint selection have no effect
    # on the model actually used for inference.
    if checkpoint.best_model_path:
        best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best["state_dict"] if "state_dict" in best else best)

    # Evaluate on the validation split.
    print(f"\nEvaluating {name} on validation set ...")
    model.eval()
    all_preds, all_true = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in val_loader:
            cat = batch["cat"].to(device, non_blocking=True)
            num = batch["num"].to(device, non_blocking=True)
            y_hat = model(cat, num)
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
    # DataLoader workers. 4 is a good default on a multi-core CPU with ~16GB RAM
    # (each worker holds a ~300-500MB copy of the numpy-array dataset,
    # persistent across epochs). Set 0 for tiny/CI datasets.
    parser.add_argument("--num_workers", type=int, default=4)
    # Resume from the latest checkpoint in the checkpoint dir. Useful when a
    # long training run is interrupted (e.g. a timeout) — re-running with
    # --resume continues from the best saved epoch instead of restarting.
    parser.add_argument("--resume", action="store_true")
    return parser
