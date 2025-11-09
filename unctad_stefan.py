# -*- coding: utf-8 -*-
"""unctad_stefan.ipynb (adapted for local execution)

This script implements an LSTM-based nowcasting pipeline with rolling
cross-validation. It has been refactored to accept local input files and
optionally export the preprocessed dataset.
"""

import argparse
import os
import math
import time
import itertools
import random
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LSTM time-series nowcasting pipeline with rolling CV.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.environ.get("UNCTAD_INPUT", "dataset_goods.csv"),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default=os.environ.get("UNCTAD_DATE_COL", "date"),
        help="Name of the date column in the dataset.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=os.environ.get("UNCTAD_TARGET_COL", "target"),
        help="Name of the target column to forecast.",
    )
    parser.add_argument(
        "--save-preprocessed",
        type=str,
        default=os.environ.get("UNCTAD_SAVE_PREPROCESSED", ""),
        help="Optional path to save the cleaned/preprocessed dataset.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only preprocess the dataset (and optionally save it); skip model training.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=int(os.environ.get("UNCTAD_TEST_SIZE", 12)),
        help="Number of final observations to reserve for out-of-sample testing.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.environ.get("UNCTAD_RESULTS_DIR", ""),
        help="Optional directory to save plots and test predictions.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable interactive plot display (plots will be skipped unless results-dir is provided).",
    )
    return parser.parse_args()


ARGS = parse_args()
CSV_PATH = Path(ARGS.input).expanduser()
DATE_COL = ARGS.date_col
TARGET_COL = ARGS.target_col
TEST_SIZE = ARGS.test_size

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Input CSV not found: {CSV_PATH}")

if ARGS.no_plots:
    plt.switch_backend("Agg")

RESULTS_DIR = Path(ARGS.results_dir).expanduser() if ARGS.results_dir else None
if RESULTS_DIR:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[Info] Loading data from: {CSV_PATH}")
print(f"[Info] Date column: {DATE_COL} | Target column: {TARGET_COL}")

df = pd.read_csv(CSV_PATH)

# ===========================
# LSTM Time-Series Nowcasting
# ===========================
# - Rolling time-series cross-validation with expanding window
# - Hyperparameter grid search
# - Proper train/validation/test separation (no leakage)
# - Progress prints + metrics + plots
#
# Requirements: numpy, pandas, torch, scikit-learn, matplotlib
# File: updated_dataset_goods.csv with columns: ["date", <features...>, "target"]
# ===========================



# ---------------------------
# Reproducibility & Device
# ---------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
print(f"[Info] Using device: {DEVICE}")

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------



# Parse & sort by date
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

# Keep same feature set: 'date' first, 'target' last
feature_cols = [c for c in df.columns if c not in [DATE_COL, TARGET_COL]]
df = df[[DATE_COL] + feature_cols + [TARGET_COL]]

# Ensure numeric dtypes for features/target
for c in feature_cols + [TARGET_COL]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with missing target; (features with NaN will be handled by interpolation or dropped later)
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# Optional: time-based interpolation for features (comment out if you don't want any imputation)
# NOTE: This uses full-series info. For strict backtests, prefer pre-clean data.
# If you need strict "no future" imputations, implement per-fold filling limited to training-window only.
df_feat = df[[DATE_COL] + feature_cols].set_index(DATE_COL)
df_feat = df_feat.interpolate(method="time").ffill().bfill()
df[feature_cols] = df_feat.reset_index(drop=True)[feature_cols]

# Final NaN drop if any remain
nan_rows = df[feature_cols + [TARGET_COL]].isna().any(axis=1).sum()
if nan_rows > 0:
    print(f"[Warn] Dropping {nan_rows} rows with remaining NaNs after interpolation.")
    df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)

print(f"[Info] Final dataset shape: {df.shape} (rows, cols)")
print(f"[Info] Features: {len(feature_cols)} | First/Last dates: {df[DATE_COL].iloc[0].date()} → {df[DATE_COL].iloc[-1].date()}")

if ARGS.save_preprocessed:
    save_path = Path(ARGS.save_preprocessed).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Info] Saved preprocessed dataset to: {save_path.resolve()}")

# ---------------------------
# Sequence Builder (X[t-n:t] -> y[t])
# ---------------------------
def build_sequences(X_2d: np.ndarray, y_1d: np.ndarray, n_timesteps: int):
    """
    X_2d: (N, n_features)
    y_1d: (N,)
    Returns:
      Xseq: (N - n_timesteps, n_timesteps, n_features)
      yseq: (N - n_timesteps,)
    """
    N = X_2d.shape[0]
    if N <= n_timesteps:
        return np.empty((0, n_timesteps, X_2d.shape[1])), np.empty((0,))
    X_seq = np.array([X_2d[i - n_timesteps:i, :] for i in range(n_timesteps, N)])
    y_seq = y_1d[n_timesteps:]
    return X_seq, y_seq

# ---------------------------
# PyTorch LSTM Regressor
# ---------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, (hn, cn) = self.lstm(x)
        last_hidden = out[:, -1, :]  # (B, hidden)
        return self.fc(last_hidden).squeeze(-1)

@dataclass
class TrainConfig:
    n_timesteps: int
    hidden_size: int
    num_layers: int
    dropout: float
    lr: float
    batch_size: int
    epochs: int
    weight_decay: float = 0.0
    grad_clip: float = 1.0  # clip to avoid exploding gradients

# ---------------------------
# Helpers: metrics, training
# ---------------------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE (avoid division by zero)
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2}


def finalize_plot(fig_name: str):
    """
    Save or display the current matplotlib figure based on CLI arguments.
    """
    saved = False
    if RESULTS_DIR:
        filepath = RESULTS_DIR / f"{fig_name}.png"
        plt.savefig(filepath, bbox_inches="tight")
        print(f"[Info] Saved plot: {filepath.resolve()}")
        saved = True
    if ARGS.no_plots:
        plt.close()
    elif saved:
        plt.close()
    else:
        plt.show()

def train_one_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, cfg: TrainConfig, verbose=False):
    model = LSTMRegressor(
        input_dim=X_train_seq.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
    val_ds   = TensorDataset(torch.tensor(X_val_seq,   dtype=torch.float32), torch.tensor(y_val_seq,   dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if len(val_losses) else np.nan

        if verbose and (epoch % max(1, cfg.epochs // 5) == 0 or epoch == 1 or epoch == cfg.epochs):
            print(f"    [Epoch {epoch:3d}/{cfg.epochs}] Train MSE: {np.mean(train_losses):.6f} | Val MSE: {val_loss:.6f}")

        # track best
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def predict_model(model, X_seq):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
        preds = model(X_t).cpu().numpy()
    return preds

# ---------------------------
# CV Splits (expanding window)
# ---------------------------
def make_time_splits(n_samples, n_splits, fold_size, min_train_size):
    """
    Returns list of (train_end_index_exclusive, val_end_index_exclusive)
    over raw index time steps (NOT sequence indices).
    """
    splits = []
    train_end = min_train_size
    for k in range(n_splits):
        val_end = train_end + fold_size
        if val_end > n_samples:
            break
        splits.append((train_end, val_end))
        train_end = val_end  # expanding window
    return splits

# ---------------------------
# GRID SEARCH with Rolling CV
# ---------------------------
if ARGS.skip_training:
    print("[Info] --skip-training flag provided; skipping model training/evaluation.")
else:
    N = len(df)
    assert N > TEST_SIZE + 36, "Not enough data for CV + Test. Increase data or reduce TEST_SIZE."

    # Prepare arrays
    dates_all = df[DATE_COL].values
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df[TARGET_COL].values.astype(np.float32)

    # We will scale features per fold (fit on train only) to avoid leakage.
    # Define grid (keep modest first; expand later)
    param_grid = {
        "n_timesteps": [6, 12],
        "hidden_size": [32, 64],
        "num_layers":  [1, 2],
        "dropout":     [0.0, 0.2],
        "lr":          [1e-3],
        "batch_size":  [32],
        "epochs":      [60],  # increase for more thorough training
    }

    # CV config within training+validation region (exclude last TEST_SIZE for final OOS)
    N_trainval = N - TEST_SIZE
    CV_SPLITS = 4
    FOLD_SIZE = max(6, (N_trainval // (CV_SPLITS + 2)))  # make folds reasonable
    MIN_TRAIN = max(36, FOLD_SIZE)  # at least ~3 years if monthly (adjust)

    print(f"[Info] N={N}, Train+Val={N_trainval}, Test={TEST_SIZE}")
    print(f"[Info] CV: n_splits={CV_SPLITS}, fold_size={FOLD_SIZE}, min_train={MIN_TRAIN}")

    grid_combos = list(itertools.product(
        param_grid["n_timesteps"],
        param_grid["hidden_size"],
        param_grid["num_layers"],
        param_grid["dropout"],
        param_grid["lr"],
        param_grid["batch_size"],
        param_grid["epochs"],
    ))
    print(f"[Info] Grid combinations: {len(grid_combos)}")

    best_combo = None
    best_cv_mae = float("inf")
    best_cv_report = None

    combo_num = 0
    for (n_steps, hidden, layers, drop, lr, bs, epochs) in grid_combos:
        combo_num += 1
        print(f"\n=== Grid {combo_num}/{len(grid_combos)} ===")
        print(f"Config: n_steps={n_steps}, hidden={hidden}, layers={layers}, dropout={drop}, lr={lr}, batch={bs}, epochs={epochs}")

        # Build CV splits over raw indices (0..N_trainval-1)
        splits = make_time_splits(
            N_trainval,
            n_splits=CV_SPLITS,
            fold_size=FOLD_SIZE,
            min_train_size=max(MIN_TRAIN, n_steps + 12),
        )
        if len(splits) == 0:
            print("  [Skip] Not enough data for this n_timesteps/min_train.")
            continue

        cv_metrics = []
        fold_id = 0
        t0 = time.time()
        for train_end, val_end in splits:
            fold_id += 1
            print(f"  - Fold {fold_id}: train=[0..{train_end-1}] val=[{train_end}..{val_end-1}]")

            # Fit scaler ONLY on training window
            scaler = StandardScaler()
            scaler.fit(X_all[:train_end])
            X_scaled = scaler.transform(X_all[:val_end])  # transform train+val region for sequence building
            y_sub = y_all[:val_end]

            # Build sequences with this n_steps using the transformed subset
            X_seq, y_seq = build_sequences(X_scaled, y_sub, n_timesteps=n_steps)
            # Map raw index to sequence index: seq_idx corresponds to raw_idx = n_steps + seq_idx
            seq_train_end = train_end - n_steps
            seq_val_end   = val_end   - n_steps
            if seq_train_end <= 0 or seq_val_end <= seq_train_end:
                print("    [Skip fold] Not enough sequence samples. Increase data or reduce n_timesteps.")
                continue

            X_tr, y_tr = X_seq[:seq_train_end], y_seq[:seq_train_end]
            X_va, y_va = X_seq[seq_train_end:seq_val_end], y_seq[seq_train_end:seq_val_end]

            cfg = TrainConfig(
                n_timesteps=n_steps, hidden_size=hidden, num_layers=layers,
                dropout=drop, lr=lr, batch_size=bs, epochs=epochs
            )

            seed_everything(42)  # stable runs per fold
            model = train_one_model(X_tr, y_tr, X_va, y_va, cfg, verbose=False)
            y_pred_va = predict_model(model, X_va)

            metrics = compute_metrics(y_va, y_pred_va)
            cv_metrics.append(metrics)
            print(f"    Fold {fold_id} MAE: {metrics['MAE']:.6f} | RMSE: {metrics['RMSE']:.6f} | MSE: {metrics['MSE']:.6f}")

        if len(cv_metrics) == 0:
            print("  [Skip combo] No valid folds.")
            continue

        # Average CV metrics
        avg = {k: float(np.mean([m[k] for m in cv_metrics])) for k in cv_metrics[0].keys()}
        dt = time.time() - t0
        print(f"  -> CV Avg: MAE={avg['MAE']:.6f}, RMSE={avg['RMSE']:.6f}, MSE={avg['MSE']:.6f}, MAPE%={avg['MAPE_%']:.2f}, R2={avg['R2']:.4f} | {dt:.1f}s")

        # Track best by MAE (or RMSE)
        if avg["MAE"] < best_cv_mae:
            best_cv_mae = avg["MAE"]
            best_combo = (n_steps, hidden, layers, drop, lr, bs, epochs)
            best_cv_report = avg

    print("\n==============================")
    print("[Result] Best CV configuration")
    if best_combo is None:
        raise RuntimeError("No valid configuration found in grid search. Adjust grid or data.")
    print(f"Best config: n_steps={best_combo[0]}, hidden={best_combo[1]}, layers={best_combo[2]}, dropout={best_combo[3]}, lr={best_combo[4]}, batch={best_combo[5]}, epochs={best_combo[6]}")
    print(f"Best CV Avg: MAE={best_cv_report['MAE']:.6f} | RMSE={best_cv_report['RMSE']:.6f} | MSE={best_cv_report['MSE']:.6f} | MAPE%={best_cv_report['MAPE_%']:.2f} | R2={best_cv_report['R2']:.4f}")
    print("==============================\n")

    # ---------------------------
    # Retrain on Full Train+Val and Test on Held-Out OOS
    # ---------------------------
    n_steps, hidden, layers, drop, lr, bs, epochs = best_combo

    # Split raw data into TrainVal [0..N_trainval-1] and Test [N_trainval..N-1]
    dates_trainval = dates_all[:N_trainval]
    dates_test = dates_all[N_trainval:]

    # Scale on TrainVal only
    scaler = StandardScaler()
    scaler.fit(X_all[:N_trainval])
    X_trainval_scaled = scaler.transform(X_all[:N_trainval])
    X_test_scaled     = scaler.transform(X_all[N_trainval:])

    # Build sequences for TrainVal and for Test (note: for Test we must have at least n_steps context;
    # we concatenate the tail of TrainVal to provide context)
    X_concat_for_test = np.vstack([X_all[:N_trainval], X_all[N_trainval:]])  # raw for safety
    X_concat_scaled   = np.vstack([X_trainval_scaled, X_test_scaled])

    # TrainVal sequences
    X_seq_trv, y_seq_trv = build_sequences(X_trainval_scaled, y_all[:N_trainval], n_timesteps=n_steps)

    # Test sequences:
    X_seq_all, y_seq_all = build_sequences(X_concat_scaled, y_all, n_timesteps=n_steps)
    seq_start_test = N_trainval - n_steps
    seq_end_test   = len(y_seq_all)  # exclusive
    if seq_start_test < 0:
        raise RuntimeError("Not enough context before test to build sequences. Reduce n_timesteps or increase training size.")
    X_seq_te = X_seq_all[seq_start_test:seq_end_test]
    y_seq_te = y_seq_all[seq_start_test:seq_end_test]
    dates_seq_all = dates_all[n_steps:]
    dates_seq_te  = dates_seq_all[seq_start_test:seq_end_test]

    # Make a small validation split from the tail of TrainVal sequences for final training
    val_share = 0.15
    val_len = max(1, int(len(X_seq_trv) * val_share))
    X_tr_final, y_tr_final = X_seq_trv[:-val_len], y_seq_trv[:-val_len]
    X_va_final, y_va_final = X_seq_trv[-val_len:],  y_seq_trv[-val_len:]

    cfg_best = TrainConfig(
        n_timesteps=n_steps, hidden_size=hidden, num_layers=layers,
        dropout=drop, lr=lr, batch_size=bs, epochs=epochs
    )

    print("[Train] Fitting best model on Train+Val ...")
    seed_everything(123)
    model_best = train_one_model(X_tr_final, y_tr_final, X_va_final, y_va_final, cfg_best, verbose=True)

    print("[Predict] Generating OOS test predictions ...")
    y_pred_test = predict_model(model_best, X_seq_te)

    # ---------------------------
    # Metrics & Plots on Test
    # ---------------------------
    metrics_test = compute_metrics(y_seq_te, y_pred_test)
    print("\n[OOS Test Metrics]")
    for k, v in metrics_test.items():
        if k == "MAPE_%":
            print(f"  {k:7s}: {v:8.3f}")
        else:
            print(f"  {k:7s}: {v:10.6f}")

    # Build a results DataFrame for test window
    test_res = pd.DataFrame({
        "date": pd.to_datetime(dates_seq_te),
        "actual": y_seq_te,
        "pred": y_pred_test,
    })
    print("\n[Preview of OOS predictions]")
    print(test_res.tail())

    if RESULTS_DIR:
        pred_path = RESULTS_DIR / "oos_predictions.csv"
        test_res.to_csv(pred_path, index=False)
        print(f"[Info] Saved OOS predictions to: {pred_path.resolve()}")

    # ---- Plots ----
    plt.figure()
    plt.plot(test_res["date"], test_res["actual"], label="Actual")
    plt.plot(test_res["date"], test_res["pred"],   label="Predicted")
    plt.title("Out-of-Sample: Actual vs Predicted")
    plt.xlabel("Date"); plt.ylabel("Target"); plt.legend(); plt.xticks(rotation=45)
    plt.tight_layout()
    finalize_plot("oos_actual_vs_pred")

    plt.figure()
    plt.scatter(test_res["actual"], test_res["pred"], alpha=0.7)
    min_v = min(test_res["actual"].min(), test_res["pred"].min())
    max_v = max(test_res["actual"].max(), test_res["pred"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.title("OOS: Predicted vs Actual (scatter)")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.tight_layout()
    finalize_plot("oos_scatter")

    plt.figure()
    resid = test_res["pred"] - test_res["actual"]
    plt.plot(test_res["date"], resid)
    plt.axhline(0, linestyle="--")
    plt.title("OOS Residuals Over Time")
    plt.xlabel("Date"); plt.ylabel("Prediction Error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    finalize_plot("oos_residuals")

    plt.figure()
    names = ["MSE", "RMSE", "MAE", "MAPE_%"]
    vals = [metrics_test[k] for k in names]
    plt.bar(names, vals)
    plt.title("OOS Metrics")
    plt.tight_layout()
    finalize_plot("oos_metrics_bar")

    # Optional: save model state
    # if RESULTS_DIR:
    #     torch.save(model_best.state_dict(), RESULTS_DIR / "best_lstm_model.pt")

    print("\n[Done]")