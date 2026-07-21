#!/usr/bin/env python3
"""Train multivariate PatchTST for SOH forecasting on daily ev trajectories.

Pipeline
--------
1) Load outputs/traj/daily_trajectories.csv (run build_trajectories.py first)
2) Build sliding windows: past L days of FEATURES → ΔSOH over horizon H
3) Leave-One-Vehicle-Out evaluation vs hold-last baseline
4) Fit final model on all vehicles (or --holdout-device) and save .pt/.json

Example
-------
    python scripts/build_trajectories.py
    python scripts/train_patchtst.py
    python scripts/train_patchtst.py --L 14 --H 7 --epochs 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.patchtst import PatchTST  # noqa: E402

TRAJ_PATH = ROOT / "outputs" / "traj" / "daily_trajectories.csv"
MODEL_DIR = ROOT / "outputs" / "models"

# Input channels (includes soh itself for forecasting context)
FEATURES = [
    "soh",
    "odometer",
    "chg_mode",
    "chrg_cnt",
    "chrg_cnt_q",
    "cumul_current_dischrgd",
    "cumul_pw_chrgd",
    "cumul_energy_chrgd_q",
    "op_time",
    "mod_min_temp",
    "mod_max_temp",
    "batt_coolant_inlet_temp",
    "ext_temp",
    "pack_current",
    "pack_volt",
    "acceptable_chrg_pw",
    "cell_volt_mean",
]


# Tolerance bands for "accuracy": |pred - true| <= tol  (SOH percentage points)
ACC_TOLS = (0.5, 1.0, 2.0)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def accuracy_at(y_true, y_pred, tol: float) -> float:
    """Fraction of predictions within ±tol SOH points (0~100%)."""
    err = np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))
    return float(np.mean(err <= tol) * 100.0)


def metrics_dict(y_true, y_pred, prefix: str = "") -> dict:
    """MAE + Acc@0.5/1.0/2.0 (%)."""
    out = {f"{prefix}mae": mae(y_true, y_pred)}
    for tol in ACC_TOLS:
        key = f"{prefix}acc_{str(tol).replace('.', 'p')}"  # acc_0p5, acc_1p0, ...
        out[key] = accuracy_at(y_true, y_pred, tol)
    return out


def fmt_acc_line(y_true, y_pred, label: str) -> str:
    parts = [f"{label} Acc@±{t}: {accuracy_at(y_true, y_pred, t):.1f}%" for t in ACC_TOLS]
    return " | ".join(parts) + f" | MAE {mae(y_true, y_pred):.4f}"


def fill_series(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").ffill().bfill()
    if x.isna().any():
        x = x.fillna(float(x.median()) if x.notna().any() else 0.0)
    return x.values.astype(np.float32)


def build_windows(df: pd.DataFrame, L: int, H: int) -> dict:
    """Return arrays: X (N,L,F), y_abs, last, vid, date_end."""
    xs, ys, lasts, vids, dates = [], [], [], [], []
    for vid, g in df.groupby("device_no"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < L + H:
            continue
        chans = {c: fill_series(g[c]) for c in FEATURES}
        soh = chans["soh"]
        mat = np.stack([chans[c] for c in FEATURES], axis=1)  # (T, F)
        for i in range(L - 1, len(g) - H):
            sl = slice(i - L + 1, i + 1)
            j = i + H
            xs.append(mat[sl])
            lasts.append(float(soh[i]))
            ys.append(float(soh[j]))
            vids.append(str(vid))
            dates.append(str(g.loc[i, "date"])[:10])
    if not xs:
        raise RuntimeError("No windows — check L/H vs trajectory length")
    return dict(
        X=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, np.float32),
        last=np.asarray(lasts, np.float32),
        vid=np.asarray(vids),
        date=np.asarray(dates),
    )


def standardize_fit(X: np.ndarray, y_delta: np.ndarray):
    F = X.shape[2]
    mus = X.reshape(-1, F).mean(0).astype(np.float32)
    sds = X.reshape(-1, F).std(0)
    sds = np.where(sds > 1e-8, sds, 1.0).astype(np.float32)
    mu_t = float(y_delta.mean())
    sd_t = float(y_delta.std()) or 1.0
    return mus, sds, mu_t, sd_t


def apply_std(X, y_delta, mus, sds, mu_t, sd_t):
    return (X - mus) / sds, (y_delta - mu_t) / sd_t


def train_model(
    X: np.ndarray,
    y_delta: np.ndarray,
    *,
    L: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    model_kw: dict,
    device: torch.device,
):
    mus, sds, mu_t, sd_t = standardize_fit(X, y_delta)
    Xn, yn = apply_std(X, y_delta, mus, sds, mu_t, sd_t)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PatchTST(seq_len=L, n_features=X.shape[2], **model_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(yn.astype(np.float32)))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    meta = dict(
        features=FEATURES,
        L=L,
        mus=mus.tolist(),
        sds=sds.tolist(),
        mu_t=mu_t,
        sd_t=sd_t,
        model=model_kw,
    )
    return model, meta


@torch.no_grad()
def predict(model, X, last, meta, device) -> np.ndarray:
    mus = np.asarray(meta["mus"], np.float32)
    sds = np.asarray(meta["sds"], np.float32)
    Xn = (X - mus) / sds
    model.eval()
    pred_delta = (
        model(torch.from_numpy(Xn).to(device)).cpu().numpy() * meta["sd_t"] + meta["mu_t"]
    )
    return last + pred_delta


def leave_one_vehicle_out(W: dict, args, device) -> None:
    vids = np.unique(W["vid"])
    print(f"\n[LOVO] devices={list(vids)}  L={args.L} H={args.H} windows={len(W['y'])}")
    rows = []
    for te_vid in vids:
        te = W["vid"] == te_vid
        tr = ~te
        if tr.sum() < 16:
            print(f"  skip {te_vid}: train windows {tr.sum()}")
            continue
        y_delta = W["y"][tr] - W["last"][tr]
        model, meta = train_model(
            W["X"][tr],
            y_delta,
            L=args.L,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=args.seed,
            model_kw=dict(
                patch_len=args.patch_len,
                stride=args.stride,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                d_ff=args.d_ff,
                dropout=args.dropout,
            ),
            device=device,
        )
        pred = predict(model, W["X"][te], W["last"][te], meta, device)
        y = W["y"][te]
        hl = W["last"][te]
        m = metrics_dict(y, pred)
        h = metrics_dict(y, hl, prefix="hl_")
        # Prefer Acc@±1.0 for pass/fail vs hold-last; MAE margin kept for reference
        acc_margin = m["acc_1p0"] - h["hl_acc_1p0"]
        row = dict(test_device=te_vid, n_test=int(te.sum()), **m, **h)
        row["acc_1p0_margin"] = acc_margin
        row["mae_margin"] = h["hl_mae"] - m["mae"]
        rows.append(row)
        flag = "✓" if acc_margin > 0 else "✗"
        print(
            f"  test={te_vid}  n={int(te.sum())}\n"
            f"    model     Acc@±0.5 {m['acc_0p5']:.1f}% | Acc@±1.0 {m['acc_1p0']:.1f}% | "
            f"Acc@±2.0 {m['acc_2p0']:.1f}% | MAE {m['mae']:.4f}\n"
            f"    hold-last Acc@±0.5 {h['hl_acc_0p5']:.1f}% | Acc@±1.0 {h['hl_acc_1p0']:.1f}% | "
            f"Acc@±2.0 {h['hl_acc_2p0']:.1f}% | MAE {h['hl_mae']:.4f}\n"
            f"    Acc@±1.0 margin {acc_margin:+.1f}%p {flag}"
        )
    if rows:
        summary = pd.DataFrame(rows)
        print(
            f"  mean Acc@±0.5 {summary['acc_0p5'].mean():.1f}% "
            f"(hl {summary['hl_acc_0p5'].mean():.1f}%) | "
            f"Acc@±1.0 {summary['acc_1p0'].mean():.1f}% "
            f"(hl {summary['hl_acc_1p0'].mean():.1f}%) | "
            f"Acc@±2.0 {summary['acc_2p0'].mean():.1f}% "
            f"(hl {summary['hl_acc_2p0'].mean():.1f}%)"
        )
        summary.to_csv(MODEL_DIR / "lovo_metrics.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, default=TRAJ_PATH)
    ap.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--L", type=int, default=14, help="lookback days")
    ap.add_argument("--H", type=int, default=7, help="forecast horizon days")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patch-len", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--holdout-device",
        type=str,
        default=None,
        help="If set, exclude this device from final fit (kept for eval only)",
    )
    ap.add_argument("--skip-lovo", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if not args.traj.exists():
        raise SystemExit(
            f"Trajectory not found: {args.traj}\n"
            f"Run: python scripts/build_trajectories.py"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"device={device}")

    df = pd.read_csv(args.traj)
    df["date"] = pd.to_datetime(df["date"])
    for c in FEATURES:
        if c not in df.columns:
            raise KeyError(f"Missing feature column: {c}")
    print(
        f"traj rows={len(df)} devices={df['device_no'].nunique()} "
        f"span={df['date'].min().date()}~{df['date'].max().date()}"
    )

    W = build_windows(df, args.L, args.H)
    print(f"windows={len(W['y'])}  X={W['X'].shape}")

    if not args.skip_lovo:
        leave_one_vehicle_out(W, args, device)

    # Final fit
    mask = np.ones(len(W["y"]), dtype=bool)
    if args.holdout_device:
        mask = W["vid"] != str(args.holdout_device)
        print(f"[final] holdout={args.holdout_device} train_windows={mask.sum()}")
    y_delta = W["y"][mask] - W["last"][mask]
    model, meta = train_model(
        W["X"][mask],
        y_delta,
        L=args.L,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        model_kw=dict(
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ),
        device=device,
    )
    meta["H"] = args.H
    meta["holdout_device"] = args.holdout_device

    stem = args.out_dir / f"ev_L{args.L}_H{args.H}_patchtst"
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    with open(stem.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {stem.with_suffix('.pt')}")
    print(f"[saved] {stem.with_suffix('.json')}")

    # quick train-set sanity
    pred = predict(model, W["X"][mask], W["last"][mask], meta, device)
    y_tr, hl_tr = W["y"][mask], W["last"][mask]
    print(f"[train] {fmt_acc_line(y_tr, pred, 'model')}")
    print(f"[train] {fmt_acc_line(y_tr, hl_tr, 'hold-last')}")


if __name__ == "__main__":
    main()
