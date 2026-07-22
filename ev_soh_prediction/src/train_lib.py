"""PatchTST training / LOVO evaluation helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ACC_TOLS, FEATURES, ensure_exp_dirs, run_dir, traj_path
from .patchtst import PatchTST


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def accuracy_at(y_true, y_pred, tol: float) -> float:
    err = np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))
    return float(np.mean(err <= tol) * 100.0)


def metrics_dict(y_true, y_pred, prefix: str = "") -> dict:
    out = {f"{prefix}mae": mae(y_true, y_pred)}
    for tol in ACC_TOLS:
        key = f"{prefix}acc_{str(tol).replace('.', 'p')}"
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


def _fmt_time(v, time_col: str) -> str:
    """daily(date) → YYYY-MM-DD, raw(ts) → full timestamp string."""
    if time_col == "date":
        return str(v)[:10]
    try:
        return pd.Timestamp(v).isoformat(sep=" ", timespec="seconds")
    except Exception:
        return str(v)


def build_windows(
    df: pd.DataFrame,
    L: int,
    H: int,
    time_col: str = "date",
    win_step: int = 1,
) -> dict:
    """슬라이딩 윈도우.

    win_step>1 이면 윈도우 시작을 win_step 간격으로만 잡아 개수를 줄인다 (raw용).
    """
    step = max(1, int(win_step))
    xs, ys, lasts, vids, dates = [], [], [], [], []
    for vid, g in df.groupby("device_no"):
        g = g.sort_values(time_col).reset_index(drop=True)
        if len(g) < L + H:
            continue
        chans = {c: fill_series(g[c]) for c in FEATURES}
        soh = chans["soh"]
        mat = np.stack([chans[c] for c in FEATURES], axis=1)
        for i in range(L - 1, len(g) - H, step):
            sl = slice(i - L + 1, i + 1)
            j = i + H
            xs.append(mat[sl])
            lasts.append(float(soh[i]))
            ys.append(float(soh[j]))
            vids.append(str(vid))
            dates.append(_fmt_time(g.loc[i, time_col], time_col))
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
def predict(model, X, last, meta, device, batch_size: int = 512) -> np.ndarray:
    mus = np.asarray(meta["mus"], np.float32)
    sds = np.asarray(meta["sds"], np.float32)
    Xn = (X - mus) / sds
    model.eval()
    outs = []
    bs = max(1, int(batch_size))
    for i in range(0, len(Xn), bs):
        xb = torch.from_numpy(Xn[i : i + bs]).to(device)
        outs.append(model(xb).cpu().numpy())
    pred_delta = np.concatenate(outs, axis=0) * meta["sd_t"] + meta["mu_t"]
    return last + pred_delta


def leave_one_vehicle_out(W: dict, args, device, metrics_csv: Path) -> pd.DataFrame | None:
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
    if not rows:
        return None
    summary = pd.DataFrame(rows)
    print(
        f"  mean Acc@±0.5 {summary['acc_0p5'].mean():.1f}% "
        f"(hl {summary['hl_acc_0p5'].mean():.1f}%) | "
        f"Acc@±1.0 {summary['acc_1p0'].mean():.1f}% "
        f"(hl {summary['hl_acc_1p0'].mean():.1f}%) | "
        f"Acc@±2.0 {summary['acc_2p0'].mean():.1f}% "
        f"(hl {summary['hl_acc_2p0'].mean():.1f}%)"
    )
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_csv, index=False)
    return summary


def run_train(experiment: str, args, mode: str | None = None) -> Path:
    mode = mode or getattr(args, "mode", None)
    if experiment == "by_chg_mode" and mode is None:
        raise SystemExit("by_chg_mode requires --mode slow|fast")

    ensure_exp_dirs(experiment, mode)
    traj = Path(args.traj) if args.traj else traj_path(experiment, mode)
    rdir = run_dir(experiment, args.L, args.H, mode=mode)
    out_dir = Path(args.out_dir) if args.out_dir else (rdir / "models")

    if not traj.exists():
        script = "run_raw.py" if experiment == "raw" else "run_daily.py"
        raise SystemExit(
            f"Trajectory not found: {traj}\n"
            f"Run: python scripts/{script} prepare"
            + (f" --exp {experiment}" if experiment != "raw" else "")
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    tag = f" mode={mode}" if mode else ""
    win_step = int(getattr(args, "win_step", 1) or 1)
    print(f"[train] exp={experiment}{tag} L={args.L} H={args.H} win_step={win_step} device={device}")
    print(f"[train] traj={traj}")
    print(f"[train] out={out_dir}")

    df = pd.read_csv(traj)
    time_col = "date" if "date" in df.columns else "ts"
    df[time_col] = pd.to_datetime(df[time_col])
    for c in FEATURES:
        if c not in df.columns:
            raise KeyError(f"Missing feature column: {c}")
    print(
        f"traj rows={len(df)} devices={df['device_no'].nunique()} "
        f"span={df[time_col].min().date()}~{df[time_col].max().date()}"
    )

    W = build_windows(df, args.L, args.H, time_col=time_col, win_step=win_step)
    print(f"windows={len(W['y'])}  X={W['X'].shape}")

    if not args.skip_lovo:
        leave_one_vehicle_out(W, args, device, out_dir / "lovo_metrics.csv")

    mask = np.ones(len(W["y"]), dtype=bool)
    if args.holdout_device:
        mask = W["vid"] != str(args.holdout_device)
        print(f"[final] holdout={args.holdout_device} train_windows={mask.sum()}")
    y_delta = W["y"][mask] - W["last"][mask]
    model_kw = dict(
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model, meta = train_model(
        W["X"][mask],
        y_delta,
        L=args.L,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        model_kw=model_kw,
        device=device,
    )
    meta["H"] = args.H
    meta["holdout_device"] = args.holdout_device
    meta["experiment"] = experiment
    meta["mode"] = mode

    stem = out_dir / f"ev_L{args.L}_H{args.H}_patchtst"
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    with open(stem.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {stem.with_suffix('.pt')}")
    print(f"[saved] {stem.with_suffix('.json')}")

    pred = predict(model, W["X"][mask], W["last"][mask], meta, device)
    y_tr, hl_tr = W["y"][mask], W["last"][mask]
    print(f"[train] {fmt_acc_line(y_tr, pred, 'model')}")
    print(f"[train] {fmt_acc_line(y_tr, hl_tr, 'hold-last')}")
    return stem.with_suffix(".pt")
