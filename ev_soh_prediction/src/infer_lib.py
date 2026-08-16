"""Online / replay inference for a trained PatchTST SOH model.

실시간처럼 한 행씩 버퍼에 넣고, lookback L이 차면 H 스텝 후 SOH를 예측한다.
오프라인 리플레이에서는 H 스텝 뒤 실제 SOH가 도착하면 오차도 같이 기록한다.
"""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (
    DATA_DIR,
    FEATURES,
    OUTPUTS_ROOT,
    resolve_raw_row_stride,
    run_tag,
    traj_path,
)
from .data_prep import _clean_raw_chunk
from .patchtst import PatchTST
from .train_lib import accuracy_at, fill_series, mae, predict


DEFAULT_DEVICE_NO = "1241225226"
# R11: raw · train_all · L=120 · H=300 · row_stride=60 · win_step=5
R11_L = 120
R11_H = 300
R11_ROW_STRIDE = 60
R11_WIN_STEP = 5
R11_EVAL = "train_all"


def ckpt_stem(
    experiment: str,
    L: int,
    H: int,
    *,
    mode: str | None = None,
    row_stride: int | None = None,
    eval_protocol: str = "lovo",
    ckpt: Path | str | None = None,
) -> Path:
    """Locate trained weights. Does not create run folders."""
    if ckpt is not None:
        p = Path(ckpt)
        return p.with_suffix("") if p.suffix in {".pt", ".json"} else p
    if experiment == "raw" and row_stride is None:
        found = _find_raw_ckpt(L, H, eval_protocol)
        if found is not None:
            return found
        tag = run_tag(L, H, row_stride=R11_ROW_STRIDE, eval_protocol=eval_protocol)
        return OUTPUTS_ROOT / "raw" / "runs" / tag / "models" / f"ev_L{L}_H{H}_patchtst"
    tag = run_tag(
        L,
        H,
        row_stride=row_stride if experiment == "raw" else None,
        eval_protocol=eval_protocol,
    )
    base = OUTPUTS_ROOT / experiment
    if experiment == "by_chg_mode" and mode:
        base = base / mode
    return base / "runs" / tag / "models" / f"ev_L{L}_H{H}_patchtst"


def _find_raw_ckpt(L: int, H: int, eval_protocol: str) -> Path | None:
    runs = OUTPUTS_ROOT / "raw" / "runs"
    if not runs.exists():
        return None
    suffix = "_trainall" if eval_protocol == "train_all" else ""
    cands: list[Path] = []
    for d in sorted(runs.glob(f"stride*_L{L}_H{H}{suffix}")):
        if eval_protocol != "train_all" and d.name.endswith("_trainall"):
            continue
        stem = d / "models" / f"ev_L{L}_H{H}_patchtst"
        if stem.with_suffix(".pt").exists() and stem.with_suffix(".json").exists():
            cands.append(stem)
    if not cands:
        return None
    for c in cands:
        if f"stride{R11_ROW_STRIDE}_" in c.parts[-3]:
            return c
    return cands[0]


def load_predictor(
    stem: Path,
    device: torch.device | None = None,
) -> tuple[PatchTST, dict, torch.device]:
    pt = stem.with_suffix(".pt")
    js = stem.with_suffix(".json")
    if not pt.exists() or not js.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n  {pt}\n  {js}\n"
            "Train first, e.g. python scripts/run_raw.py train --L 100 --H 50"
        )
    with open(js) as f:
        meta = json.load(f)
    L = int(meta["L"])
    kw = dict(meta["model"])
    n_features = len(meta.get("features") or FEATURES)
    model = PatchTST(seq_len=L, n_features=n_features, **kw)
    state = torch.load(pt, map_location="cpu")
    model.load_state_dict(state)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, meta, device


class LookbackBuffer:
    """차량 1대의 최근 L 스텝을 유지하는 실시간 버퍼."""

    def __init__(self, L: int, features: list[str] | None = None):
        self.L = int(L)
        self.features = list(features or FEATURES)
        self._times: deque = deque()
        self._soh: deque = deque()
        self._rows: deque = deque()
        self._last_feat: dict[str, float] | None = None
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self._rows)

    def _fill_row(self, feat: dict[str, float]) -> np.ndarray:
        out = []
        last = self._last_feat or {}
        filled: dict[str, float] = {}
        for c in self.features:
            v = feat.get(c, np.nan)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = np.nan
            if not np.isfinite(v):
                v = float(last[c]) if c in last and np.isfinite(last[c]) else 0.0
            filled[c] = v
            out.append(v)
        self._last_feat = filled
        return np.asarray(out, dtype=np.float32)

    def push(self, ts, soh: float, feat: dict[str, float]) -> dict | None:
        """새 샘플을 넣고, L이 차면 현재 시점 기준 윈도우를 반환한다."""
        vec = self._fill_row(feat)
        self._times.append(ts)
        self._soh.append(float(soh))
        self._rows.append(vec)
        self.n_seen += 1
        while len(self._rows) > self.L:
            self._times.popleft()
            self._soh.popleft()
            self._rows.popleft()
        if len(self._rows) < self.L:
            return None
        return {
            "X": np.stack(list(self._rows), axis=0),
            "last": float(self._soh[-1]),
            "ts": self._times[-1],
            "n_seen": self.n_seen,
        }


def _norm_device(v) -> str:
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.lstrip("0") or "0"


def list_devices(df: pd.DataFrame) -> list[str]:
    return sorted(df["device_no"].map(_norm_device).unique())


def filter_device(df: pd.DataFrame, device_no: str) -> pd.DataFrame:
    want = _norm_device(device_no)
    out = df[df["device_no"].map(_norm_device) == want].copy()
    if out.empty:
        have = ", ".join(list_devices(df))
        raise SystemExit(f"device {device_no!r} not found. available: {have}")
    return out


def load_traj_device(
    experiment: str,
    device_no: str,
    *,
    mode: str | None = None,
    traj: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    path = Path(traj) if traj else traj_path(experiment, mode)
    if not path.exists():
        script = "run_raw.py" if experiment == "raw" else "run_daily.py"
        raise SystemExit(
            f"Trajectory not found: {path}\n"
            f"Run: python scripts/{script} prepare"
            + (f" --exp {experiment}" if experiment != "raw" else "")
        )
    df = pd.read_csv(path)
    time_col = "date" if "date" in df.columns else "ts"
    df[time_col] = pd.to_datetime(df[time_col])
    g = filter_device(df, device_no).sort_values(time_col).reset_index(drop=True)
    for c in FEATURES:
        if c not in g.columns:
            raise KeyError(f"Missing feature column: {c}")
        g[c] = fill_series(g[c])
    return g, time_col


def find_vehicle_csv(data_dir: Path, device_no: str) -> Path:
    want = _norm_device(device_no)
    files = sorted(data_dir.glob("*.csv"))
    for p in files:
        stem = p.stem.lstrip("0") or "0"
        if stem == want or want in p.stem:
            return p
    raise FileNotFoundError(
        f"No CSV for device {device_no} in {data_dir}. files={[p.name for p in files]}"
    )


def iter_raw_csv_device(
    csv_path: Path,
    device_no: str,
    *,
    row_stride: int = 100,
    chunksize: int = 200_000,
):
    """원본 BMS CSV를 청크 단위로 읽어 학습과 같은 피처 행을 yield."""
    want = _norm_device(device_no)
    stride = max(1, int(row_stride))
    kept = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        if "device_no" not in chunk.columns:
            raise KeyError(f"{csv_path.name}: device_no missing")
        chunk = chunk[chunk["device_no"].map(_norm_device) == want]
        if chunk.empty:
            continue
        cleaned = _clean_raw_chunk(chunk, row_stride=1)
        if cleaned.empty:
            continue
        for _, row in cleaned.iterrows():
            if kept % stride == 0:
                yield row
            kept += 1


def _feat_from_row(row: pd.Series) -> dict[str, float]:
    return {c: float(row[c]) for c in FEATURES}


def replay_stream(
    rows: list[tuple],
    *,
    model,
    meta: dict,
    device: torch.device,
    L: int,
    H: int,
    win_step: int = 1,
    print_every: int = 200,
    sleep_s: float = 0.0,
) -> pd.DataFrame:
    """rows: (ts, soh, feat_dict) 시간순.

    예측은 시점 t에서 내고, 실제값은 t+H 샘플이 들어온 뒤에 채운다.
    """
    import time as _time

    buf = LookbackBuffer(L, meta.get("features") or FEATURES)
    pending: deque = deque()
    records: list[dict] = []
    step = max(1, int(win_step))
    n_pred = 0

    for ts, soh, feat in rows:
        soh = float(soh)
        win = buf.push(ts, soh, feat)
        while pending and pending[0]["actual_at"] == buf.n_seen:
            rec = pending.popleft()
            rec["soh_true"] = soh
            rec["abs_err"] = abs(rec["soh_pred"] - soh)
            rec["hold_last"] = rec["soh_last"]
            rec["hold_err"] = abs(rec["soh_last"] - soh)
            records.append(rec)

        if win is None:
            continue
        if (buf.n_seen - L) % step != 0:
            continue

        X = win["X"][np.newaxis, ...]
        last = np.asarray([win["last"]], dtype=np.float32)
        pred = float(predict(model, X, last, meta, device, batch_size=1)[0])
        n_pred += 1
        rec = {
            "i": n_pred,
            "ts": win["ts"],
            "n_seen": win["n_seen"],
            "soh_last": win["last"],
            "soh_pred": pred,
            "delta_pred": pred - win["last"],
            "actual_at": win["n_seen"] + int(H),
            "soh_true": np.nan,
            "abs_err": np.nan,
            "hold_last": win["last"],
            "hold_err": np.nan,
        }
        pending.append(rec)
        if print_every > 0 and (n_pred == 1 or n_pred % print_every == 0):
            print(
                f"  [{n_pred}] t={_fmt_ts(win['ts'])}  last={win['last']:.3f}  "
                f"pred(t+{H})={pred:.3f}  Δ={pred - win['last']:+.4f}",
                flush=True,
            )
        if sleep_s > 0:
            _time.sleep(sleep_s)

    leftover = list(pending)
    out = pd.DataFrame(records + leftover)
    return out


def _fmt_ts(v) -> str:
    try:
        return pd.Timestamp(v).isoformat(sep=" ", timespec="seconds")
    except Exception:
        return str(v)[:19]


def summarize_infer(df: pd.DataFrame, H: int) -> dict:
    scored = df.dropna(subset=["soh_true"])
    if scored.empty:
        print(f"[infer] predictions={len(df)}  (H={H} 뒤 실제값이 아직 없음)")
        return {}
    y = scored["soh_true"].to_numpy()
    p = scored["soh_pred"].to_numpy()
    hl = scored["soh_last"].to_numpy()
    stats = {
        "n": int(len(scored)),
        "n_pending": int(df["soh_true"].isna().sum()),
        "mae": mae(y, p),
        "hl_mae": mae(y, hl),
        "acc_0p5": accuracy_at(y, p, 0.5),
        "acc_1p0": accuracy_at(y, p, 1.0),
        "acc_2p0": accuracy_at(y, p, 2.0),
        "hl_acc_1p0": accuracy_at(y, hl, 1.0),
    }
    print(
        f"[infer] scored={stats['n']} pending={stats['n_pending']}\n"
        f"  model     Acc@±0.5 {stats['acc_0p5']:.1f}% | Acc@±1.0 {stats['acc_1p0']:.1f}% | "
        f"Acc@±2.0 {stats['acc_2p0']:.1f}% | MAE {stats['mae']:.4f}\n"
        f"  hold-last Acc@±1.0 {stats['hl_acc_1p0']:.1f}% | MAE {stats['hl_mae']:.4f}"
    )
    return stats


def plot_infer(df: pd.DataFrame, out_png: Path, title: str) -> Path | None:
    scored = df.dropna(subset=["soh_true"])
    if scored.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = pd.to_datetime(scored["ts"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ts, scored["soh_true"], label="actual (t+H)", lw=1.4)
    ax.plot(ts, scored["soh_pred"], label="pred (t+H)", lw=1.2, alpha=0.9)
    ax.plot(ts, scored["soh_last"], label="hold-last", lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("SOH")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def collect_rows_from_traj(
    g: pd.DataFrame,
    time_col: str,
    max_steps: int | None = None,
) -> list[tuple]:
    rows = []
    n = len(g) if max_steps is None else min(len(g), int(max_steps))
    for i in range(n):
        r = g.iloc[i]
        rows.append((r[time_col], float(r["soh"]), _feat_from_row(r)))
    return rows


def collect_rows_from_raw_csv(
    csv_path: Path,
    device_no: str,
    row_stride: int,
    max_steps: int | None = None,
    chunksize: int = 200_000,
) -> list[tuple]:
    rows = []
    for row in iter_raw_csv_device(
        csv_path, device_no, row_stride=row_stride, chunksize=chunksize
    ):
        rows.append((row["ts"], float(row["soh"]), _feat_from_row(row)))
        if max_steps is not None and len(rows) >= int(max_steps):
            break
    if not rows:
        raise SystemExit(f"No usable rows for device {device_no} in {csv_path}")
    return rows


def run_infer(
    *,
    experiment: str,
    device_no: str = DEFAULT_DEVICE_NO,
    L: int | None = None,
    H: int | None = None,
    mode: str | None = None,
    row_stride: int | None = None,
    eval_protocol: str = R11_EVAL,
    ckpt: Path | None = None,
    source: str = "traj",
    data_dir: Path | None = None,
    csv_path: Path | None = None,
    traj: Path | None = None,
    win_step: int = R11_WIN_STEP,
    max_steps: int | None = None,
    print_every: int = 200,
    sleep_s: float = 0.0,
    cpu: bool = False,
    no_plot: bool = False,
) -> Path:
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    if experiment == "raw":
        L = int(L if L is not None else R11_L)
        H = int(H if H is not None else R11_H)
        if row_stride is None:
            row_stride = R11_ROW_STRIDE
    else:
        L = int(L if L is not None else 30)
        H = int(H if H is not None else 1)
        row_stride = None

    stem = ckpt_stem(
        experiment,
        L,
        H,
        mode=mode,
        row_stride=row_stride,
        eval_protocol=eval_protocol,
        ckpt=ckpt,
    )
    model, meta, device = load_predictor(stem, device=device)
    L = int(meta["L"])
    H = int(meta.get("H", H))
    device_no = _norm_device(device_no)
    if experiment == "raw":
        if row_stride is None:
            row_stride = meta.get("row_stride")
        if row_stride is None:
            m = re.search(r"stride(\d+)_", str(stem))
            row_stride = int(m.group(1)) if m else resolve_raw_row_stride(None)
        row_stride = int(row_stride)

    print(
        f"[infer] exp={experiment} device={device_no} L={L} H={H} "
        f"source={source} win_step={win_step} device_hw={device}"
    )
    print(f"[infer] ckpt={stem.with_suffix('.pt')}")

    if source == "csv":
        if experiment != "raw":
            raise SystemExit("--source csv is only for --exp raw (use traj for daily)")
        data_dir = Path(data_dir) if data_dir else DATA_DIR
        csv_path = Path(csv_path) if csv_path else find_vehicle_csv(data_dir, device_no)
        print(f"[infer] csv={csv_path}  row_stride={row_stride}")
        rows = collect_rows_from_raw_csv(
            csv_path, device_no, row_stride=int(row_stride), max_steps=max_steps
        )
    else:
        g, time_col = load_traj_device(
            experiment, device_no, mode=mode, traj=traj
        )
        print(f"[infer] traj rows={len(g)}  span={g[time_col].min()} ~ {g[time_col].max()}")
        rows = collect_rows_from_traj(g, time_col, max_steps=max_steps)

    print(f"[infer] streaming {len(rows)} samples (need L={L} before first pred)")
    pred_df = replay_stream(
        rows,
        model=model,
        meta=meta,
        device=device,
        L=L,
        H=H,
        win_step=win_step,
        print_every=print_every,
        sleep_s=sleep_s,
    )
    summarize_infer(pred_df, H)

    rdir = stem.parent.parent
    fig_dir = rdir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f"infer_{device_no}"
    csv_out = fig_dir / f"{tag}.csv"
    pred_df.to_csv(csv_out, index=False)
    print(f"[saved] {csv_out}  rows={len(pred_df)}")

    if not no_plot:
        png = plot_infer(
            pred_df,
            fig_dir / f"{tag}.png",
            title=f"{experiment} device={device_no}  L={L} H={H}",
        )
        if png is not None:
            print(f"[saved] {png}")
    return csv_out
