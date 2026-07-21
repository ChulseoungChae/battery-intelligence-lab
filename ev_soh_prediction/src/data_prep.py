"""Trajectory preparation for each experiment mode."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, MAX_COLS, MEAN_COLS, USECOLS, ensure_exp_dirs, traj_path


def _cell_volt_mean(series: pd.Series) -> np.ndarray:
    out = np.empty(len(series), dtype=np.float64)
    vals = series.to_numpy()
    for i, v in enumerate(vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[i] = np.nan
            continue
        arr = np.fromstring(v if isinstance(v, str) else str(v), sep=",", dtype=np.float64)
        out[i] = arr.mean() if arr.size else np.nan
    return out


def _pick_time(df: pd.DataFrame) -> pd.Series:
    if "msg_time" in df.columns:
        t = pd.to_datetime(df["msg_time"], errors="coerce")
        if t.notna().mean() > 0.5:
            return t
    return pd.to_datetime(df.get("time"), errors="coerce")


def _aggregate_daily_chunk(chunk: pd.DataFrame, volt_stride: int = 10) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["ts"] = _pick_time(chunk)
    chunk = chunk.dropna(subset=["ts"])
    chunk["date"] = chunk["ts"].dt.floor("D")
    chunk["device_no"] = chunk["device_no"].astype(str)
    chunk["soh"] = pd.to_numeric(chunk["soh"], errors="coerce")
    chunk = chunk[chunk["soh"].between(50, 100, inclusive="both")]

    for c in MAX_COLS + MEAN_COLS:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

    mode = chunk["chg_mode"].astype(str).str.lower()
    chunk["chg_mode"] = mode.map({"slow": 0.0, "fast": 1.0}).fillna(0.0)

    volt = chunk.iloc[:: max(1, volt_stride)][["device_no", "date", "cell_volt_list"]].copy()
    volt["cell_volt_mean"] = _cell_volt_mean(volt["cell_volt_list"])
    volt_daily = volt.groupby(["device_no", "date"], as_index=False)["cell_volt_mean"].mean()

    agg = {
        "soh": "median",
        "chg_mode": "mean",
        **{c: "max" for c in MAX_COLS},
        **{c: "mean" for c in MEAN_COLS},
        "ts": "count",
    }
    g = chunk.groupby(["device_no", "date"], as_index=False).agg(agg)
    g = g.rename(columns={"ts": "n_rows"})
    return g.merge(volt_daily, on=["device_no", "date"], how="left")


def prepare_daily(data_dir: Path | None = None, chunksize: int = 200_000) -> Path:
    """일별 압축 궤적 → outputs/daily/traj/trajectories.csv"""
    data_dir = data_dir or DATA_DIR
    ensure_exp_dirs("daily")
    out = traj_path("daily")

    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in {data_dir}")

    parts: list[pd.DataFrame] = []
    for path in files:
        print(f"[daily][read] {path.name}", flush=True)
        n = 0
        for chunk in pd.read_csv(
            path,
            usecols=lambda c: c in USECOLS,
            chunksize=chunksize,
            low_memory=False,
        ):
            if "cell_volt_list" not in chunk.columns:
                raise KeyError(f"{path.name}: cell_volt_list missing")
            parts.append(_aggregate_daily_chunk(chunk))
            n += len(chunk)
            if n % 1_000_000 < chunksize:
                print(f"  rows≈{n:,}", flush=True)
        print(f"  done {path.name}", flush=True)

    daily = pd.concat(parts, ignore_index=True)
    final_agg = {
        "soh": "median",
        "chg_mode": "mean",
        "cell_volt_mean": "mean",
        "n_rows": "sum",
        **{c: "max" for c in MAX_COLS},
        **{c: "mean" for c in MEAN_COLS},
    }
    daily = (
        daily.groupby(["device_no", "date"], as_index=False)
        .agg(final_agg)
        .sort_values(["device_no", "date"])
        .reset_index(drop=True)
    )
    daily.to_csv(out, index=False)
    print(f"[daily][saved] {out}  rows={len(daily)}  devices={daily['device_no'].nunique()}")
    print(daily.groupby("device_no")["soh"].agg(["min", "mean", "max", "count"]).round(3))
    return out


def prepare_session(data_dir: Path | None = None) -> Path:
    """세션/고해상도 궤적 (일별 압축 없음) — 구현 예정."""
    ensure_exp_dirs("session")
    raise NotImplementedError(
        "session 실험은 아직 미구현입니다. "
        "outputs/session/ 아래에 궤적을 넣는 prepare 로직을 추가하세요."
    )


def prepare_by_chg_mode(data_dir: Path | None = None) -> Path:
    """충전 방식(slow/fast) 분리 궤적 — 구현 예정."""
    ensure_exp_dirs("by_chg_mode")
    (ensure_exp_dirs("by_chg_mode") / "slow").mkdir(exist_ok=True)
    (ensure_exp_dirs("by_chg_mode") / "fast").mkdir(exist_ok=True)
    raise NotImplementedError(
        "by_chg_mode 실험은 아직 미구현입니다. "
        "slow/fast 각각 outputs/by_chg_mode/{slow,fast}/traj/ 에 저장하도록 구현하세요."
    )


PREPARE_FN = {
    "daily": prepare_daily,
    "session": prepare_session,
    "by_chg_mode": prepare_by_chg_mode,
}


def prepare(experiment: str, data_dir: Path | None = None, chunksize: int = 200_000) -> Path:
    fn = PREPARE_FN[experiment]
    if experiment == "daily":
        return fn(data_dir=data_dir, chunksize=chunksize)  # type: ignore[call-arg]
    return fn(data_dir=data_dir)  # type: ignore[call-arg]
