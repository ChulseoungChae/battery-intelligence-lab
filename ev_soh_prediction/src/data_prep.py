"""Trajectory preparation for daily / by_chg_mode / raw."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CHG_MODES,
    DATA_DIR,
    FEATURES,
    MAX_COLS,
    MEAN_COLS,
    USECOLS,
    ensure_exp_dirs,
    save_raw_traj_meta,
    traj_path,
)


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


def _aggregate_daily_chunk(
    chunk: pd.DataFrame,
    volt_stride: int = 10,
    mode_filter: str | None = None,
) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["ts"] = _pick_time(chunk)
    chunk = chunk.dropna(subset=["ts"])
    chunk["date"] = chunk["ts"].dt.floor("D")
    chunk["device_no"] = chunk["device_no"].astype(str)
    chunk["soh"] = pd.to_numeric(chunk["soh"], errors="coerce")
    chunk = chunk[chunk["soh"].between(50, 100, inclusive="both")]

    mode = chunk["chg_mode"].astype(str).str.lower().str.strip()
    if mode_filter is not None:
        chunk = chunk[mode == mode_filter]
        if chunk.empty:
            return pd.DataFrame()
        mode = chunk["chg_mode"].astype(str).str.lower().str.strip()

    for c in MAX_COLS + MEAN_COLS:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

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


def _finalize_daily(parts: list[pd.DataFrame], out: Path, label: str) -> Path:
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        raise RuntimeError(f"[{label}] no rows after aggregation")
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
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out, index=False)
    print(f"[{label}][saved] {out}  rows={len(daily)}  devices={daily['device_no'].nunique()}")
    print(daily.groupby("device_no")["soh"].agg(["min", "mean", "max", "count"]).round(3))
    return out


def _build_daily_from_files(
    data_dir: Path,
    out: Path,
    label: str,
    chunksize: int,
    mode_filter: str | None = None,
) -> Path:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in {data_dir}")

    parts: list[pd.DataFrame] = []
    for path in files:
        print(f"[{label}][read] {path.name}", flush=True)
        n = 0
        for chunk in pd.read_csv(
            path,
            usecols=lambda c: c in USECOLS,
            chunksize=chunksize,
            low_memory=False,
        ):
            if "cell_volt_list" not in chunk.columns:
                raise KeyError(f"{path.name}: cell_volt_list missing")
            agg = _aggregate_daily_chunk(chunk, mode_filter=mode_filter)
            if len(agg):
                parts.append(agg)
            n += len(chunk)
            if n % 1_000_000 < chunksize:
                print(f"  rows≈{n:,}", flush=True)
        print(f"  done {path.name}", flush=True)
    return _finalize_daily(parts, out, label)


def prepare_daily(data_dir: Path | None = None, chunksize: int = 200_000) -> Path:
    """일별 압축 궤적 → outputs/daily/traj/trajectories.csv"""
    data_dir = data_dir or DATA_DIR
    ensure_exp_dirs("daily")
    return _build_daily_from_files(data_dir, traj_path("daily"), "daily", chunksize)


def prepare_by_chg_mode(
    data_dir: Path | None = None,
    chunksize: int = 200_000,
    modes: tuple[str, ...] = CHG_MODES,
) -> dict[str, Path]:
    """원본 chg_mode로 필터 → slow/fast 각각 일별 궤적."""
    data_dir = data_dir or DATA_DIR
    ensure_exp_dirs("by_chg_mode")
    outs: dict[str, Path] = {}
    for mode in modes:
        if mode not in CHG_MODES:
            raise ValueError(f"unknown mode {mode}")
        ensure_exp_dirs("by_chg_mode", mode)
        out = traj_path("by_chg_mode", mode)
        outs[mode] = _build_daily_from_files(
            data_dir, out, f"by_chg_mode/{mode}", chunksize, mode_filter=mode
        )
    return outs


def _clean_raw_chunk(chunk: pd.DataFrame, row_stride: int) -> pd.DataFrame:
    """일별 집계 없이 행 단위 피처 정리. 메모리용으로 chunk 내 row_stride 샘플링."""
    chunk = chunk.copy()
    chunk["ts"] = _pick_time(chunk)
    chunk = chunk.dropna(subset=["ts"])
    chunk["device_no"] = chunk["device_no"].astype(str)
    chunk["soh"] = pd.to_numeric(chunk["soh"], errors="coerce")
    chunk = chunk[chunk["soh"].between(50, 100, inclusive="both")]
    if chunk.empty:
        return pd.DataFrame()

    stride = max(1, int(row_stride))
    chunk = chunk.sort_values("ts").iloc[::stride].reset_index(drop=True)

    mode = chunk["chg_mode"].astype(str).str.lower().str.strip()
    chunk["chg_mode"] = mode.map({"slow": 0.0, "fast": 1.0}).fillna(0.0)
    for c in MAX_COLS + MEAN_COLS:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
    chunk["cell_volt_mean"] = _cell_volt_mean(chunk["cell_volt_list"])

    keep = ["device_no", "ts"] + FEATURES
    return chunk[keep]


def prepare_raw(
    data_dir: Path | None = None,
    chunksize: int = 200_000,
    row_stride: int = 100,
) -> Path:
    """일별 압축 없이 로우 시계열 → outputs/raw/traj/trajectories.csv

    원본 ~600만 행을 그대로 두면 학습 윈도우가 과도해지므로,
    청크 안에서 ``row_stride`` 간격으로 샘플링한다 (기본 100 → 약 1/100).
    """
    data_dir = data_dir or DATA_DIR
    ensure_exp_dirs("raw")
    out = traj_path("raw")
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in {data_dir}")

    parts: list[pd.DataFrame] = []
    for path in files:
        print(f"[raw][read] {path.name}  row_stride={row_stride}", flush=True)
        n = 0
        for chunk in pd.read_csv(
            path,
            usecols=lambda c: c in USECOLS,
            chunksize=chunksize,
            low_memory=False,
        ):
            if "cell_volt_list" not in chunk.columns:
                raise KeyError(f"{path.name}: cell_volt_list missing")
            cleaned = _clean_raw_chunk(chunk, row_stride=row_stride)
            if len(cleaned):
                parts.append(cleaned)
            n += len(chunk)
            if n % 1_000_000 < chunksize:
                print(f"  rows≈{n:,}", flush=True)
        print(f"  done {path.name}", flush=True)

    if not parts:
        raise RuntimeError("[raw] no rows after cleaning")
    traj = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["device_no", "ts"])
        .drop_duplicates(subset=["device_no", "ts"], keep="last")
        .reset_index(drop=True)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    traj.to_csv(out, index=False)
    meta_path = save_raw_traj_meta(row_stride)
    print(f"[raw][saved] {out}  rows={len(traj)}  devices={traj['device_no'].nunique()}")
    print(f"[raw][meta] {meta_path}  row_stride={row_stride}")
    print(traj.groupby("device_no")["soh"].agg(["min", "mean", "max", "count"]).round(3))
    return out


# backward-compatible alias
prepare_session = prepare_raw

PREPARE_FN = {
    "daily": prepare_daily,
    "by_chg_mode": prepare_by_chg_mode,
    "raw": prepare_raw,
}


def prepare(
    experiment: str,
    data_dir: Path | None = None,
    chunksize: int = 200_000,
    row_stride: int = 100,
):
    fn = PREPARE_FN[experiment]
    if experiment == "raw":
        return fn(data_dir=data_dir, chunksize=chunksize, row_stride=row_stride)
    if experiment == "by_chg_mode":
        return fn(data_dir=data_dir, chunksize=chunksize)
    return fn(data_dir=data_dir, chunksize=chunksize)
