#!/usr/bin/env python3
"""Raw BMS CSV → daily multivariate trajectories for PatchTST.

Reads ./data/*.csv, aggregates per (device_no, date), computes cell_volt_mean
from cell_volt_list, and writes outputs/traj/daily_trajectories.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "outputs" / "traj" / "daily_trajectories.csv"

# Counters / levels: take max within day
MAX_COLS = [
    "odometer",
    "chrg_cnt",
    "chrg_cnt_q",
    "cumul_current_dischrgd",
    "cumul_pw_chrgd",
    "cumul_energy_chrgd_q",
    "op_time",
]
# Means within day
MEAN_COLS = [
    "mod_min_temp",
    "mod_max_temp",
    "batt_coolant_inlet_temp",
    "ext_temp",
    "pack_current",
    "pack_volt",
    "acceptable_chrg_pw",
]
USECOLS = (
    ["device_no", "msg_time", "time", "soh", "cell_volt_list", "chg_mode"]
    + MAX_COLS
    + MEAN_COLS
)


def _cell_volt_mean(series: pd.Series) -> np.ndarray:
    """Fast mean over comma-separated cell voltages."""
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


def aggregate_chunk(chunk: pd.DataFrame, volt_stride: int = 10) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["ts"] = _pick_time(chunk)
    chunk = chunk.dropna(subset=["ts"])
    chunk["date"] = chunk["ts"].dt.floor("D")
    chunk["device_no"] = chunk["device_no"].astype(str)
    chunk["soh"] = pd.to_numeric(chunk["soh"], errors="coerce")
    chunk = chunk[chunk["soh"].between(50, 100, inclusive="both")]

    for c in MAX_COLS + MEAN_COLS:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

    # chg_mode: slow=0, fast=1 → 일별 평균 = 급속 비율 (0~1)
    mode = chunk["chg_mode"].astype(str).str.lower()
    chunk["chg_mode"] = mode.map({"slow": 0.0, "fast": 1.0}).fillna(0.0)

    # cell voltages change slowly within a day — subsample for speed
    volt = chunk.iloc[:: max(1, volt_stride)][["device_no", "date", "cell_volt_list"]].copy()
    volt["cell_volt_mean"] = _cell_volt_mean(volt["cell_volt_list"])
    volt_daily = (
        volt.groupby(["device_no", "date"], as_index=False)["cell_volt_mean"].mean()
    )

    agg = {
        "soh": "median",
        "chg_mode": "mean",
        **{c: "max" for c in MAX_COLS},
        **{c: "mean" for c in MEAN_COLS},
        "ts": "count",
    }
    g = chunk.groupby(["device_no", "date"], as_index=False).agg(agg)
    g = g.rename(columns={"ts": "n_rows"})
    g = g.merge(volt_daily, on=["device_no", "date"], how="left")
    return g


def build(data_dir: Path, out_path: Path, chunksize: int = 200_000) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV in {data_dir}")

    parts: list[pd.DataFrame] = []
    for path in files:
        print(f"[read] {path.name}", flush=True)
        n = 0
        for chunk in pd.read_csv(
            path,
            usecols=lambda c: c in USECOLS,
            chunksize=chunksize,
            low_memory=False,
        ):
            missing = [c for c in USECOLS if c not in chunk.columns and c != "msg_time"]
            # msg_time optional if time exists
            if "cell_volt_list" not in chunk.columns:
                raise KeyError(f"{path.name}: cell_volt_list missing")
            parts.append(aggregate_chunk(chunk))
            n += len(chunk)
            if n % 1_000_000 < chunksize:
                print(f"  rows≈{n:,}", flush=True)
        print(f"  done {path.name}", flush=True)

    daily = pd.concat(parts, ignore_index=True)
    # Re-aggregate in case same (device, date) spanned chunks/files
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)
    print(f"[saved] {out_path}  rows={len(daily)}  devices={daily['device_no'].nunique()}")
    print(daily.groupby("device_no")["soh"].agg(["min", "mean", "max", "count"]).round(3))
    return daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()
    build(args.data_dir, args.out, args.chunksize)


if __name__ == "__main__":
    main()
