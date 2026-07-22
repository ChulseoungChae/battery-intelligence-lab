"""Experiment config and output paths.

Two execution tracks
--------------------
daily / by_chg_mode : 일별 압축 궤적  → scripts/run_daily.py
raw                 : 일별 압축 없음(로우) → scripts/run_raw.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_ROOT = ROOT / "outputs"

# daily track
DAILY_EXPERIMENTS = ("daily", "by_chg_mode")
# raw track
RAW_EXPERIMENTS = ("raw",)
EXPERIMENTS = DAILY_EXPERIMENTS + RAW_EXPERIMENTS
CHG_MODES = ("slow", "fast")

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

ACC_TOLS = (0.5, 1.0, 2.0)

MAX_COLS = [
    "odometer",
    "chrg_cnt",
    "chrg_cnt_q",
    "cumul_current_dischrgd",
    "cumul_pw_chrgd",
    "cumul_energy_chrgd_q",
    "op_time",
]
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


def exp_dir(name: str) -> Path:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {name!r}; choose from {EXPERIMENTS}")
    return OUTPUTS_ROOT / name


def mode_dir(experiment: str, mode: str | None = None) -> Path:
    """Base dir for experiment; by_chg_mode uses .../slow or .../fast."""
    d = exp_dir(experiment)
    if experiment == "by_chg_mode":
        if mode not in CHG_MODES:
            raise ValueError(f"by_chg_mode requires mode in {CHG_MODES}, got {mode!r}")
        d = d / mode
    return d


def traj_path(name: str, mode: str | None = None) -> Path:
    return mode_dir(name, mode) / "traj" / "trajectories.csv"


def raw_traj_meta_path() -> Path:
    return traj_path("raw").parent / "meta.json"


def save_raw_traj_meta(row_stride: int) -> Path:
    p = raw_traj_meta_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"row_stride": max(1, int(row_stride))}, f, indent=2)
    return p


def read_raw_traj_meta() -> dict | None:
    p = raw_traj_meta_path()
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def resolve_raw_row_stride(row_stride: int | None = None, default: int = 100) -> int:
    if row_stride is not None:
        return max(1, int(row_stride))
    meta = read_raw_traj_meta()
    if meta and "row_stride" in meta:
        return max(1, int(meta["row_stride"]))
    return default


def ensure_exp_dirs(name: str, mode: str | None = None) -> Path:
    if name == "by_chg_mode" and mode is None:
        for m in CHG_MODES:
            ensure_exp_dirs(name, m)
        return exp_dir(name)
    d = mode_dir(name, mode)
    for sub in ("traj", "runs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def run_tag(L: int, H: int, *, row_stride: int | None = None) -> str:
    if row_stride is not None:
        return f"stride{row_stride}_L{L}_H{H}"
    return f"L{L}_H{H}"


def run_dir(
    experiment: str,
    L: int,
    H: int,
    mode: str | None = None,
    *,
    row_stride: int | None = None,
) -> Path:
    """outputs/<exp>[/mode]/runs/[stride{S}_]L{L}_H{H}/"""
    ensure_exp_dirs(experiment, mode)
    rs = row_stride if experiment == "raw" else None
    d = mode_dir(experiment, mode) / "runs" / run_tag(L, H, row_stride=rs)
    for sub in ("models", "figs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d
