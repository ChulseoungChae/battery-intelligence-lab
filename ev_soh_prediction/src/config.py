"""Experiment config and output paths.

Experiments
-----------
daily       : 일별 집계 궤적
session     : 일별 압축 없이 세션/고해상도 궤적 (예정)
by_chg_mode : 충전 방식(slow/fast) 분리 학습
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_ROOT = ROOT / "outputs"

EXPERIMENTS = ("daily", "session", "by_chg_mode")
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


def ensure_exp_dirs(name: str, mode: str | None = None) -> Path:
    if name == "by_chg_mode" and mode is None:
        # create both mode trees
        for m in CHG_MODES:
            ensure_exp_dirs(name, m)
        return exp_dir(name)
    d = mode_dir(name, mode)
    for sub in ("traj", "models", "figs", "runs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def run_tag(L: int, H: int) -> str:
    return f"L{L}_H{H}"


def run_dir(experiment: str, L: int, H: int, mode: str | None = None) -> Path:
    """outputs/<exp>[/mode]/runs/L{L}_H{H}/"""
    ensure_exp_dirs(experiment, mode)
    d = mode_dir(experiment, mode) / "runs" / run_tag(L, H)
    for sub in ("models", "figs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d
