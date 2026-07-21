"""Experiment config and output paths.

Experiments
-----------
daily       : 일별 집계 궤적 (현재 기본)
session     : 일별 압축 없이 세션/고해상도 궤적 (예정)
by_chg_mode : 충전 방식(slow/fast) 분리 학습 (예정)
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_ROOT = ROOT / "outputs"

EXPERIMENTS = ("daily", "session", "by_chg_mode")

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


def traj_path(name: str) -> Path:
    return exp_dir(name) / "traj" / "trajectories.csv"


def models_dir(name: str) -> Path:
    return exp_dir(name) / "models"


def figs_dir(name: str) -> Path:
    return exp_dir(name) / "figs"


def ensure_exp_dirs(name: str) -> Path:
    d = exp_dir(name)
    for sub in ("traj", "models", "figs", "runs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def run_tag(L: int, H: int) -> str:
    return f"L{L}_H{H}"


def run_dir(experiment: str, L: int, H: int) -> Path:
    """outputs/<exp>/runs/L{L}_H{H}/ — 설정별 결과 분리."""
    ensure_exp_dirs(experiment)
    d = exp_dir(experiment) / "runs" / run_tag(L, H)
    for sub in ("models", "figs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d
