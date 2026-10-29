"""충전 끝·방전 끝의 셀 전압 분포로 팩 일치도를 계산한다.

불균일률(%) = (최대 셀 − 최소 셀) / 셀 중앙값 × 100
일치도(%) = 100 − 불균일률. 유효 셀 전압이 모두 같으면 100이다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from soh_charge import (
    CLEAN_DIR,
    COLOR_A,
    COLOR_B,
    CURRENT_ON_A,
    FONT,
    GAP_BREAK_S,
    IMG_DIR,
    MIN_SPAN_V,
    NEAR_END_V,
)

VOL_COLS = [f"vol_m{m}_c{c}" for m in range(1, 6) for c in range(1, 21)]
CELL_MIN_V = 0.5
CELL_MAX_V = 4.5
OUT_IMG = IMG_DIR / "cell_consistency.png"
OUT_CSV = Path(__file__).resolve().parent / "cell_consistency.csv"


def use_korean_font() -> None:
    if Path(FONT).exists():
        font_manager.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def load_gateway(gateway: str, clean_dir: Path) -> pd.DataFrame:
    columns = [
        "timestamp",
        "total_current",
        "total_voltage",
        "flag_gap",
        "flag_current_spike",
        "flag_pack_dropout",
        *VOL_COLS,
    ]
    frame = pd.read_parquet(clean_dir / f"{gateway}.parquet", columns=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    for column in VOL_COLS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("timestamp").reset_index(drop=True)


def endpoint_indexes(frame: pd.DataFrame, charging: bool) -> list[int]:
    elapsed = frame["timestamp"].diff().dt.total_seconds()
    broken = (
        frame["flag_gap"].fillna(False)
        | (elapsed.fillna(0) > GAP_BREAK_S)
        | frame["flag_current_spike"].fillna(False)
        | frame["flag_pack_dropout"].fillna(False)
        | frame["total_voltage"].isna()
    )
    if charging:
        bulk = (frame["total_current"] > CURRENT_ON_A) & ~broken
    else:
        bulk = (frame["total_current"] < -CURRENT_ON_A) & ~broken
    session = np.zeros(len(frame), dtype=int)
    current_id = 0
    inside = False
    is_bulk = bulk.to_numpy()
    is_broken = broken.to_numpy()
    for index in range(len(frame)):
        if is_broken[index] and index > 0:
            inside = False
        if is_bulk[index]:
            if not inside:
                current_id += 1
                inside = True
            session[index] = current_id
        else:
            inside = False
    voltage = frame["total_voltage"].to_numpy()
    ends: list[int] = []
    for session_id in range(1, current_id + 1):
        indexes = np.flatnonzero(session == session_id)
        if len(indexes) < 10:
            continue
        volts = voltage[indexes]
        if float(volts.max() - volts.min()) < MIN_SPAN_V:
            continue
        if charging:
            hit = np.where(volts >= volts.max() - NEAR_END_V)[0]
        else:
            hit = np.where(volts <= volts.min() + NEAR_END_V)[0]
        if len(hit) == 0 or hit[0] < 10:
            continue
        ends.append(int(indexes[int(hit[0])]))
    return ends


def row_metrics(frame: pd.DataFrame, index: int) -> dict | None:
    values = frame.loc[index, VOL_COLS].to_numpy(dtype=float)
    active = np.isfinite(values) & (values > CELL_MIN_V) & (values < CELL_MAX_V)
    if int(active.sum()) < 80:
        return None
    volts = values[active]
    median = float(np.median(volts))
    if median <= 0:
        return None
    spread = float(volts.max() - volts.min()) / median * 100
    deviation = np.full(len(VOL_COLS), np.nan)
    deviation[active] = (values[active] - median) / median * 100
    return {
        "spread_pct": spread,
        "match_pct": max(0.0, 100 - spread),
        "n_cells": int(active.sum()),
        "sum_error_v": float(np.nansum(values[active]) - frame.at[index, "total_voltage"]),
        "deviation": deviation,
    }


def gateway_events(gateway: str, clean_dir: Path) -> pd.DataFrame:
    frame = load_gateway(gateway, clean_dir)
    rows: list[dict] = []
    number = int(gateway.replace("GW", ""))
    for charging, kind in ((True, "charge"), (False, "discharge")):
        for index in endpoint_indexes(frame, charging):
            metrics = row_metrics(frame, index)
            if metrics is None:
                continue
            deviation = metrics.pop("deviation")
            lowest = int(np.nanargmin(deviation))
            rows.append(
                {
                    "gw": gateway,
                    "ess": "A" if number <= 4 else "B",
                    "kind": kind,
                    "timestamp": frame.at[index, "timestamp"],
                    "low_cell": VOL_COLS[lowest],
                    "low_cell_pct": float(deviation[lowest]),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def plot(events: pd.DataFrame, path: Path) -> None:
    use_korean_font()
    figure, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
    for number, axis in enumerate(axes.ravel(), start=1):
        gateway = f"GW{number}"
        part = events[events["gw"] == gateway]
        color = COLOR_A if number <= 4 else COLOR_B
        charge = part[part["kind"] == "charge"]
        discharge = part[part["kind"] == "discharge"]
        axis.plot(charge["timestamp"], charge["match_pct"], color=color, marker="o", markersize=3.5, linewidth=1, label="충전 끝")
        axis.plot(
            discharge["timestamp"],
            discharge["match_pct"],
            color=color,
            marker="x",
            markersize=3.5,
            linewidth=1,
            alpha=0.75,
            label="방전 끝",
        )
        axis.axhline(100, color="#666666", linewidth=0.8)
        axis.set_ylim(60, 102)
        axis.set_title(f"{gateway}")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#e6e6e6")
        if number == 1:
            axis.legend(frameon=False, loc="lower right")
    for axis in axes[:, 0]:
        axis.set_ylabel("일치도 (%)")
    for axis in axes[-1, :]:
        axis.set_xlabel("시각")
    figure.autofmt_xdate()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def cell_label(name: str) -> str:
    module = name.split("_")[1][1:]
    cell = name.split("_")[2][1:]
    return f"{module}모듈 {cell}셀"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=CLEAN_DIR)
    parser.add_argument("--out", type=Path, default=OUT_IMG)
    parser.add_argument("--csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()
    events = pd.concat([gateway_events(f"GW{number}", args.clean) for number in range(1, 9)], ignore_index=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    events.drop(columns=[], errors="ignore").to_csv(args.csv, index=False)
    plot(events, args.out)
    for gateway, part in events.groupby("gw", sort=True):
        for kind in ("charge", "discharge"):
            group = part[part["kind"] == kind]
            march = group[group["timestamp"] < "2026-04-01"]["match_pct"]
            june = group[group["timestamp"] >= "2026-06-01"]["match_pct"]
            counts = group["low_cell"].value_counts()
            usual = counts.index[0]
            usual_pct = group.loc[group["low_cell"] == usual, "low_cell_pct"]
            print(
                gateway,
                kind,
                "n",
                len(group),
                "med",
                round(group["match_pct"].median(), 1),
                "mar",
                round(march.median(), 1) if len(march) else None,
                "jun",
                round(june.median(), 1) if len(june) else None,
                "low",
                cell_label(usual),
                f"{int(counts.iloc[0])}/{len(group)}",
                "pct",
                round(usual_pct.median(), 1),
                "sumV",
                round(group["sum_error_v"].median(), 2),
            )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
