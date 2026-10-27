"""충전마다 SOH를 계산하고 GW1~GW8 그래프를 그린다.

SOH_k = Q_k / Q_1 × 100. Q_1은 그 GW의 첫 충전 용량이고 이때 SOH는 100이다.
Q는 충전 전류를 전압이 그 충전 최대 근처에 처음 닿을 때까지 적산한 Ah이다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

CLEAN_DIR = Path("/data1/ess_data/preproced_data/clean")
IMG_DIR = Path(__file__).resolve().parent / "img"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
COLOR_A = "#1f4e79"
COLOR_B = "#c65911"
CURRENT_ON_A = 5.0
MIN_RISE_V = 25.0
NEAR_PEAK_V = 1.5
GAP_BREAK_S = 600


def use_korean_font() -> None:
    if Path(FONT).exists():
        font_manager.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def charge_capacity(frame: pd.DataFrame) -> pd.DataFrame:
    """전류가 이어진 충전마다, 전압 최대 근처까지의 Ah를 구한다."""
    data = frame.sort_values("timestamp").reset_index(drop=True)
    elapsed = data["timestamp"].diff().dt.total_seconds()
    broken = (
        data["flag_gap"].fillna(False)
        | (elapsed.fillna(0) > GAP_BREAK_S)
        | data["flag_current_spike"].fillna(False)
        | data["flag_pack_dropout"].fillna(False)
        | data["total_voltage"].isna()
    )
    bulk = (data["total_current"] > CURRENT_ON_A) & ~broken
    session = np.zeros(len(data), dtype=int)
    current_id = 0
    inside = False
    is_bulk = bulk.to_numpy()
    is_broken = broken.to_numpy()
    for index in range(len(data)):
        if is_broken[index] and index > 0:
            inside = False
        if is_bulk[index]:
            if not inside:
                current_id += 1
                inside = True
            session[index] = current_id
        else:
            inside = False
    data["session"] = session
    data["dt"] = elapsed.fillna(30).clip(lower=0, upper=30)
    rows: list[dict] = []
    for _, group in data[data["session"] > 0].groupby("session"):
        voltage = group["total_voltage"].to_numpy()
        current = group["total_current"].to_numpy()
        step = group["dt"].to_numpy()
        rise = float(voltage.max() - voltage.min())
        if rise < MIN_RISE_V:
            continue
        hit = np.where(voltage >= voltage.max() - NEAR_PEAK_V)[0]
        if len(hit) == 0 or hit[0] < 10:
            continue
        end = int(hit[0])
        ampere_hours = float(np.sum(current[: end + 1] * step[: end + 1]) / 3600)
        rows.append(
            {
                "t_start": group["timestamp"].iloc[0],
                "t_peak": group["timestamp"].iloc[end],
                "ah": ampere_hours,
                "vmin": float(voltage.min()),
                "vmax": float(voltage.max()),
                "rise_v": rise,
            }
        )
    charges = pd.DataFrame(rows)
    if charges.empty:
        return charges
    charges["soh"] = 100 * charges["ah"] / charges["ah"].iloc[0]
    charges["charge_index"] = np.arange(1, len(charges) + 1)
    return charges


def load_gateway(gateway: str, clean_dir: Path) -> pd.DataFrame:
    columns = [
        "timestamp",
        "total_current",
        "total_voltage",
        "flag_gap",
        "flag_current_spike",
        "flag_pack_dropout",
    ]
    frame = pd.read_parquet(clean_dir / f"{gateway}.parquet", columns=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def collect(clean_dir: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for number in range(1, 9):
        gateway = f"GW{number}"
        charges = charge_capacity(load_gateway(gateway, clean_dir))
        charges.insert(0, "gw", gateway)
        charges.insert(1, "ess", "A" if number <= 4 else "B")
        pieces.append(charges)
    return pd.concat(pieces, ignore_index=True)


def plot_soh(charges: pd.DataFrame, path: Path) -> None:
    use_korean_font()
    figure, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
    for number, axis in enumerate(axes.ravel(), start=1):
        gateway = f"GW{number}"
        part = charges[charges["gw"] == gateway]
        color = COLOR_A if number <= 4 else COLOR_B
        axis.plot(part["t_start"], part["soh"], color=color, marker="o", markersize=3.5, linewidth=1)
        axis.axhline(100, color="#666666", linewidth=0.8)
        first = part["ah"].iloc[0]
        axis.set_title(f"{gateway}   첫 충전 {first:.1f}Ah = 100%   {len(part)}회")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#e6e6e6")
    for axis in axes[:, 0]:
        axis.set_ylabel("SOH (%)")
    for axis in axes[-1, :]:
        axis.set_xlabel("충전 시작")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=CLEAN_DIR)
    parser.add_argument("--out", type=Path, default=IMG_DIR / "soh_by_charge.png")
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent / "soh_by_charge.csv")
    args = parser.parse_args()
    charges = collect(args.clean)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    charges.to_csv(args.csv, index=False)
    plot_soh(charges, args.out)
    for gateway, part in charges.groupby("gw", sort=True):
        print(
            gateway,
            "n",
            len(part),
            "Q1",
            round(part["ah"].iloc[0], 1),
            "SOHmed",
            round(part["soh"].median(), 1),
            "SOHlast",
            round(part["soh"].iloc[-1], 1),
            "SOHmin",
            round(part["soh"].min(), 1),
        )


if __name__ == "__main__":
    main()
