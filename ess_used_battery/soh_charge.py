"""충전·방전마다 SOH를 계산하고 GW1~GW8 그래프를 그린다.

SOH = 이번 구간 에너지 / 시험표의 같은 방향 전력량 × 100.
충전은 전류가 양수인 구간만, 방전은 전류가 음수인 구간만 사용한다.
에너지는 전압이 그 구간 끝(충전은 최대, 방전은 최저) 근처에 처음 닿을 때까지의 kWh이다.
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
MIN_SPAN_V = 25.0
NEAR_END_V = 1.5
GAP_BREAK_S = 600

# 검증 시험표. 배터리팩 번호 = GW 번호.
# ESS A(GW1~4): EV용 충전용 ESS. ESS B(GW5~8): 수요관리형 ESS.
# 방전 SOH의 분모는 방전 전력량, 충전 SOH의 분모는 충전 전력량이다.
INITIAL_DISCHARGE_KWH = {
    1: 66.915,
    2: 64.461,
    3: 66.309,
    4: 62.126,
    5: 63.532,
    6: 63.733,
    7: 63.549,
    8: 65.145,
}
INITIAL_CHARGE_KWH = {
    1: 69.644,
    2: 67.289,
    3: 68.761,
    4: 64.354,
    5: 66.290,
    6: 66.610,
    7: 65.789,
    8: 67.644,
}


def use_korean_font() -> None:
    if Path(FONT).exists():
        font_manager.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def session_energy(frame: pd.DataFrame, direction: str) -> pd.DataFrame:
    """한 방향으로 전류가 이어진 구간마다, 전압 끝 근처까지의 kWh를 구한다."""
    charging = direction == "charge"
    data = frame.sort_values("timestamp").reset_index(drop=True)
    elapsed = data["timestamp"].diff().dt.total_seconds()
    broken = (
        data["flag_gap"].fillna(False)
        | (elapsed.fillna(0) > GAP_BREAK_S)
        | data["flag_current_spike"].fillna(False)
        | data["flag_pack_dropout"].fillna(False)
        | data["total_voltage"].isna()
    )
    if charging:
        bulk = (data["total_current"] > CURRENT_ON_A) & ~broken
    else:
        bulk = (data["total_current"] < -CURRENT_ON_A) & ~broken
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
        span = float(voltage.max() - voltage.min())
        if span < MIN_SPAN_V:
            continue
        if charging:
            hit = np.where(voltage >= voltage.max() - NEAR_END_V)[0]
            signed = current
        else:
            hit = np.where(voltage <= voltage.min() + NEAR_END_V)[0]
            signed = -current
        if len(hit) == 0 or hit[0] < 10:
            continue
        end = int(hit[0])
        kilowatt_hours = float(np.sum(signed[: end + 1] * voltage[: end + 1] * step[: end + 1]) / 3.6e6)
        rows.append(
            {
                "t_start": group["timestamp"].iloc[0],
                "t_end": group["timestamp"].iloc[end],
                "kwh": kilowatt_hours,
                "vmin": float(voltage.min()),
                "vmax": float(voltage.max()),
                "span_v": span,
            }
        )
    return pd.DataFrame(rows)


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


def collect(clean_dir: Path, direction: str) -> pd.DataFrame:
    initial_table = INITIAL_CHARGE_KWH if direction == "charge" else INITIAL_DISCHARGE_KWH
    pieces: list[pd.DataFrame] = []
    for number in range(1, 9):
        gateway = f"GW{number}"
        cycles = session_energy(load_gateway(gateway, clean_dir), direction)
        initial = initial_table[number]
        cycles.insert(0, "gw", gateway)
        cycles.insert(1, "ess", "A" if number <= 4 else "B")
        cycles.insert(2, "pack", number)
        cycles["initial_kwh"] = initial
        cycles["soh"] = 100 * cycles["kwh"] / initial
        cycles["cycle_index"] = np.arange(1, len(cycles) + 1)
        pieces.append(cycles)
    return pd.concat(pieces, ignore_index=True)


def plot_soh(cycles: pd.DataFrame, path: Path, direction: str) -> None:
    use_korean_font()
    charging = direction == "charge"
    figure, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
    for number, axis in enumerate(axes.ravel(), start=1):
        gateway = f"GW{number}"
        part = cycles[cycles["gw"] == gateway]
        color = COLOR_A if number <= 4 else COLOR_B
        axis.plot(part["t_start"], part["soh"], color=color, marker="o", markersize=3.5, linewidth=1)
        axis.axhline(100, color="#666666", linewidth=0.8)
        axis.set_ylim(0, 110)
        initial = part["initial_kwh"].iloc[0]
        kind = "충전" if charging else "방전"
        axis.set_title(f"{gateway}   초기 {kind} {initial:.1f} kWh = 100%   {len(part)}회")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#e6e6e6")
    for axis in axes[:, 0]:
        axis.set_ylabel("SOH (%)")
    for axis in axes[-1, :]:
        axis.set_xlabel("충전 시작" if charging else "방전 시작")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_direction(clean_dir: Path, direction: str, image: Path, csv_path: Path) -> None:
    cycles = collect(clean_dir, direction)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    image.parent.mkdir(parents=True, exist_ok=True)
    cycles.to_csv(csv_path, index=False)
    plot_soh(cycles, image, direction)
    print(direction)
    for gateway, part in cycles.groupby("gw", sort=True):
        print(
            gateway,
            "n",
            len(part),
            "Emed",
            round(part["kwh"].median(), 1),
            "SOHmed",
            round(part["soh"].median(), 1),
            "SOHlast",
            round(part["soh"].iloc[-1], 1),
            "SOHmin",
            round(part["soh"].min(), 1),
            "SOHmax",
            round(part["soh"].max(), 1),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=CLEAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=IMG_DIR)
    parser.add_argument("--csv-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    write_direction(
        args.clean,
        "discharge",
        args.out_dir / "soh_by_discharge.png",
        args.csv_dir / "soh_by_discharge.csv",
    )
    write_direction(
        args.clean,
        "charge",
        args.out_dir / "soh_by_charge.png",
        args.csv_dir / "soh_by_charge.csv",
    )


if __name__ == "__main__":
    main()
