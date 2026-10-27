"""방전마다 SOH를 계산하고 GW1~GW8 그래프를 그린다.

기준 용량은 검증 시험의 팩별 방전 전력량(kWh)이다.
SOH = 이번 방전 에너지 / 그 팩의 초기 방전 전력량 × 100.
방전 에너지는 전압이 그 방전 최저 근처에 처음 닿을 때까지의 kWh이다.
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
MIN_DROP_V = 25.0
NEAR_BOTTOM_V = 1.5
GAP_BREAK_S = 600

# 검증 시험표. 배터리팩 번호 = GW 번호.
# ESS A(GW1~4): EV용 충전용 ESS. ESS B(GW5~8): 수요관리형 ESS.
# 초기용량으로 쓰는 값은 방전 전력량이다.
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


def discharge_energy(frame: pd.DataFrame) -> pd.DataFrame:
    """전류가 이어진 방전마다, 전압 최저 근처까지의 kWh를 구한다."""
    data = frame.sort_values("timestamp").reset_index(drop=True)
    elapsed = data["timestamp"].diff().dt.total_seconds()
    broken = (
        data["flag_gap"].fillna(False)
        | (elapsed.fillna(0) > GAP_BREAK_S)
        | data["flag_current_spike"].fillna(False)
        | data["flag_pack_dropout"].fillna(False)
        | data["total_voltage"].isna()
    )
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
        drop = float(voltage.max() - voltage.min())
        if drop < MIN_DROP_V:
            continue
        hit = np.where(voltage <= voltage.min() + NEAR_BOTTOM_V)[0]
        if len(hit) == 0 or hit[0] < 10:
            continue
        end = int(hit[0])
        kilowatt_hours = float(np.sum((-current[: end + 1]) * voltage[: end + 1] * step[: end + 1]) / 3.6e6)
        rows.append(
            {
                "t_start": group["timestamp"].iloc[0],
                "t_bottom": group["timestamp"].iloc[end],
                "kwh": kilowatt_hours,
                "vmin": float(voltage.min()),
                "vmax": float(voltage.max()),
                "drop_v": drop,
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


def collect(clean_dir: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for number in range(1, 9):
        gateway = f"GW{number}"
        cycles = discharge_energy(load_gateway(gateway, clean_dir))
        initial = INITIAL_DISCHARGE_KWH[number]
        cycles.insert(0, "gw", gateway)
        cycles.insert(1, "ess", "A" if number <= 4 else "B")
        cycles.insert(2, "pack", number)
        cycles["initial_kwh"] = initial
        cycles["charge_kwh_table"] = INITIAL_CHARGE_KWH[number]
        cycles["soh"] = 100 * cycles["kwh"] / initial
        cycles["discharge_index"] = np.arange(1, len(cycles) + 1)
        pieces.append(cycles)
    return pd.concat(pieces, ignore_index=True)


def plot_soh(cycles: pd.DataFrame, path: Path) -> None:
    use_korean_font()
    figure, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
    for number, axis in enumerate(axes.ravel(), start=1):
        gateway = f"GW{number}"
        part = cycles[cycles["gw"] == gateway]
        color = COLOR_A if number <= 4 else COLOR_B
        axis.plot(part["t_start"], part["soh"], color=color, marker="o", markersize=3.5, linewidth=1)
        axis.axhline(100, color="#666666", linewidth=0.8)
        axis.set_ylim(0, 110)
        initial = INITIAL_DISCHARGE_KWH[number]
        axis.set_title(f"{gateway}   초기 방전 {initial:.1f} kWh = 100%   {len(part)}회")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#e6e6e6")
    for axis in axes[:, 0]:
        axis.set_ylabel("SOH (%)")
    for axis in axes[-1, :]:
        axis.set_xlabel("방전 시작")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=CLEAN_DIR)
    parser.add_argument("--out", type=Path, default=IMG_DIR / "soh_by_discharge.png")
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent / "soh_by_discharge.csv")
    args = parser.parse_args()
    cycles = collect(args.clean)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cycles.to_csv(args.csv, index=False)
    plot_soh(cycles, args.out)
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


if __name__ == "__main__":
    main()
