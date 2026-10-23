"""랙별 충·방전 그래프.

aligned/ess_pair.parquet의 랙 전류, 전압, 전력으로 그림을 만든다.
전류·전력의 양수는 그 랙이 충전되는 방향이다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ALIGNED = Path("/data1/ess_data/preproced_data/aligned/ess_pair.parquet")
IMG_DIR = Path(__file__).resolve().parent / "img"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
COLOR_A = "#1f4e79"
COLOR_B = "#c65911"
EXAMPLE_START = "2026-03-20"
EXAMPLE_END = "2026-03-22"


def use_korean_font() -> None:
    if Path(FONT).exists():
        font_manager.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def load_aligned(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path, columns=["timestamp", "I_A", "I_B", "Vmean_A", "Vmean_B", "P_A", "P_B"]
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp")


def daily_energy(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(subset=["P_A", "P_B"]).copy()
    work["date"] = work["timestamp"].dt.floor("D")
    # 30초 간격 전력(W)을 kWh로 적산한다. 양수는 그날 그 랙이 충전된 양이다.
    daily = work.groupby("date")[["P_A", "P_B"]].sum() * 30 / 3_600_000
    daily = daily.rename(columns={"P_A": "kwh_a", "P_B": "kwh_b"})
    return daily.reset_index()


def plot_daily(daily: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.2))
    width = 0.8
    axis.bar(daily["date"], daily["kwh_a"], width=width, color=COLOR_A, label="ESS A (GW1–4)")
    axis.bar(
        daily["date"],
        daily["kwh_b"],
        width=width * 0.45,
        color=COLOR_B,
        label="ESS B (GW5–8)",
    )
    axis.axhline(0, color="#666666", linewidth=0.8)
    axis.set_ylabel("하루 순에너지 (kWh)")
    axis.set_title("랙별 하루 충·방전  (양수 = 충전, 음수 = 방전)")
    axis.legend(frameon=False, ncol=2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def break_at_gaps(frame: pd.DataFrame, columns: list[str], seconds: float = 120) -> pd.DataFrame:
    """기록이 끊긴 구간은 선으로 잇지 않는다."""
    out = frame.copy()
    gap = out["timestamp"].diff().dt.total_seconds()
    out.loc[gap > seconds, columns] = np.nan
    return out


def plot_example(frame: pd.DataFrame, path: Path) -> None:
    window = frame[
        (frame["timestamp"] >= EXAMPLE_START) & (frame["timestamp"] < EXAMPLE_END)
    ].copy()
    columns = ["I_A", "I_B", "Vmean_A", "Vmean_B"]
    window = break_at_gaps(window, columns)
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    current, voltage = axes
    current.plot(window["timestamp"], window["I_A"], color=COLOR_A, linewidth=0.9, label="ESS A 전류 합 I_A")
    current.plot(window["timestamp"], window["I_B"], color=COLOR_B, linewidth=0.9, label="ESS B 전류 합 I_B")
    current.axhline(0, color="#666666", linewidth=0.8)
    current.set_ylabel("전류 (A)")
    current.set_title("2026-03-20 ~ 03-21  랙 전류와 평균 전압")
    current.legend(frameon=False, ncol=2, loc="upper left")
    voltage.plot(window["timestamp"], window["Vmean_A"], color=COLOR_A, linewidth=0.9, label="ESS A 평균 전압")
    voltage.plot(window["timestamp"], window["Vmean_B"], color=COLOR_B, linewidth=0.9, label="ESS B 평균 전압")
    voltage.set_ylabel("평균 팩 전압 (V)")
    voltage.legend(frameon=False, ncol=2, loc="upper left")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, default=ALIGNED)
    parser.add_argument("--out", type=Path, default=IMG_DIR)
    args = parser.parse_args()
    use_korean_font()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = load_aligned(args.aligned)
    daily = daily_energy(frame)
    plot_daily(daily, args.out / "rack_daily_energy.png")
    plot_example(frame, args.out / "rack_exchange_example.png")
    opposite = ((daily["kwh_a"] > 1) & (daily["kwh_b"] < -1)) | (
        (daily["kwh_a"] < -1) & (daily["kwh_b"] > 1)
    )
    print(f"days {len(daily)} opposite {int(opposite.sum())}")
    print(f"net kWh A {daily.kwh_a.sum():.1f} B {daily.kwh_b.sum():.1f}")
    print(daily.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
