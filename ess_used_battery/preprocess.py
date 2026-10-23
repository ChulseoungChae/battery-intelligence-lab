"""사용후 배터리 2-ESS 맞충·방전 로그 전처리.

원본 CSV를 30초 격자로 맞추고, 복제·날짜 오류 파일을 제외한 뒤
센서 결측과 품질 플래그를 붙여 parquet로 저장한다.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("/data1/ess_data/battery_data")
OUT_DIR = (
    Path(__file__).resolve().parent / ".." / ".." / ".." / "ess_data" / "preproced_data"
).resolve()

GW_ESS = {f"GW{i}": ("A" if i <= 4 else "B") for i in range(1, 9)}
VOL_COLS = [f"vol_m{m}_c{c}" for m in range(1, 6) for c in range(1, 21)]
RES_COLS = [f"res_m{m}_c{c}" for m in range(1, 6) for c in range(1, 21)]
TEMP_COLS = [f"temp_m{m}_s{s}" for m in range(1, 6) for s in range(1, 5)]
COOL_COLS = ["cool_temp_1", "cool_temp_2"]
MEASURE_COLS = (
    ["total_current", "total_voltage"] + VOL_COLS + RES_COLS + TEMP_COLS + COOL_COLS
)
# 복제 판정에 쓰는 셀. 같은 랙의 병렬 스트링과 복사본을 구분한다.
SIG_CELLS = [
    "vol_m1_c1",
    "vol_m1_c10",
    "vol_m2_c5",
    "vol_m2_c15",
    "vol_m3_c3",
    "vol_m3_c12",
    "vol_m4_c8",
    "vol_m4_c18",
    "vol_m5_c1",
    "vol_m5_c11",
]
SIG_COLS = ["total_current", "total_voltage"] + SIG_CELLS

GAP_SECONDS = 600
CURRENT_SPIKE_A = 20.0
PACK_DROPOUT_V = 200.0
LOW_CELL_V = 2.5
VSUM_MISMATCH_V = 5.0
INACTIVE_ZERO_RATE = 0.8
TEMP_SENTINEL = -40.0
COOL_SENTINEL_LOW = -40.0
COOL_SENTINEL_HIGH = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    return parser.parse_args()


def detect_sep(path: Path) -> str:
    with path.open("rb") as handle:
        header = handle.readline()
    return "\t" if b"\t" in header else ","


def column_names() -> list[str]:
    # 환경 센서 중 CO2, TVOC, 내부온도, 습도는 원본이 전부 0이라 읽지 않는다.
    return (
        ["timestamp_raw", "total_current", "total_voltage"]
        + VOL_COLS
        + RES_COLS
        + TEMP_COLS
        + COOL_COLS
    )


def read_measurements(path: Path) -> pd.DataFrame:
    names = column_names()
    frame = pd.read_csv(
        path,
        sep=detect_sep(path),
        header=None,
        skiprows=1,
        usecols=range(len(names)),
        names=names,
        dtype=str,
        engine="c",
        na_filter=False,
    )
    raw_time = frame["timestamp_raw"].str.strip().str.lstrip("'").str.strip('"')
    timestamp = pd.to_datetime(raw_time, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    missing = timestamp.isna()
    if missing.any():
        timestamp.loc[missing] = pd.to_datetime(
            raw_time[missing], format="%Y-%m-%d %H:%M", errors="coerce"
        )
    frame["timestamp"] = timestamp
    frame = frame.drop(columns=["timestamp_raw"])
    for column in MEASURE_COLS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    return frame.reset_index(drop=True)


def repair_minute_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """초가 없는 시각은 같은 분 안에서 파일 순서대로 0초, 30초를 부여한다."""
    timestamp = frame["timestamp"]
    if timestamp.empty or not timestamp.duplicated().any():
        return frame
    if timestamp.dt.second.ne(0).any() or timestamp.dt.microsecond.ne(0).any():
        return frame
    offset = timestamp.groupby(timestamp).cumcount()
    frame = frame.copy()
    frame["timestamp"] = timestamp + pd.to_timedelta(offset * 30, unit="s")
    return frame


def resample_30s(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.dropna(subset=["timestamp"]).copy()
    working["bin"] = working["timestamp"].dt.floor("30s")
    spike = (
        working["total_current"].abs().gt(CURRENT_SPIKE_A).groupby(working["bin"]).any()
    )
    averaged = working.groupby("bin", as_index=False)[MEASURE_COLS].mean()
    averaged = averaged.rename(columns={"bin": "timestamp"})
    averaged["flag_current_spike"] = (
        averaged["timestamp"].map(spike).fillna(False).astype(bool)
    )
    return averaged.sort_values("timestamp").reset_index(drop=True)


def filename_date(path: Path) -> pd.Timestamp:
    return pd.Timestamp(path.stem.split("_")[-1])


def content_date_mismatch(frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty:
        return True
    return frame["timestamp"].iloc[0].normalize() != filename_date(path)


def compact_signature(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    timestamp = pd.to_datetime(frame["timestamp"])
    tod = (
        timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second
    ).to_numpy(dtype=np.int32)
    values = frame[SIG_COLS].to_numpy(dtype=np.float32)
    return tod, values


def series_match(
    tod_a: np.ndarray, sig_a: np.ndarray, tod_b: np.ndarray, sig_b: np.ndarray
) -> bool:
    common, index_a, index_b = np.intersect1d(tod_a, tod_b, return_indices=True)
    shorter = min(len(tod_a), len(tod_b))
    if shorter == 0 or len(common) < 0.95 * shorter:
        return False
    left = sig_a[index_a]
    right = sig_b[index_b]
    both_missing = np.isnan(left) & np.isnan(right)
    close = np.isclose(left, right, rtol=0, atol=1e-3) | both_missing
    return float(close.mean()) >= 0.999


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def duplicate_groups(records: list[dict]) -> list[list[int]]:
    count = len(records)
    union = UnionFind(count)
    buckets: dict[tuple, list[int]] = {}
    for index, record in enumerate(records):
        signature = record["sig"]
        current = signature[:, 0]
        voltage = signature[:, 1]
        finite_current = current[np.isfinite(current)]
        finite_voltage = voltage[np.isfinite(voltage)]
        mean_current = float(np.mean(finite_current)) if len(finite_current) else 0.0
        mean_voltage = float(np.mean(finite_voltage)) if len(finite_voltage) else 0.0
        key = (round(mean_current, 1), round(mean_voltage, 0))
        buckets.setdefault(key, []).append(index)

    for indices in buckets.values():
        for pos, left in enumerate(indices):
            for right in indices[pos + 1 :]:
                if series_match(
                    records[left]["tod"],
                    records[left]["sig"],
                    records[right]["tod"],
                    records[right]["sig"],
                ):
                    union.union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(union.find(index), []).append(index)
    return [members for members in grouped.values() if len(members) > 1]


def choose_duplicate_drops(records: list[dict], groups: list[list[int]]) -> dict[int, str]:
    """같은 GW의 복제는 가장 이른 날짜만 남긴다. GW가 섞이면 전부 제외한다."""
    drops: dict[int, str] = {}
    for members in groups:
        members = sorted(members, key=lambda index: records[index]["file_date"])
        gateways = {records[index]["gw"] for index in members}
        names = [records[index]["source"] for index in members]
        if len(gateways) > 1:
            reason = "랙이 다른데 측정값이 동일: " + ", ".join(names)
            for index in members:
                drops[index] = reason
            continue
        kept = records[members[0]]["source"]
        for index in members[1:]:
            drops[index] = f"같은 GW 복제. 유지: {kept}"
    return drops


def inactive_voltage_columns(frame: pd.DataFrame) -> list[str]:
    inactive = []
    for column in VOL_COLS:
        values = frame[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            inactive.append(column)
            continue
        zero_rate = float(np.mean(values[finite] == 0))
        if zero_rate >= INACTIVE_ZERO_RATE:
            inactive.append(column)
    return inactive


def apply_quality_rules(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    inactive = inactive_voltage_columns(out)
    inactive_res = [column.replace("vol_", "res_", 1) for column in inactive]
    if inactive:
        out[inactive] = np.nan
        out[inactive_res] = np.nan

    active_voltage = [column for column in VOL_COLS if column not in inactive]
    voltage = out[active_voltage]
    zero_cell = voltage.eq(0).any(axis=1) if active_voltage else False
    low_cell = ((voltage > 0) & (voltage < LOW_CELL_V)).any(axis=1) if active_voltage else False
    out.loc[:, active_voltage] = voltage.mask(voltage.eq(0))

    out[TEMP_COLS] = out[TEMP_COLS].mask(out[TEMP_COLS] < TEMP_SENTINEL)
    cool = out[COOL_COLS]
    out[COOL_COLS] = cool.mask((cool < COOL_SENTINEL_LOW) | (cool > COOL_SENTINEL_HIGH))
    out[RES_COLS] = out[RES_COLS].mask(out[RES_COLS].eq(0))

    pack = out["total_voltage"]
    dropout = pack.lt(PACK_DROPOUT_V)
    out["total_voltage"] = pack.mask(dropout)

    if active_voltage:
        minimum = max(int(len(active_voltage) * 0.5), 1)
        voltage_sum = out[active_voltage].sum(axis=1, min_count=minimum)
        mismatch = (voltage_sum - out["total_voltage"]).abs().gt(VSUM_MISMATCH_V)
        mismatch = mismatch.fillna(False)
    else:
        mismatch = pd.Series(False, index=out.index)

    spike = out["flag_current_spike"].fillna(False).astype(bool)
    out = out.drop(columns=["flag_current_spike"])
    flags = pd.DataFrame(
        {
            "gap_before_s": np.nan,
            "flag_gap": False,
            "flag_pack_dropout": dropout.fillna(False).to_numpy(),
            "flag_cell_zero": pd.Series(zero_cell, index=out.index).fillna(False).to_numpy(),
            "flag_low_cell": pd.Series(low_cell, index=out.index).fillna(False).to_numpy(),
            "flag_vsum_mismatch": mismatch.to_numpy(),
            "flag_current_spike": spike.to_numpy(),
        },
        index=out.index,
    )
    out = pd.concat([out, flags], axis=1)
    return out, inactive


def recompute_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """하루를 이어 붙인 뒤, 직전 샘플까지 10분을 넘으면 적산 금지 플래그를 단다."""
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    delta = ordered["timestamp"].diff().dt.total_seconds()
    ordered["gap_before_s"] = delta
    ordered["flag_gap"] = delta.gt(GAP_SECONDS).fillna(False)
    return ordered


def build_aligned(clean_dir: Path) -> pd.DataFrame:
    pieces = []
    for gateway in GW_ESS:
        path = clean_dir / f"{gateway}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path, columns=["timestamp", "total_current", "total_voltage"]
        )
        frame = frame.rename(
            columns={
                "total_current": f"I_{gateway}",
                "total_voltage": f"V_{gateway}",
            }
        )
        pieces.append(frame.set_index("timestamp"))
    if len(pieces) != 8:
        return pd.DataFrame()
    aligned = pieces[0]
    for piece in pieces[1:]:
        aligned = aligned.join(piece, how="inner")
    aligned = aligned.reset_index().sort_values("timestamp")
    group_a = [f"GW{i}" for i in range(1, 5)]
    group_b = [f"GW{i}" for i in range(5, 9)]
    aligned["I_A"] = aligned[[f"I_{gw}" for gw in group_a]].sum(axis=1, min_count=4)
    aligned["I_B"] = aligned[[f"I_{gw}" for gw in group_b]].sum(axis=1, min_count=4)
    aligned["Vmean_A"] = aligned[[f"V_{gw}" for gw in group_a]].mean(axis=1)
    aligned["Vmean_B"] = aligned[[f"V_{gw}" for gw in group_b]].mean(axis=1)
    aligned["P_A"] = sum(
        aligned[f"V_{gw}"] * aligned[f"I_{gw}"] for gw in group_a
    )
    aligned["P_B"] = sum(
        aligned[f"V_{gw}"] * aligned[f"I_{gw}"] for gw in group_b
    )
    aligned["flag_opposite_current"] = np.sign(aligned["I_A"]) * np.sign(aligned["I_B"]) < 0
    return aligned


def write_summary(path: Path, manifest: pd.DataFrame, aligned: pd.DataFrame) -> dict:
    kept = manifest[manifest["status"] == "kept"]
    summary = {
        "files_total": int(len(manifest)),
        "files_kept": int((manifest["status"] == "kept").sum()),
        "files_quarantine": int((manifest["status"] == "quarantine").sum()),
        "files_duplicate": int((manifest["status"] == "duplicate").sum()),
        "rows_kept": int(kept["rows_out"].sum()),
        "aligned_rows": int(len(aligned)),
        "by_gw": (
            kept.groupby("gw")["rows_out"].sum().astype(int).to_dict()
            if len(kept)
            else {}
        ),
    }
    if len(aligned):
        active = aligned["I_A"].abs().ge(2) | aligned["I_B"].abs().ge(2)
        opposite = aligned["flag_opposite_current"] & active
        summary["aligned_start"] = str(aligned["timestamp"].iloc[0])
        summary["aligned_end"] = str(aligned["timestamp"].iloc[-1])
        summary["aligned_opposite_ratio"] = float(opposite.sum() / max(int(active.sum()), 1))
        summary["aligned_power_corr"] = float(aligned["P_A"].corr(aligned["P_B"]))
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def preprocess(raw_dir: Path, out_dir: Path) -> dict:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"원본 디렉터리가 없습니다: {raw_dir}")
    staging = out_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    clean_dir = out_dir / "clean"
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    staging.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    records: list[dict] = []
    paths = sorted(raw_dir.rglob("*.csv"))
    print(f"원본 파일 {len(paths)}개", flush=True)

    for number, path in enumerate(paths, start=1):
        gateway = path.parts[-3]
        source = str(path.relative_to(raw_dir))
        file_date = filename_date(path)
        try:
            raw = read_measurements(path)
            raw = repair_minute_timestamps(raw)
        except Exception as exc:  # noqa: BLE001 - 파일 단위로 격리하고 계속 진행한다.
            manifest_rows.append(
                {
                    "source": source,
                    "gw": gateway,
                    "ess": GW_ESS.get(gateway, ""),
                    "file_date": file_date.date().isoformat(),
                    "status": "quarantine",
                    "reason": f"읽기 실패: {exc}",
                    "rows_raw": 0,
                    "rows_out": 0,
                    "t_start": "",
                    "t_end": "",
                }
            )
            continue
        if content_date_mismatch(raw, path):
            start = "" if raw.empty else str(raw["timestamp"].iloc[0])
            manifest_rows.append(
                {
                    "source": source,
                    "gw": gateway,
                    "ess": GW_ESS.get(gateway, ""),
                    "file_date": file_date.date().isoformat(),
                    "status": "quarantine",
                    "reason": f"파일명 날짜와 내용 시각이 다름 ({start})",
                    "rows_raw": int(len(raw)),
                    "rows_out": 0,
                    "t_start": start,
                    "t_end": "" if raw.empty else str(raw["timestamp"].iloc[-1]),
                }
            )
            continue

        resampled = resample_30s(raw)
        resampled["gw"] = gateway
        resampled["ess"] = GW_ESS[gateway]
        resampled["source_file"] = path.name
        stage_path = staging / f"{gateway}_{file_date.strftime('%Y%m%d')}.parquet"
        resampled.to_parquet(stage_path, index=False)
        tod, signature = compact_signature(resampled)
        records.append(
            {
                "source": source,
                "gw": gateway,
                "ess": GW_ESS[gateway],
                "file_date": file_date,
                "stage": stage_path,
                "rows_raw": int(len(raw)),
                "rows_out": int(len(resampled)),
                "t_start": resampled["timestamp"].iloc[0],
                "t_end": resampled["timestamp"].iloc[-1],
                "tod": tod,
                "sig": signature,
            }
        )
        if number % 50 == 0:
            print(f"읽기 {number}/{len(paths)}", flush=True)

    print("복제 파일 비교", flush=True)
    groups = duplicate_groups(records)
    drops = choose_duplicate_drops(records, groups)
    group_rows = []
    for group_id, members in enumerate(groups, start=1):
        for index in members:
            record = records[index]
            group_rows.append(
                {
                    "group": group_id,
                    "source": record["source"],
                    "gw": record["gw"],
                    "file_date": record["file_date"].date().isoformat(),
                    "action": "drop" if index in drops else "keep",
                }
            )
    pd.DataFrame(group_rows).to_csv(report_dir / "duplicate_groups.csv", index=False)

    inactive_rows = []
    kept_by_gw: dict[str, list[pd.DataFrame]] = {gw: [] for gw in GW_ESS}
    for index, record in enumerate(records):
        if index in drops:
            manifest_rows.append(
                {
                    "source": record["source"],
                    "gw": record["gw"],
                    "ess": record["ess"],
                    "file_date": record["file_date"].date().isoformat(),
                    "status": "duplicate",
                    "reason": drops[index],
                    "rows_raw": record["rows_raw"],
                    "rows_out": 0,
                    "t_start": str(record["t_start"]),
                    "t_end": str(record["t_end"]),
                }
            )
            continue
        frame = pd.read_parquet(record["stage"])
        cleaned, inactive = apply_quality_rules(frame)
        for slot in inactive:
            inactive_rows.append(
                {
                    "gw": record["gw"],
                    "file_date": record["file_date"].date().isoformat(),
                    "slot": slot,
                }
            )
        kept_by_gw[record["gw"]].append(cleaned)
        manifest_rows.append(
            {
                "source": record["source"],
                "gw": record["gw"],
                "ess": record["ess"],
                "file_date": record["file_date"].date().isoformat(),
                "status": "kept",
                "reason": "",
                "rows_raw": record["rows_raw"],
                "rows_out": int(len(cleaned)),
                "t_start": str(cleaned["timestamp"].iloc[0]),
                "t_end": str(cleaned["timestamp"].iloc[-1]),
            }
        )

    for gateway, frames in kept_by_gw.items():
        if not frames:
            continue
        merged = recompute_gaps(pd.concat(frames, ignore_index=True))
        merged.to_parquet(clean_dir / f"{gateway}.parquet", index=False)

    manifest = pd.DataFrame(manifest_rows).sort_values(["gw", "file_date", "source"])
    manifest.to_csv(report_dir / "file_manifest.csv", index=False)
    pd.DataFrame(inactive_rows).to_csv(report_dir / "inactive_slots.csv", index=False)

    print("두 ESS 시각 정합", flush=True)
    aligned = build_aligned(clean_dir)
    if len(aligned):
        aligned.to_parquet(aligned_dir / "ess_pair.parquet", index=False)
    summary = write_summary(report_dir / "summary.json", manifest, aligned)
    shutil.rmtree(staging)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    preprocess(args.raw.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
