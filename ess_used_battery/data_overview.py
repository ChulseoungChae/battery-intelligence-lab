"""원본 CSV의 줄 수, 주기, 공백과 전처리 후 운행일을 집계한다.

1초 파일은 시각 간격의 중앙값이 2초 이하인 파일이다.
30초 파일의 중간 공백은 첫 시각과 끝 시각 사이에 비어 있는 30초 칸이다.
짧은 기록은 00:00~23:59:30(2,880칸)보다 늦게 시작하거나 일찍 끝나 없는 칸이다.
분 단위 시각은 같은 분 안에서 0초, 30초로 펼친 뒤 30초 기록으로 센다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocess import OUT_DIR, RAW_DIR

FULL_DAY_SLOTS = 2880
CAMPAIGN_DAYS = 122
RACKS = {"A": [f"GW{i}" for i in range(1, 5)], "B": [f"GW{i}" for i in range(5, 9)]}


def read_timestamps(path: Path) -> pd.Series:
    with path.open("rb") as handle:
        header = handle.readline()
    separator = "\t" if b"\t" in header else ","
    column = pd.read_csv(path, sep=separator, usecols=[0], header=0, dtype=str, engine="c")
    raw = column.iloc[:, 0].astype(str).str.strip().str.lstrip("'").str.strip('"')
    timestamp = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    missing = timestamp.isna()
    if missing.any():
        timestamp.loc[missing] = pd.to_datetime(raw[missing], format="%Y-%m-%d %H:%M", errors="coerce")
    return timestamp.dropna().reset_index(drop=True)


def file_gaps(timestamp: pd.Series) -> dict:
    count = len(timestamp)
    minute_stamp = count > 1 and bool(timestamp.dt.second.eq(0).all() and timestamp.duplicated().any())
    if minute_stamp:
        offset = timestamp.groupby(timestamp).cumcount()
        timestamp = timestamp + pd.to_timedelta(offset * 30, unit="s")
    timestamp = timestamp.sort_values().reset_index(drop=True)
    empty = {"rows": count, "kind": "30s", "minute": minute_stamp, "hole": 0, "short": 0, "hole_1s": 0}
    if count < 2:
        return empty
    median_step = float(timestamp.diff().dt.total_seconds().median())
    kind = "1s" if median_step <= 2 else "30s"
    span = (timestamp.iloc[-1] - timestamp.iloc[0]).total_seconds()
    if kind == "1s":
        empty["kind"] = kind
        empty["hole_1s"] = int(round(span) + 1 - timestamp.nunique())
        return empty
    hole = span / 30 + 1 - count
    short = max(FULL_DAY_SLOTS - (span / 30 + 1), 0)
    return {
        "rows": count,
        "kind": kind,
        "minute": minute_stamp,
        "hole": int(round(hole)) if hole > 2 else 0,
        "short": int(round(short)),
        "hole_1s": 0,
    }


def scan_raw(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for number in range(1, 9):
        gateway = f"GW{number}"
        for path in sorted((raw_dir / gateway).rglob("*.csv")):
            gaps = file_gaps(read_timestamps(path))
            rows.append(
                {
                    "gw": gateway,
                    "ess": "A" if number <= 4 else "B",
                    "file": str(path.relative_to(raw_dir)),
                    "bytes": path.stat().st_size,
                    **gaps,
                }
            )
    return pd.DataFrame(rows)


def gateway_table(raw: pd.DataFrame, manifest: pd.DataFrame | None) -> pd.DataFrame:
    kept_rows = {}
    kept_days = {}
    if manifest is not None:
        kept = manifest[manifest["status"] == "kept"]
        kept_rows = kept.groupby("gw")["rows_out"].sum().to_dict()
        kept_days = kept.groupby("gw")["file_date"].nunique().to_dict()
    rows = []
    for (gateway, ess), part in raw.groupby(["gw", "ess"], sort=True):
        rows.append(
            {
                "gw": gateway,
                "ess": ess,
                "files": len(part),
                "mb": round(part["bytes"].sum() / 1_000_000),
                "rows": int(part["rows"].sum()),
                "rows_1s": int(part.loc[part["kind"] == "1s", "rows"].sum()),
                "rows_30s": int(part.loc[part["kind"] == "30s", "rows"].sum()),
                "hole_30s": int(part["hole"].sum()),
                "short_30s": int(part["short"].sum()),
                "hole_1s": int(part["hole_1s"].sum()),
                "minute_files": int(part["minute"].sum()),
                "kept_days": int(kept_days.get(gateway, 0)),
                "kept_rows": int(kept_rows.get(gateway, 0)),
            }
        )
    return pd.DataFrame(rows)


def rack_length(manifest: pd.DataFrame, gateways: list[str]) -> dict:
    kept = manifest[(manifest["status"] == "kept") & (manifest["gw"].isin(gateways))]
    by_day = kept.pivot_table(index="file_date", columns="gw", values="rows_out", aggfunc="sum")
    shared = by_day.dropna()
    span = shared.max(axis=1) - shared.min(axis=1)
    differing = span[span > 0]
    widest = span.idxmax()
    return {
        "union_days": int(by_day.shape[0]),
        "shared_days": int(shared.shape[0]),
        "differ_days": int(differing.shape[0]),
        "median_span": int(span.median()) if len(span) else 0,
        "widest_day": pd.Timestamp(widest).strftime("%m-%d"),
        "widest_rows": {gateway: int(shared.loc[widest, gateway]) for gateway in gateways},
    }


def print_report(raw: pd.DataFrame, table: pd.DataFrame, manifest: pd.DataFrame | None) -> None:
    print(f"원본 파일 {len(raw)}개, {raw['bytes'].sum()/1e9:.2f}GB, {int(raw['rows'].sum()):,}줄")
    if manifest is not None:
        dates = pd.to_datetime(manifest["file_date"]).dt.normalize().nunique()
        kept = manifest[manifest["status"] == "kept"]
        print(f"파일명 기준 날짜 {dates}일 / {CAMPAIGN_DAYS}일")
        print(f"남긴 파일 {kept['source'].nunique()}개, 남긴 줄 {int(kept['rows_out'].sum()):,}")
    print()
    for row in table.itertuples(index=False):
        print(
            f"{row.gw} {row.ess} files={row.files} {row.mb}MB rows={row.rows:,} "
            f"1s={row.rows_1s:,} 30s={row.rows_30s:,} hole={row.hole_30s:,} "
            f"short={row.short_30s:,} hole1s={row.hole_1s:,} minute={row.minute_files} "
            f"kept_days={row.kept_days} kept_rows={row.kept_rows:,}"
        )
    if manifest is None:
        return
    print()
    for ess, gateways in RACKS.items():
        stats = rack_length(manifest, gateways)
        widest = ", ".join(f"{gateway} {count:,}" for gateway, count in stats["widest_rows"].items())
        print(
            f"ESS {ess} union={stats['union_days']} shared={stats['shared_days']} "
            f"differ={stats['differ_days']} median_span={stats['median_span']} "
            f"widest={stats['widest_day']} [{widest}]"
        )
        part = table[table["ess"] == ess]
        print(f"  filled_2880 {int((part['kept_days'] * FULL_DAY_SLOTS).sum()):,} 각 GW {dict(zip(part['gw'], part['kept_days'] * FULL_DAY_SLOTS))}")
    kept = manifest[manifest["status"] == "kept"]
    any_days = pd.to_datetime(kept["file_date"]).dt.normalize().nunique()
    print(f"any_gw_days={any_days} full_grid={any_days * FULL_DAY_SLOTS:,} campaign_grid={CAMPAIGN_DAYS * FULL_DAY_SLOTS:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent / "data_overview_gw.csv")
    args = parser.parse_args()
    raw = scan_raw(args.raw)
    manifest_path = args.out / "reports" / "file_manifest.csv"
    manifest = pd.read_csv(manifest_path, parse_dates=["file_date"]) if manifest_path.exists() else None
    table = gateway_table(raw, manifest)
    table.to_csv(args.csv, index=False)
    print_report(raw, table, manifest)
    print("wrote", args.csv)


if __name__ == "__main__":
    main()
