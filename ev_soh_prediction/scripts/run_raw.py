#!/usr/bin/env python3
"""로우(일별 압축 없음) 궤적 실험 CLI.

원본 BMS 행을 일별로 합치지 않고 시계열로 쓰고,
기본은 lookback L=100 → 바로 다음 시점(H=1) SOH 예측.

메모리·속도를 위해 prepare 시 --row-stride(기본 100)로 행을 샘플링합니다.

Usage
-----
  python scripts/run_raw.py prepare
  python scripts/run_raw.py train   --L 100 --H 1
  python scripts/run_raw.py plot    --L 100 --H 1
  python scripts/run_raw.py all     --L 100 --H 1

일별 압축 실험은 scripts/run_daily.py 를 사용하세요.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep import prepare_raw  # noqa: E402
from src.plot_lib import run_plot  # noqa: E402
from src.train_lib import run_train  # noqa: E402

EXP = "raw"


def _add_row_stride_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--row-stride",
        type=int,
        default=None,
        help="must match prepare; default: outputs/raw/traj/meta.json",
    )


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--traj", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--L", type=int, default=100, help="lookback steps (default 100)")
    p.add_argument("--H", type=int, default=1, help="horizon steps (default 1 = next)")
    p.add_argument(
        "--win-step",
        type=int,
        default=5,
        help="sliding window stride when building samples (default 5 for raw)",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    # L=100 에 맞춤
    p.add_argument("--patch-len", type=int, default=10)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--holdout-device", type=str, default=None)
    p.add_argument("--skip-lovo", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print per-epoch train L1 loss (LOVO folds + final)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="with --verbose, print every N epochs (always prints first/last)",
    )
    _add_row_stride_arg(p)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="EV SOH PatchTST — raw (no daily aggregation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="build raw trajectories")
    p_prep.add_argument("--data-dir", type=Path, default=None)
    p_prep.add_argument("--chunksize", type=int, default=200_000)
    p_prep.add_argument(
        "--row-stride",
        type=int,
        default=100,
        help="keep every N-th row inside each read chunk (default 100)",
    )

    p_train = sub.add_parser("train", help="train PatchTST on raw traj")
    _add_train_args(p_train)

    p_plot = sub.add_parser("plot", help="LOVO figures")
    p_plot.add_argument("--L", type=int, default=100)
    p_plot.add_argument("--H", type=int, default=1)
    p_plot.add_argument("--win-step", type=int, default=5)
    p_plot.add_argument("--epochs", type=int, default=40)
    p_plot.add_argument("--patch-len", type=int, default=10)
    p_plot.add_argument("--stride", type=int, default=5)
    _add_row_stride_arg(p_plot)

    p_all = sub.add_parser("all", help="prepare → train → plot")
    p_all.add_argument("--data-dir", type=Path, default=None)
    p_all.add_argument("--chunksize", type=int, default=200_000)
    p_all.add_argument("--row-stride", type=int, default=100)
    _add_train_args(p_all)

    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        prepare_raw(
            data_dir=args.data_dir,
            chunksize=args.chunksize,
            row_stride=args.row_stride,
        )
    elif args.cmd == "train":
        run_train(EXP, args, mode=None)
    elif args.cmd == "plot":
        run_plot(
            EXP,
            L=args.L,
            H=args.H,
            epochs=args.epochs,
            win_step=args.win_step,
            patch_len=args.patch_len,
            stride=args.stride,
            row_stride=args.row_stride,
        )
    elif args.cmd == "all":
        prepare_raw(
            data_dir=args.data_dir,
            chunksize=args.chunksize,
            row_stride=args.row_stride,
        )
        run_train(EXP, args, mode=None)
        run_plot(
            EXP,
            L=args.L,
            H=args.H,
            epochs=args.epochs,
            win_step=args.win_step,
            patch_len=args.patch_len,
            stride=args.stride,
            row_stride=args.row_stride,
        )
    else:
        ap.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
