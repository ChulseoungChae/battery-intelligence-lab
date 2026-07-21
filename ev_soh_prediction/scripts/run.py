#!/usr/bin/env python3
"""Unified CLI for EV SOH PatchTST experiments.

Experiments (--exp)
-------------------
  daily       일별 압축 궤적 (기본, 구현됨)
  session     일별 압축 없음 / 세션 단위 (준비 중)
  by_chg_mode 충전 방식(slow/fast) 분리 (준비 중)

Usage
-----
  python scripts/run.py prepare --exp daily
  python scripts/run.py train   --exp daily
  python scripts/run.py plot    --exp daily
  python scripts/run.py all     --exp daily   # prepare + train + plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EXPERIMENTS  # noqa: E402
from src.data_prep import prepare  # noqa: E402
from src.plot_lib import run_plot  # noqa: E402
from src.train_lib import run_train  # noqa: E402


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--traj", type=Path, default=None, help="override trajectory csv")
    p.add_argument("--out-dir", type=Path, default=None, help="override models dir")
    p.add_argument("--L", type=int, default=14)
    p.add_argument("--H", type=int, default=7)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch-len", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--holdout-device", type=str, default=None)
    p.add_argument("--skip-lovo", action="store_true")
    p.add_argument("--cpu", action="store_true")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="EV SOH PatchTST — prepare / train / plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--exp",
        choices=EXPERIMENTS,
        default="daily",
        help="experiment bucket under outputs/<exp>/",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="build trajectories for --exp")
    p_prep.add_argument("--data-dir", type=Path, default=None)
    p_prep.add_argument("--chunksize", type=int, default=200_000)

    p_train = sub.add_parser("train", help="train PatchTST for --exp")
    _add_train_args(p_train)

    p_plot = sub.add_parser("plot", help="LOVO figures for --exp")
    p_plot.add_argument("--L", type=int, default=14)
    p_plot.add_argument("--H", type=int, default=7)
    p_plot.add_argument("--epochs", type=int, default=60)

    p_all = sub.add_parser("all", help="prepare → train → plot")
    p_all.add_argument("--data-dir", type=Path, default=None)
    p_all.add_argument("--chunksize", type=int, default=200_000)
    _add_train_args(p_all)

    args = ap.parse_args(argv)
    exp = args.exp

    if args.cmd == "prepare":
        prepare(exp, data_dir=args.data_dir, chunksize=args.chunksize)
    elif args.cmd == "train":
        run_train(exp, args)
    elif args.cmd == "plot":
        run_plot(exp, L=args.L, H=args.H, epochs=args.epochs)
    elif args.cmd == "all":
        prepare(exp, data_dir=args.data_dir, chunksize=args.chunksize)
        run_train(exp, args)
        run_plot(exp, L=args.L, H=args.H, epochs=args.epochs)
    else:
        ap.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
