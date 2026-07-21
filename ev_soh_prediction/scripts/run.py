#!/usr/bin/env python3
"""Unified CLI for EV SOH PatchTST experiments.

Experiments (--exp)
-------------------
  daily       일별 압축 궤적
  session     일별 압축 없음 (준비 중)
  by_chg_mode 충전 방식(slow/fast) 분리

Usage
-----
  python scripts/run.py prepare --exp daily
  python scripts/run.py train   --exp daily --L 30 --H 7
  python scripts/run.py plot    --exp daily --L 30 --H 7

  python scripts/run.py prepare --exp by_chg_mode
  python scripts/run.py train   --exp by_chg_mode --mode slow --L 30 --H 7
  python scripts/run.py train   --exp by_chg_mode --mode fast --L 30 --H 7
  python scripts/run.py train   --exp by_chg_mode --mode all  --L 30 --H 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHG_MODES, EXPERIMENTS  # noqa: E402
from src.data_prep import prepare  # noqa: E402
from src.plot_lib import run_plot  # noqa: E402
from src.train_lib import run_train  # noqa: E402


def _add_exp_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--exp",
        choices=EXPERIMENTS,
        default="daily",
        help="experiment bucket under outputs/<exp>/",
    )
    p.add_argument(
        "--mode",
        choices=(*CHG_MODES, "all"),
        default=None,
        help="for by_chg_mode: slow | fast | all",
    )


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


def _modes_to_run(exp: str, mode: str | None) -> list[str | None]:
    if exp != "by_chg_mode":
        return [None]
    if mode is None or mode == "all":
        return list(CHG_MODES)
    return [mode]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="EV SOH PatchTST — prepare / train / plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="build trajectories for --exp")
    _add_exp_arg(p_prep)
    p_prep.add_argument("--data-dir", type=Path, default=None)
    p_prep.add_argument("--chunksize", type=int, default=200_000)

    p_train = sub.add_parser("train", help="train PatchTST for --exp")
    _add_exp_arg(p_train)
    _add_train_args(p_train)

    p_plot = sub.add_parser("plot", help="LOVO figures for --exp")
    _add_exp_arg(p_plot)
    p_plot.add_argument("--L", type=int, default=14)
    p_plot.add_argument("--H", type=int, default=7)
    p_plot.add_argument("--epochs", type=int, default=60)

    p_all = sub.add_parser("all", help="prepare → train → plot")
    _add_exp_arg(p_all)
    p_all.add_argument("--data-dir", type=Path, default=None)
    p_all.add_argument("--chunksize", type=int, default=200_000)
    _add_train_args(p_all)

    args = ap.parse_args(argv)
    exp = args.exp
    modes = _modes_to_run(exp, args.mode)

    if args.cmd == "prepare":
        prepare(exp, data_dir=args.data_dir, chunksize=args.chunksize)
    elif args.cmd == "train":
        for m in modes:
            run_train(exp, args, mode=m)
    elif args.cmd == "plot":
        for m in modes:
            run_plot(exp, L=args.L, H=args.H, epochs=args.epochs, mode=m)
    elif args.cmd == "all":
        prepare(exp, data_dir=args.data_dir, chunksize=args.chunksize)
        for m in modes:
            run_train(exp, args, mode=m)
            run_plot(exp, L=args.L, H=args.H, epochs=args.epochs, mode=m)
    else:
        ap.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
