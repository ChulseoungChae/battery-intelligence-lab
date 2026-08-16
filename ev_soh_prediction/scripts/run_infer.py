#!/usr/bin/env python3
"""학습된 PatchTST로 차량 1대 SOH를 실시간처럼 예측.

기본은 R11 (raw · train_all · L=120 · H=300 · row_stride=60 · win_step=5).
학습에 쓴 차량 ``1241225226`` 궤적을 한 행씩 버퍼에 넣는다.

Usage
-----
  # R11 기본 (인자 생략 가능)
  python scripts/run_infer.py --device 1241225226

  # 원본 BMS CSV를 스트림처럼 읽기 (raw만)
  python scripts/run_infer.py --source csv --device 1241225226

  # 일별 압축 R2
  python scripts/run_infer.py --exp daily --L 30 --H 1 --eval-protocol lovo --device 1241225226
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer_lib import (  # noqa: E402
    DEFAULT_DEVICE_NO,
    R11_EVAL,
    R11_H,
    R11_L,
    R11_ROW_STRIDE,
    R11_WIN_STEP,
    run_infer,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="EV SOH PatchTST — 차량 1대 실시간(리플레이) 추론",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--exp", choices=("raw", "daily"), default="raw")
    ap.add_argument(
        "--device",
        default=DEFAULT_DEVICE_NO,
        help=f"vehicle id (default {DEFAULT_DEVICE_NO})",
    )
    ap.add_argument("--L", type=int, default=None, help=f"lookback; default raw={R11_L} (R11) daily=30")
    ap.add_argument("--H", type=int, default=None, help=f"horizon; default raw={R11_H} (R11) daily=1")
    ap.add_argument("--mode", choices=("slow", "fast"), default=None)
    ap.add_argument(
        "--row-stride",
        type=int,
        default=None,
        help=f"raw prepare stride; default {R11_ROW_STRIDE} (R11)",
    )
    ap.add_argument(
        "--eval-protocol",
        choices=("lovo", "train_all"),
        default=R11_EVAL,
        help="which run folder / weights to load (R11: train_all)",
    )
    ap.add_argument("--ckpt", type=Path, default=None, help="path to .pt or stem without suffix")
    ap.add_argument(
        "--source",
        choices=("traj", "csv"),
        default="traj",
        help="traj=prepared trajectories.csv, csv=original BMS file",
    )
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--csv", dest="csv_path", type=Path, default=None)
    ap.add_argument("--traj", type=Path, default=None)
    ap.add_argument(
        "--win-step",
        type=int,
        default=R11_WIN_STEP,
        help=f"predict every N new samples (R11 train used {R11_WIN_STEP})",
    )
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N streamed samples")
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="seconds to wait after each prediction (simulate live feed)",
    )
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    run_infer(
        experiment=args.exp,
        device_no=args.device,
        L=args.L,
        H=args.H,
        mode=args.mode,
        row_stride=args.row_stride,
        eval_protocol=args.eval_protocol,
        ckpt=args.ckpt,
        source=args.source,
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        traj=args.traj,
        win_step=args.win_step,
        max_steps=args.max_steps,
        print_every=args.print_every,
        sleep_s=args.sleep,
        cpu=args.cpu,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
