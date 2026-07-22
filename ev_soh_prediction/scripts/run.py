#!/usr/bin/env python3
"""Deprecated entrypoint — use run_daily.py or run_raw.py.

  일별 압축:  python scripts/run_daily.py ...
  로우 비압축: python scripts/run_raw.py ...
"""
from __future__ import annotations

import sys


def main() -> None:
    print(
        "scripts/run.py 는 더 이상 쓰지 않습니다.\n"
        "  일별 압축  →  python scripts/run_daily.py prepare|train|plot|all ...\n"
        "  로우 비압축 →  python scripts/run_raw.py   prepare|train|plot|all ...\n",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
