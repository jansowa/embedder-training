"""Prepare PIRB datasets before parallel benchmark workers are launched."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pirb-root", required=True, type=Path)
    parser.add_argument("--benchmark-config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pirb_root = args.pirb_root.resolve()
    sys.path.insert(0, str(pirb_root))

    from data import Benchmark

    logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    benchmark = Benchmark.from_config(str(args.benchmark_config.resolve()))
    benchmark.prepare(str(args.data_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
