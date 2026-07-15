"""Prepare PIRB datasets before parallel benchmark workers are launched."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import logging
from pathlib import Path
import sys
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pirb-root", required=True, type=Path)
    parser.add_argument("--benchmark-config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--scope", default=None)
    parser.add_argument("--manifest-json", default=None, type=Path)
    return parser


def _task_groups(benchmark: Any, data_dir: Path) -> list[dict[str, Any]]:
    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    group_paths: dict[str, set[Path]] = {}
    for task in benchmark.tasks:
        cache_name = str(task.task_cache_name())
        group = groups.setdefault(
            cache_name,
            {
                "cache_name": cache_name,
                "task_ids": [],
                "size_bytes": 0,
            },
        )
        group["task_ids"].append(str(task.task_id))
        paths = group_paths.setdefault(cache_name, set())
        paths.add(Path(task.passages_path(str(data_dir))))
        paths.add(Path(task.queries_path(str(data_dir))))

    for cache_name, paths in group_paths.items():
        groups[cache_name]["size_bytes"] = sum(path.stat().st_size for path in paths if path.is_file())
    return list(groups.values())


def main() -> int:
    args = build_parser().parse_args()
    pirb_root = args.pirb_root.resolve()
    sys.path.insert(0, str(pirb_root))

    from data import Benchmark

    logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    benchmark = Benchmark.from_config(str(args.benchmark_config.resolve()))
    benchmark.prepare(str(args.data_dir.resolve()))
    if args.manifest_json is not None:
        if args.scope is None:
            raise ValueError("--scope is required with --manifest-json")
        scope = "full" if args.scope == "all" else args.scope
        selected_benchmark = benchmark.filter(scope, str(args.data_dir.resolve()))
        manifest = {
            "scope": args.scope,
            "groups": _task_groups(selected_benchmark, args.data_dir.resolve()),
        }
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
