#!/usr/bin/env python
"""Fold single-task regime results into a task-set result (e.g. a re-run of fiqa-pl into heldout8.json).

  python scripts/merge_regime_task.py --into results-regimes/quantized/M/heldout8.json \\
      --from results-regimes/quantized/M/heldout8_fiqa.json [more ...] --taskset heldout8 --n-tasks 8

Tasks of every --from file are added to (or replace in) the target's ``tasks``; ``mean_ndcg@10`` is recomputed and
``n_tasks`` set to the number of non-null tasks. The mean is written only when all --n-tasks tasks are present
(otherwise it stays, and the file keeps reporting the partial count).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--into", type=Path, required=True)
    p.add_argument("--from", dest="sources", type=Path, nargs="+", required=True)
    p.add_argument("--taskset", default=None)
    p.add_argument("--n-tasks", type=int, default=8)
    args = p.parse_args(argv)
    target = json.load(open(args.into))
    tasks = {k: v for k, v in target.get("tasks", {}).items() if v is not None}
    for src in args.sources:
        if not src.is_file():
            print(f"[WARN] {src} missing", file=sys.stderr)
            continue
        for k, v in json.load(open(src)).get("tasks", {}).items():
            if v is not None:
                tasks[k] = v
    target["tasks"] = tasks
    target["n_tasks"] = len(tasks)
    if len(tasks) >= args.n_tasks:
        target["mean_ndcg@10"] = sum(tasks.values()) / len(tasks)
    if args.taskset:
        target["taskset"] = args.taskset
    json.dump(target, open(args.into, "w"), indent=2)
    print(f"{args.into}: {len(tasks)} tasks, mean {target.get('mean_ndcg@10')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
