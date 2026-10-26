#!/usr/bin/env python
"""Combine per-task PIRB runs (one metrics.json per task) into one summary.

Used for the second held-out set, which is run one task per process so that an
OOM on the biggest corpus cannot take the other tasks down with it.

  python scripts/aggregate_task_runs.py results-holdout2/soup-of-ft
      -> results-holdout2/soup-of-ft/summary.json  (per-task nDCG@10 + mean)
  python scripts/aggregate_task_runs.py results-holdout2/soup-of-ft --base results-holdout2/base
      -> also prints the per-task deltas against the base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(root: Path) -> dict[str, float]:
    out = {}
    for mf in sorted(root.glob("*/final/metrics.json")):
        d = json.loads(mf.read_text())
        for k, v in d.items():
            if k.startswith("epoch0/pirb_") and isinstance(v, (int, float)) and "average" not in k and "ndcg_tasks" not in k and k != "epoch0/pirb_datasets":
                out[k[len("epoch0/pirb_"):]] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--base", type=Path, default=None)
    ap.add_argument("--expect", type=int, default=None, help="expected number of tasks; the summary notes if fewer are present")
    args = ap.parse_args()
    tasks = collect(args.root)
    summary = {"model": args.root.name, "tasks": tasks, "n_tasks": len(tasks),
               "mean_ndcg@10": (sum(tasks.values()) / len(tasks)) if tasks else None,
               "complete": (args.expect is None or len(tasks) == args.expect)}
    if args.base is not None:
        base = collect(args.base)
        common = sorted(set(base) & set(tasks))
        summary["base"] = args.base.name
        summary["delta_per_task"] = {t: tasks[t] - base[t] for t in common}
        summary["mean_delta_common"] = (sum(tasks[t] - base[t] for t in common) / len(common)) if common else None
        summary["tasks_up"] = sum(tasks[t] > base[t] for t in common)
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.root.name}: {len(tasks)} tasks, mean nDCG@10 = {summary['mean_ndcg@10']}")
    for t, v in sorted(tasks.items()):
        line = f"  {t:40s} {v:6.2f}"
        if "delta_per_task" in summary and t in summary["delta_per_task"]:
            line += f"  delta {summary['delta_per_task'][t]:+.2f}"
        print(line)
    if "mean_delta_common" in summary and summary["mean_delta_common"] is not None:
        print(f"  mean delta vs {summary['base']}: {summary['mean_delta_common']:+.2f} ({summary['tasks_up']}/{len(summary['delta_per_task'])} up)")


if __name__ == "__main__":
    main()
