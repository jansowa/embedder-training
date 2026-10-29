#!/usr/bin/env python
"""Per-task table of the leaderboard-regime results (results-regimes/<regime>/<model>/<taskset>.json).

The means hide where a model wins: long-easy-5070 takes arguana-pl by 10 points and loses
fiqa-pl by 8, and the whole gain of soup-v7a over soup-v6a is arguana plus trec-covid. This
prints one Markdown table per task set with the models as columns, every task as a row and
the mean as the last row, so a queue summary can show the per-task picture next to the means.

    python scripts/regime_task_table.py --models base-splade-wrapped n1-5070 soup-v7a \
        --tasksets tiny10 heldout8 heldout2_treccovid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(regime_dir: Path, model: str, taskset: str) -> dict[str, float] | None:
    path = regime_dir / model / f"{taskset}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks") or {}
    return {task: float(value) for task, value in tasks.items() if isinstance(value, (int, float))}


def table(regime_dir: Path, models: list[str], taskset: str) -> str:
    columns = {m: load(regime_dir, m, taskset) for m in models}
    columns = {m: v for m, v in columns.items() if v}
    if not columns:
        return f"### {taskset}\n\n(no results)\n"
    tasks = sorted({t for v in columns.values() for t in v})
    lines = [f"### {taskset}", "", "| task | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    for task in tasks:
        cells = [f"{v[task]:.2f}" if task in v else "" for v in columns.values()]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    means = [f"**{sum(v.values()) / len(v):.2f}**" for v in columns.values()]
    lines.append("| **mean** | " + " | ".join(means) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regime-dir", type=Path, default=Path("results-regimes/quantized"))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tasksets", nargs="+", default=["tiny10", "heldout8"])
    args = parser.parse_args()
    print("\n".join(table(args.regime_dir, args.models, ts) for ts in args.tasksets))


if __name__ == "__main__":
    main()
