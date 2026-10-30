#!/usr/bin/env python
"""Per-task table of the same recipe trained with different seeds: mean, spread and std.

Reads results/<model>/final/metrics.json (tiny) and results-holdout/<model>/final/metrics.json
(held-out-8) and prints one Markdown table per regime. Models without a result are skipped.
Usage: scripts/seed_noise_table.py --models arm-cluster-like-fiqa arm-mnrl-fiqa-seed1 ...
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(root: str, model: str) -> dict[str, float]:
    path = Path(root) / model / "final" / "metrics.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for key, value in data.items():
        if key.startswith("epoch0/pirb_") and isinstance(value, (int, float)) and not key.endswith(("ndcg_tasks", "datasets")):
            out[key.replace("epoch0/pirb_", "")] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    for root, title in (("results", "tiny"), ("results-holdout", "held-out-8")):
        rows = {m: load(root, m) for m in args.models}
        rows = {m: r for m, r in rows.items() if r}
        if len(rows) < 2:
            print(f"### {title}: fewer than two models with results\n")
            continue
        tasks = [t for t in next(iter(rows.values())) if t != "average_ndcg@10"] + ["average_ndcg@10"]
        print(f"### {title} ({len(rows)} seeds: {', '.join(rows)})\n")
        print("| task | " + " | ".join(rows) + " | mean | max-min | std |")
        print("|---|" + "---:|" * (len(rows) + 3))
        for t in tasks:
            vals = [r.get(t) for r in rows.values() if r.get(t) is not None]
            if len(vals) < 2:
                continue
            spread = max(vals) - min(vals)
            std = statistics.stdev(vals)
            print(f"| {t} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {statistics.mean(vals):.2f} | {spread:.2f} | {std:.2f} |")
        print()


if __name__ == "__main__":
    main()
