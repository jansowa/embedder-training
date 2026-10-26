#!/usr/bin/env python
"""Paired comparison of benchmark metrics.json files against a base.

For every model: mean NDCG@10 over the shared tasks, mean of the per-task
differences against the base, a 95% bootstrap CI over tasks (10 000 resamples,
seeded), and how many tasks improved. Only tasks present in every file count,
so a held-out table and a tiny table are never mixed.

    .venv/bin/python scripts/paired_delta.py --base results/base-splade-wrapped/final/metrics.json \
        results/*/final/metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SKIP = ("average_ndcg", "ndcg_tasks", "datasets", "model", "task_selection", "queries")


def tasks_of(path: Path) -> dict[str, float]:
    d = json.load(open(path))
    out = {}
    for k, v in d.items():
        if "/pirb_" not in k or not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        name = k.split("/pirb_", 1)[1]
        if any(s in name for s in SKIP):
            continue
        out[name] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--per-task", action="store_true", help="also print per-task deltas")
    args = ap.parse_args()

    base = tasks_of(args.base)
    rng = random.Random(args.seed)
    rows = []
    for p in args.models:
        if not p.exists():
            continue
        m = tasks_of(p)
        common = sorted(set(base) & set(m))
        if not common:
            continue
        deltas = [m[t] - base[t] for t in common]
        mean_d = sum(deltas) / len(deltas)
        boots = []
        for _ in range(args.resamples):
            s = [deltas[rng.randrange(len(deltas))] for _ in deltas]
            boots.append(sum(s) / len(s))
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
        label = p.parts[-3] if len(p.parts) >= 3 else str(p)
        rows.append((label, sum(m[t] for t in common) / len(common), mean_d, lo, hi,
                     sum(1 for d in deltas if d > 0), len(common), {t: m[t] - base[t] for t in common}))
    rows.sort(key=lambda r: -r[2])
    base_mean = sum(base.values()) / len(base)
    print(f"base = {base_mean:.2f} over {len(base)} tasks\n")
    print(f"{'model':28s} {'mean':>7s} {'delta':>7s} {'95% CI':>18s} {'up':>6s}")
    for label, mean, d, lo, hi, up, n, _ in rows:
        print(f"{label:28s} {mean:7.2f} {d:+7.2f} [{lo:+6.2f}, {hi:+6.2f}] {up:>3d}/{n}")
    if args.per_task:
        print()
        tasks = sorted(base)
        print("task".ljust(36) + " ".join(r[0][:12].rjust(12) for r in rows))
        for t in tasks:
            print(t[:36].ljust(36) + " ".join(f"{r[7].get(t, float('nan')):+12.2f}" for r in rows))


if __name__ == "__main__":
    main()
