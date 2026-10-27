#!/usr/bin/env python
"""Per-task comparison of our models with the published PIRB leaderboard.

Reads the leaderboard export (results/leaderboard_data.json, the data.json of
sdadas/pirb-leaderboard) and our leaderboard-regime results
(results-regimes/<regime>/<model>/<taskset>.json) and prints Markdown tables,
one per task set, with the published polish-splade number, our re-measurement
of it, the fine-tuned models, their deltas, and the best standalone model of the
leaderboard on that task (the ceiling we are measured against).

  python scripts/leaderboard_compare.py --regime old-pyserini \
      --model base-splade-wrapped --model soup-of-ft --model soup-v2b \
      --taskset tiny10 --taskset heldout8 --taskset heldout2 > results/leaderboard_tables.md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

LEADERBOARD = Path("results/leaderboard_data.json")
PUBLISHED_ID = "./models/polish-splade"
HELDOUT2 = ["heldout2_czywiesz", "heldout2_gpt35wiki", "heldout2_quora", "heldout2_poquad", "heldout2_treccovid"]


def leaderboard_models(data: dict) -> list[dict]:
    found: list[dict] = []

    def walk(obj):
        if isinstance(obj, dict):
            if "id" in obj and "results" in obj:
                found.append(obj)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(data)
    return found


def ndcg(entry: dict, task: str) -> float | None:
    result = entry.get("results", {}).get(task)
    if not result:
        return None
    value = result.get("NDCG@10")
    return float(value) if value is not None else None


def our_results(regime: str, model: str, taskset: str) -> dict[str, float]:
    files = [f"results-regimes/{regime}/{model}/{t}.json" for t in HELDOUT2] if taskset == "heldout2" \
        else [f"results-regimes/{regime}/{model}/{taskset}.json"]
    tasks: dict[str, float] = {}
    for path in files:
        if not Path(path).exists():
            continue
        payload = json.load(open(path))
        for task, value in payload.get("tasks", {}).items():
            if value is not None:
                tasks[task] = float(value)
    return tasks


def fmt(value: float | None, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def delta(value: float | None, ref: float | None) -> str:
    if value is None or ref is None:
        return ""
    d = value - ref
    return f"{d:+.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regime", default="old-pyserini")
    parser.add_argument("--model", action="append", required=True, help="label under results-regimes/<regime>/; first = base")
    parser.add_argument("--taskset", action="append", required=True)
    parser.add_argument("--exclude-tag", action="append", default=[], help="leaderboard tags to skip when picking the best model (default: none)")
    args = parser.parse_args()

    data = json.load(open(LEADERBOARD))
    entries = leaderboard_models(data)
    published = next((e for e in entries if e["id"] == PUBLISHED_ID), None)
    if published is None:
        raise SystemExit(f"{PUBLISHED_ID} not found in {LEADERBOARD}")
    standalone = [e for e in entries if "standalone" in (e.get("tags") or []) and not set(e.get("tags") or []) & set(args.exclude_tag)]

    base = args.model[0]
    blocks: list[str] = []                       # per-task tables, printed after the summary
    set_means: dict[str, dict[str, float | None]] = {}   # taskset -> {"published"|model: mean}
    set_sizes: dict[str, int] = {}
    for taskset in args.taskset:
        ours = {m: our_results(args.regime, m, taskset) for m in args.model}
        tasks = sorted(ours[base]) if ours[base] else sorted({t for r in ours.values() for t in r})
        if not tasks:
            blocks.append(f"\n(no results for {taskset} in regime {args.regime})\n")
            continue
        head = ["task", "published polish-splade", f"ours {base}"] + [m for m in args.model[1:]] + [f"Δ {m} vs base" for m in args.model[1:]] + ["best standalone on PIRB"]
        lines = [f"\n### {taskset} ({args.regime}, nDCG@10)\n", "| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 2) + "---|"]
        sums: dict[str, list[float]] = {m: [] for m in args.model}
        pub_sum: list[float] = []
        for task in tasks:
            pub = ndcg(published, task)
            best = max(((ndcg(e, task) or -1.0), e["id"]) for e in standalone)
            row = [task, fmt(pub), fmt(ours[base].get(task))]
            row += [fmt(ours[m].get(task)) for m in args.model[1:]]
            row += [delta(ours[m].get(task), ours[base].get(task)) for m in args.model[1:]]
            row.append(f"{best[0]:.2f} ({best[1].split('/')[-1]})")
            lines.append("| " + " | ".join(row) + " |")
            if pub is not None:
                pub_sum.append(pub)
            for m in args.model:
                v = ours[m].get(task)
                if v is not None:
                    sums[m].append(v)
        # A model's mean counts only when it has the whole set.
        means = {m: (sum(sums[m]) / len(sums[m]) if len(sums[m]) == len(tasks) else None) for m in args.model}
        pub_mean = sum(pub_sum) / len(pub_sum) if len(pub_sum) == len(tasks) else None
        mean_row = ["**mean**", fmt(pub_mean), fmt(means[base])]
        mean_row += [fmt(means[m]) for m in args.model[1:]]
        mean_row += [delta(means[m], means[base]) for m in args.model[1:]]
        mean_row.append("")
        lines.append("| " + " | ".join(mean_row) + " |")
        blocks.append("\n".join(lines))
        set_means[taskset] = {"published": pub_mean, **means}
        set_sizes[taskset] = len(tasks)

    # Summary: mean per task set for every model, plus the mean over the tasks
    # every listed model has (so a model with a set still missing is not
    # compared on a different mix of tasks).
    all_ours = {m: {} for m in args.model}
    for taskset in args.taskset:
        for m in args.model:
            all_ours[m].update(our_results(args.regime, m, taskset))
    common = sorted(set.intersection(*(set(r) for r in all_ours.values())))
    pub_common = [v for t in common if (v := ndcg(published, t)) is not None]
    base_mean = sum(all_ours[base][t] for t in common) / len(common) if common else None
    sets = [t for t in args.taskset if t in set_means]
    print(f"\n### Średnie per zestaw ({args.regime}, nDCG@10)\n")
    head = ["model"] + [f"{t} ({set_sizes[t]})" for t in sets] + [f"wszystkie wspólne ({len(common)})", "Δ vs base (wspólne)"]
    print("| " + " | ".join(head) + " |")
    print("|---|" + "---:|" * (len(head) - 1))
    pub_row = ["published polish-splade"] + [fmt(set_means[t]["published"]) for t in sets]
    pub_row += [fmt(sum(pub_common) / len(pub_common)) if pub_common and len(pub_common) == len(common) else "", ""]
    print("| " + " | ".join(pub_row) + " |")
    for m in args.model:
        mean = sum(all_ours[m][t] for t in common) / len(common) if common else None
        row = [m] + [fmt(set_means[t][m]) for t in sets] + [fmt(mean), delta(mean, base_mean) if m != base else ""]
        print("| " + " | ".join(row) + " |")
    for block in blocks:
        print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
