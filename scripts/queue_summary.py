#!/usr/bin/env python
"""Print one Markdown table of everything measured so far.

Columns: tiny (10 tasks) and its delta vs base, held-out-8 and delta, the
second held-out set (5 tasks) and delta, index cost (doc active dims and
postings/query relative to base) where a profile exists. Reads
results/<model>/final/metrics.json, results-holdout/<model>/final/metrics.json,
results-holdout2/<model>/summary.json and results/index_cost*.json.
"""

from __future__ import annotations

import json
from pathlib import Path

KEY = "epoch0/pirb_average_ndcg@10"


def avg(path: Path) -> float | None:
    try:
        return float(json.loads(path.read_text())[KEY])
    except Exception:
        return None


def main() -> None:
    tiny = {p.parent.parent.name: avg(p) for p in Path("results").glob("*/final/metrics.json")}
    hold = {p.parent.parent.name: avg(p) for p in Path("results-holdout").glob("*/final/metrics.json")}
    hold2 = {}
    for p in Path("results-holdout2").glob("*/summary.json"):
        d = json.loads(p.read_text())
        hold2[p.parent.name] = (d.get("mean_ndcg@10"), d.get("n_tasks"))
    cost = {}
    for p in sorted(Path("results").glob("index_cost*.json")):
        try:
            for k, v in json.loads(p.read_text())["models"].items():
                cost[k] = v
        except Exception:
            pass
    base_t = tiny.get("base-splade-wrapped")
    base_h = hold.get("base")
    base_h2 = hold2.get("base", (None, None))[0]
    base_c = cost.get("base")

    def fmt(v, b=None):
        if v is None:
            return ""
        s = f"{v:.2f}"
        if b is not None:
            s += f" ({v - b:+.2f})"
        return s

    names = sorted(set(tiny) | set(hold) | set(hold2), key=lambda n: -(tiny.get(n) or 0))
    print("| model | tiny (Δ) | held-out-8 (Δ) | held-out-2 (Δ) | doc dims ×base | postings/query ×base |")
    print("|---|---:|---:|---:|---:|---:|")
    for n in names:
        if n in ("base-splade", "datasets"):
            continue
        c = cost.get(n) or cost.get(n.replace("arm-", ""))
        dd = pq = ""
        if c and base_c:
            dd = f"{c['doc_active_mean'] / base_c['doc_active_mean']:.2f}"
            pq = f"{c['postings_per_query'] / base_c['postings_per_query']:.2f}"
        h2 = hold2.get(n)
        h2s = fmt(h2[0], base_h2) + (f" [{h2[1]} tasks]" if h2 and h2[1] not in (None, 5) else "") if h2 else ""
        print(f"| {n} | {fmt(tiny.get(n), base_t)} | {fmt(hold.get(n), base_h)} | {h2s} | {dd} | {pq} |")
    print()
    print(f"base: tiny {base_t}, held-out-8 {base_h}, held-out-2 {base_h2}")


if __name__ == "__main__":
    main()
