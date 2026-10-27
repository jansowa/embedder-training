#!/usr/bin/env python
"""Markdown table of the non-zero weights (active dimensions) of SPLADE models.

Two sources, both with the usual quantisation (round(w*100) > 0 = active):
  results/sparsity/<name>.json   documents only: 300 passages from each of the
                                 10 tiny tasks, written by compare_splade_sparsity.py
  results/index_cost*.json       documents and queries on one PIRB sample
                                 (150 passages + 150 queries per task), written by
                                 splade_index_cost.py; also postings per query

  python scripts/sparsity_table.py --model base=base --model soup-of-ft --model soup-v2b
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def doc_stats(name: str) -> tuple[float | None, float | None]:
    """Mean over tasks of the per-task mean and p95 active dims per passage."""
    for path in sorted(glob.glob("results/sparsity/*.json")):
        payload = json.load(open(path))
        per_task = payload.get(name)
        if not isinstance(per_task, dict):
            continue
        means = [t["mean"] for t in per_task.values() if isinstance(t, dict) and "mean" in t]
        p95s = [t["p95"] for t in per_task.values() if isinstance(t, dict) and "p95" in t]
        if means:
            return sum(means) / len(means), (sum(p95s) / len(p95s) if p95s else None)
    return None, None


def cost_stats(name: str, files: list[str]) -> dict | None:
    for path in files:
        if not Path(path).exists():
            continue
        payload = json.load(open(path))
        entry = payload.get("models", {}).get(name)
        if entry:
            return {**entry, "_source": path, "_n": payload.get("n_docs")}
    return None


def fmt(value: float | None, digits: int = 0) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def ratio(value: float | None, ref: float | None) -> str:
    return "" if value is None or ref is None or not ref else f"×{value / ref:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True,
                        help="NAME or NAME=LABEL where LABEL is the key in index_cost json; first = base")
    parser.add_argument("--cost-json", action="append", default=["results/index_cost_best.json", "results/index_cost.json"],
                        help="index-cost files, first match wins")
    args = parser.parse_args()

    rows = []
    for spec in args.model:
        name, _, label = spec.partition("=")
        label = label or name
        d_mean, d_p95 = doc_stats(name)
        if d_mean is None and label != name:
            d_mean, d_p95 = doc_stats(label)
        cost = cost_stats(label, args.cost_json)
        rows.append((name, d_mean, d_p95, cost))
    base = rows[0]
    base_cost = base[3] or {}
    sources = {r[3]["_source"]: r[3]["_n"] for r in rows if r[3]}

    print("| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d_mean, d_p95, cost in rows:
        c = cost or {}
        print("| " + " | ".join([
            name,
            fmt(d_mean, 1), ratio(d_mean, base[1]), fmt(d_p95),
            fmt(c.get("doc_active_mean"), 1),
            fmt(c.get("query_active_mean"), 1), ratio(c.get("query_active_mean"), base_cost.get("query_active_mean")),
            fmt(c.get("postings_per_query")), ratio(c.get("postings_per_query"), base_cost.get("postings_per_query")),
        ]) + " |")
    if sources:
        described = ", ".join(f"{Path(path).name} ({n} pasaży i tyle samo zapytań)" for path, n in sorted(sources.items()))
        print(f"\nŹródło próbki kosztu: {described}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
