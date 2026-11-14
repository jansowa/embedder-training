#!/usr/bin/env python
"""Choose the model to release among measured candidates by a fixed rule.

The first candidate is the reference (the simpler recipe). The winner is the candidate with the highest score among
those within the limits (dims/doc, postings/query); it replaces the reference only when it leads by at least --margin
(with margin 0: by any positive lead). The score (--metric) is either `sets2`, the mean of the tiny10 and heldout8
means, or `tasks`, the mean over all the tasks of tiny10, heldout8 and trec-covid, every task with the same weight
(19 tasks); both use the unrounded numbers. Candidates are given as NAME=LABEL=DIMS_PER_DOC=COMPONENTS; the query-side
numbers come from the index cost JSON of scripts/splade_index_cost.py.

A queue that is already running cannot get new arguments, so two files next to its log override them when they exist,
named after --out so that they reach one queue only: logs/<stem of --out>.margin.txt (first line = the margin) and
logs/<stem of --out>.metric.txt (first line = the metric), e.g. logs/v62_freeze.metric.txt.

    python scripts/freeze_choice.py --regimes results-regimes/quantized --cost index_cost.json --out freeze.json \\
        "dose3=arm-a-bias015=374.7=dose3" "soup dq=soup-v20-dq-bias010=401.2=dose3 + faq20"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = {
    "sets2": ("mean_tiny10_heldout8", "the mean of tiny10 and heldout8", "mean of the two sets"),
    "tasks": ("mean_tasks", "the mean over the tasks of tiny10, heldout8 and trec-covid", "mean of the 19 tasks"),
}
TREC = "heldout2_treccovid"


def first_token(path: Path) -> str | None:
    if not path.exists():
        return None
    tokens = path.read_text().split()
    return tokens[0] if tokens else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--regimes", type=Path, required=True, help="results-regimes/<regime> directory")
    ap.add_argument("--cost", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-dims", type=float, default=410.0)
    ap.add_argument("--max-postings", type=float, default=16807.0)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--margin-file", type=Path, default=None, help="when this file exists, the number on its first line replaces --margin (default logs/<stem of --out>.margin.txt)")
    ap.add_argument("--metric", choices=sorted(METRICS), default="sets2")
    ap.add_argument("--metric-file", type=Path, default=None, help="when this file exists, the name on its first line replaces --metric (default logs/<stem of --out>.metric.txt)")
    ap.add_argument("--cost-key", action="append", default=[], metavar="LABEL=KEY", help="name of a label inside the cost JSON when it differs")
    ap.add_argument("candidates", nargs="+", metavar="NAME=LABEL=DIMS=COMPONENTS")
    args = ap.parse_args(argv)
    args.margin_file = args.margin_file or Path("logs") / (args.out.stem + ".margin.txt")
    args.metric_file = args.metric_file or Path("logs") / (args.out.stem + ".metric.txt")
    margin_from, metric_from = "--margin", "--metric"
    token = first_token(args.margin_file)
    if token is not None:
        try:
            args.margin, margin_from = float(token), str(args.margin_file)
        except ValueError:
            print("WARNING: %s has no number on its first line; --margin %s stays" % (args.margin_file, args.margin))
    token = first_token(args.metric_file)
    if token is not None:
        if token in METRICS:
            args.metric, metric_from = token, str(args.metric_file)
        else:
            print("WARNING: %s names no known metric (%s); --metric %s stays" % (args.metric_file, token, args.metric))
    key, metric_text, _ = METRICS[args.metric]
    bar = ">= %.2f" % args.margin if args.margin > 0 else "any lead wins: margin 0"

    cost = json.loads(args.cost.read_text())["models"] if args.cost.exists() else {}
    keys = dict(c.split("=", 1) for c in args.cost_key)

    def result(label: str, ts: str) -> tuple[float | None, int]:
        p = args.regimes / label / (ts + ".json")
        if not p.exists():
            return None, 0
        d = json.loads(p.read_text())
        return d["mean_ndcg@10"], int(d.get("n_tasks") or len(d.get("tasks") or {}))

    rows = []
    for spec in args.candidates:
        name, label, dims, parts = spec.split("=")
        (t, nt), (h, nh), (tc, ntc) = result(label, "tiny10"), result(label, "heldout8"), result(label, TREC)
        c = cost.get(keys.get(label, label), {})
        both = t is not None and h is not None
        r = {"name": name, "label": label, "components": parts, "tiny10": t, "heldout8": h, "treccovid": tc,
             "mean_tiny10_heldout8": (t + h) / 2 if both else None,
             "mean_tasks": (t * nt + h * nh + tc * ntc) / (nt + nh + ntc) if both and tc is not None else None, "n_tasks": nt + nh + ntc,
             "dims_per_doc": float(dims) if dims else None, "dims_per_query": c.get("query_active_mean"), "postings_per_query": c.get("postings_per_query")}
        r["score"] = r[key]
        r["within_limits"] = bool(r["score"] is not None and r["dims_per_doc"] is not None and r["dims_per_doc"] <= args.max_dims
                                  and r["postings_per_query"] is not None and r["postings_per_query"] <= args.max_postings)
        rows.append(r)
    ref = rows[0]
    ok = [r for r in rows if r["within_limits"]]
    others = [r for r in ok if r is not ref]
    best_other = max(others, key=lambda r: r["score"]) if others else None
    lead = best_other["score"] - ref["score"] if best_other and ref["score"] is not None else None
    if best_other is None:
        winner = ref
    elif not ref["within_limits"]:
        winner = best_other
    else:
        winner = best_other if lead > 0 and lead >= args.margin else ref
    if winner is not ref:
        why = ("%s leads %s by %.3f on %s (%s) within the limits" % (winner["name"], ref["name"], lead, metric_text, bar) if ref["within_limits"]
               else "%s is the best candidate within the limits (%s is outside them or not measured)" % (winner["name"], ref["name"]))
    elif best_other is None:
        why = "no other candidate within the limits: %s stays" % ref["name"]
    elif lead <= 0:
        why = "no candidate within the limits beats %s on %s (the best other, %s, is %.3f below)" % (ref["name"], metric_text, best_other["name"], -lead)
    else:
        why = "the best other candidate, %s, leads %s by only %.3f on %s (< %.2f): the simpler recipe stays" % (best_other["name"], ref["name"], lead, metric_text, args.margin)
    args.out.write_text(json.dumps({
        "winner": winner["label"], "winner_name": winner["name"], "components": winner["components"], "contains_faq": "faq20" in winner["components"],
        "is_reference": winner is ref, "best_other": best_other["label"] if best_other else None, "lead_of_best_other": lead, "why": why,
        "rule": {"max_dims_per_doc": args.max_dims, "max_postings_per_query": args.max_postings, "margin": args.margin, "margin_from": margin_from,
                 "metric": args.metric, "metric_from": metric_from}, "candidates": rows}, indent=2) + "\n")
    f = lambda v, p="%.2f": p % v if v is not None else "-"  # noqa: E731
    for r in rows:
        print("%s (%s): score %s (tiny10 %s / heldout8 %s / trec-covid %s; two sets %s, tasks %s), %s dims/doc, %s dims/query, %s postings/query%s" % (
            r["name"], r["label"], f(r["score"], "%.3f"), f(r["tiny10"]), f(r["heldout8"]), f(r["treccovid"]), f(r["mean_tiny10_heldout8"], "%.3f"), f(r["mean_tasks"], "%.3f"),
            f(r["dims_per_doc"], "%.1f"), f(r["dims_per_query"], "%.1f"), f(r["postings_per_query"], "%.0f"), "" if r["within_limits"] else "  OUTSIDE THE LIMITS (or not measured)"))
    print("RULE: the highest score wins, score = %s (%s), margin: %s (%s)" % (metric_text, metric_from, bar, margin_from))
    print("FROZEN: %s; %s" % (winner["label"], why))
    return 0


def table(path: Path) -> None:
    d = json.loads(path.read_text())
    metric = d.get("rule", {}).get("metric", "sets2")
    key = METRICS[metric][0]
    f = lambda v, p="%.2f": p % v if v is not None else "-"  # noqa: E731
    mark = lambda m: " (the rule)" if m == metric else ""  # noqa: E731
    print("| candidate | label | %s%s | %s%s | tiny10 | heldout8 | trec-covid | dims/doc | dims/query | postings/query | within the limits |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|" % (
        METRICS["tasks"][2], mark("tasks"), METRICS["sets2"][2], mark("sets2")))
    for r in sorted(d["candidates"], key=lambda r: -(r.get(key) or 0)):
        b = "**" if r["label"] == d["winner"] else ""
        print("| %s%s%s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (b, r["name"], b, r["label"], f(r.get("mean_tasks"), "%.3f"), f(r["mean_tiny10_heldout8"], "%.3f"), f(r["tiny10"]), f(r["heldout8"]),
              f(r["treccovid"]), f(r["dims_per_doc"], "%.1f"), f(r["dims_per_query"], "%.1f"), f(r["postings_per_query"], "%.0f"), "yes" if r["within_limits"] else "no"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--table":
        table(Path(sys.argv[2]))
        raise SystemExit(0)
    raise SystemExit(main())
