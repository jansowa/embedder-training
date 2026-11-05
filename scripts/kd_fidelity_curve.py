#!/usr/bin/env python
"""Print the fidelity curve of a distill.train_kd run (from its log.jsonl), or one ranking-agreement line.

  python scripts/kd_fidelity_curve.py runs/kd-NAME/log.jsonl
  python scripts/kd_fidelity_curve.py --agreement results/rank_agreement/LABEL.json [--queries results/fidelity/LABEL.queries.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def curve(path: str) -> None:
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        kinds = sorted({k.split("/")[1] for k in d if k.startswith("fidelity/") and k.count("/") == 2})
        if not kinds:
            continue
        parts = []
        for k in kinds:
            c, l1, dims = d.get(f"fidelity/{k}/cosine"), d.get(f"fidelity/{k}/l1"), d.get(f"fidelity/{k}/student_dims")
            part = f"{k}: cos {c:.3f}" if c is not None else f"{k}:"
            if l1 is not None:
                part += f" l1 {l1:.3f}"
            if dims is not None:
                part += f" dims {dims:.0f}"
            parts.append(part)
        teacher = f" (teacher dims {d['fidelity/teacher_dims']:.0f})" if "fidelity/teacher_dims" in d else ""
        print(f"  step {d.get('step', 0):>6}: " + " | ".join(parts) + teacher)


def agreement(path: str, queries: str | None) -> None:
    d = json.load(open(path, encoding="utf-8"))
    label = os.path.basename(path)[: -len(".json")]
    line = f"{label:<44} tau {d['kendall_tau']:.3f}  top1 {d['top1_agreement']:.3f}  pos-first {d['positive_first']:.3f}  margin {d['mean_margin']:.2f}"
    if queries and os.path.exists(queries):
        kinds = json.load(open(queries, encoding="utf-8"))["kinds"]
        a = kinds.get("query") or next(iter(kinds.values()))
        line += f"  | queries: cos {a['cosine']['mean']:.3f} l1 {a['l1_relative_error']['mean']:.3f} dims {a['student_active_dims']['mean']:.0f}/{a['teacher_active_dims']['mean']:.0f}"
    print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", help="log.jsonl of a KD run")
    parser.add_argument("--agreement", help="rank_agreement JSON")
    parser.add_argument("--queries", help="fidelity JSON on queries, printed next to the agreement line")
    args = parser.parse_args(argv)
    if args.agreement:
        agreement(args.agreement, args.queries)
    elif args.log:
        curve(args.log)
    else:
        parser.error("give a log.jsonl or --agreement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
