#!/usr/bin/env python
"""Pairs (query, passage) for the LLM annotation of docs/llm-annotation-schema.md.

Sources (31.10.2026): the three reference cuts (fiqa-2073, squadv2-2073, msmarco-clean-2073)
and the three speakleash pools of 8000 rows (nq, hotpotqa, fever; their 2073 cuts are subsets).
Pairs:
  * every positive of every row (side "pos"),
  * the false-negative candidates (side "neg"): negatives the Polish teacher roberta-v3 scored
    at or above the row's weakest positive, read from results/data/speakleash-diag/<stem>.jsonl
    (fields v3_pos / v3_neg of the first --neg-rows rows of every file).

Order = priority, so that any prefix of the file is a balanced sample if the night is short:
  tier 1: positives of the rows that are in a 2073 cut, plus all false-negative candidates;
  tier 2: positives of the remaining pool rows.
Inside a tier the sources are interleaved round-robin after a seeded shuffle.

Output: --out (jsonl, one pair per line: pair_id, source, query_id, side, index, tier, in_cut,
synthetic, query, passage) and --manifest (counts).

Usage:
  .venv/bin/python scripts/build_annotation_pairs.py --out datasets/annotation/pairs_v1.jsonl \
      --manifest datasets/annotation/pairs_v1.manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ARMS = Path("datasets/arms")
DIAG = Path("results/data/speakleash-diag")
SOURCES = {
    # name: (rows file, cut file with the query_ids that count as "in the cut")
    "fiqa-2073": (ARMS / "fiqa-2073.jsonl", None),
    "squadv2-2073": (ARMS / "squadv2-2073.jsonl", None),
    "msmarco-clean-2073": (ARMS / "msmarco-clean-2073.jsonl", None),
    "nq-pool-8000": (ARMS / "pool" / "nq-pool-8000.jsonl", ARMS / "nq-2073.jsonl"),
    "hotpotqa-pool-8000": (ARMS / "pool" / "hotpotqa-pool-8000.jsonl", ARMS / "hotpotqa-2073.jsonl"),
    "fever-pool-8000": (ARMS / "pool" / "fever-pool-8000.jsonl", ARMS / "fever-2073.jsonl"),
}


def read_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("query") and r.get("pos") and r.get("neg"):
                    yield r


def cut_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {str(r.get("query_id")) for r in read_rows(path)}


def false_negative_indices(diag_file: Path) -> dict[str, list[int]]:
    """query_id -> indices of negatives the Polish teacher ranks at or above the weakest positive."""
    out: dict[str, list[int]] = {}
    if not diag_file.exists():
        return out
    with diag_file.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "v3_neg" not in rec or not rec.get("v3_pos"):
                continue
            weakest = min(rec["v3_pos"])
            idx = [i for i, s in enumerate(rec["v3_neg"]) if s >= weakest]
            if idx:
                out[str(rec["query_id"])] = idx
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES), help="subset of the known sources")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    tiers: dict[int, dict[str, list[dict]]] = {1: {}, 2: {}}
    counts: Counter = Counter()
    for name in a.sources:
        rows_file, cut_file = SOURCES[name]
        in_cut = cut_ids(cut_file)
        fn_idx = false_negative_indices(DIAG / f"{name}.jsonl")
        t1, t2 = [], []
        for r in read_rows(rows_file):
            qid = str(r.get("query_id"))
            member = True if in_cut is None else qid in in_cut
            synth = r.get("pos_is_synthetic") or []
            for i, p in enumerate(r["pos"]):
                pair = {"pair_id": f"{name}:{qid}:pos{i}", "source": name, "query_id": qid, "side": "pos",
                        "index": i, "tier": 1 if member else 2, "in_cut": member,
                        "synthetic": bool(synth[i]) if i < len(synth) else False,
                        "query": r["query"], "passage": p}
                (t1 if member else t2).append(pair)
                counts[(name, "pos", pair["tier"])] += 1
            for i in fn_idx.get(qid, []):
                if i < len(r["neg"]):
                    t1.append({"pair_id": f"{name}:{qid}:neg{i}", "source": name, "query_id": qid, "side": "neg",
                               "index": i, "tier": 1, "in_cut": member, "synthetic": False,
                               "query": r["query"], "passage": r["neg"][i]})
                    counts[(name, "neg", 1)] += 1
        rng.shuffle(t1); rng.shuffle(t2)
        tiers[1][name] = t1; tiers[2][name] = t2
        print(f"{name}: tier1 {len(t1)} (fn candidates {sum(1 for p in t1 if p['side'] == 'neg')}), tier2 {len(t2)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with a.out.open("w", encoding="utf-8") as fh:
        for tier in (1, 2):
            lists = [tiers[tier][s] for s in a.sources if tiers[tier].get(s)]
            while lists:
                for lst in list(lists):
                    fh.write(json.dumps(lst.pop(), ensure_ascii=False) + "\n"); n += 1
                    if not lst:
                        lists.remove(lst)
    manifest = {"pairs": n, "seed": a.seed, "sources": a.sources,
                "counts": [{"source": s, "side": side, "tier": t, "pairs": c} for (s, side, t), c in sorted(counts.items())],
                "tier1_pairs": sum(c for (s, side, t), c in counts.items() if t == 1),
                "tier2_pairs": sum(c for (s, side, t), c in counts.items() if t == 2)}
    a.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {n} pairs to {a.out} (tier 1: {manifest['tier1_pairs']}, tier 2: {manifest['tier2_pairs']})")


if __name__ == "__main__":
    main()
