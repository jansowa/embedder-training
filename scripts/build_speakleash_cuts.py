#!/usr/bin/env python
"""Cuts of the new speakleash sets (fever, hotpotqa_pl, nq-pl-mined-negatives) in the dose of mix3.

For every set the script streams the Polish train file once, applies the gate that produced
fiqa-2073 / squadv2-2073 / msmarco-clean-2073 (gemma-2.5-lightweight score of the positive
> 23.5: field pos_scores_stronger_reranker for fever / hotpotqa, pos_scores for nq, where that
reranker is the only scorer), keeps only the gate-passing positives of a row, requires 10
negatives, and then samples a candidate superset (seed 42). The candidates are tokenized with
polish-splade and two length rules are applied so that the 2 x 16 recipe fits in 8 GB (the
ldose2 memory profile: at most two texts of a row over 512 tokens, none of them the positive):

  * every kept positive <= 512 tokens (a truncated positive may lose the evidence),
  * at most --max-long-negs negatives > 512 tokens; negatives are re-ordered longest first,
    so the long ones share the first columns of a micro-batch (MNRL is order-invariant).

Outputs (datasets/arms/, gitignored):
  <set>-2073.jsonl            the cut (first 2073 survivors of the shuffled candidates)
  pool/<set>-pool-8000.jsonl  the 8000-row pool the cut is the head of (for the student-tail and
                              Polish-reranker diagnostics and later variants)
  fiqa-squadv2-<set>-6219.jsonl  fiqa-2073 + squadv2-2073 + <set>-2073, shuffled (the n4 pattern)
  manifest_speakleash.json    counts, pass rates and length statistics per set

Usage: .venv/bin/python scripts/build_speakleash_cuts.py [--sets fever hotpotqa nq] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np

ARMS = Path("datasets/arms")
SRC = Path("/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/FlagEmbedding-data/speakleash")
SETS = {
    "fever": {"file": SRC / "fever/train_pl.jsonl", "gate_field": "pos_scores_stronger_reranker"},
    "hotpotqa": {"file": SRC / "hotpotqa/train_pl.jsonl", "gate_field": "pos_scores_stronger_reranker"},
    "nq": {"file": SRC / "nq/train_pl.jsonl", "gate_field": "pos_scores"},
}
POS_PARALLEL = ["pos", "pos_id", "pos_scores", "pos_is_synthetic", "original_pos_scores",
                "pos_scores_stronger_reranker", "pos_metricx", "pos_labse_margin"]
NEG_PARALLEL = ["neg", "neg_id", "neg_scores", "neg_selection_tier", "original_neg_scores"]


def pct(x, ps=(5, 25, 50, 75, 95)):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    return {f"p{p}": round(float(np.percentile(x, p)), 3) for p in ps} | {"n": int(len(x))} if len(x) else {}


def subset(row: dict, keys: list[str], idx: list[int]) -> None:
    """Keep the positions idx of every list field in keys that is parallel to keys[0]."""
    n = len(row.get(keys[0]) or [])
    for k in keys:
        v = row.get(k)
        if isinstance(v, list) and len(v) == n:
            row[k] = [v[i] for i in idx]


def gate_positives(row: dict, field: str, thr: float) -> list[int]:
    scores = row.get(field) or []
    if len(scores) != len(row.get("pos") or []):
        return []
    return [i for i, s in enumerate(scores) if s is not None and float(s) > thr]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--threshold", type=float, default=23.5)
    ap.add_argument("--rows", type=int, default=2073)
    ap.add_argument("--pool", type=int, default=8000)
    ap.add_argument("--candidates", type=int, default=24000, help="gated rows sampled before tokenizing")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-long-negs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default="models/base-splade-wrapped")
    ap.add_argument("--dry-run", action="store_true", help="statistics only, write nothing")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    def ntok(texts: list[str]) -> list[int]:
        out: list[int] = []
        for i in range(0, len(texts), 1024):
            out.extend(len(e) for e in tok(texts[i:i + 1024], add_special_tokens=True, truncation=False)["input_ids"])
        return out

    manifest_path = ARMS / "manifest_speakleash.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    fiqa = squad = None

    for name in a.sets:
        spec = SETS[name]
        rng = random.Random(a.seed)
        total = gated_rows = 0
        pos_total = pos_pass = 0
        rows_1pos = 0
        gate_scores: list[float] = []
        metricx_q: list[float] = []
        metricx_p: list[float] = []
        kept: list[dict] = []  # reservoir of candidate rows (gated, 10 negatives), positives already subset
        # pass 1: stream, gate, reservoir-sample the candidates
        with spec["file"].open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                total += 1
                scores = r.get(spec["gate_field"]) or []
                gate_scores.extend(float(s) for s in scores if s is not None)
                pos_total += len(r.get("pos") or [])
                idx = gate_positives(r, spec["gate_field"], a.threshold)
                pos_pass += len(idx)
                if "query_metricx" in r and r["query_metricx"] is not None:
                    metricx_q.append(float(r["query_metricx"]))
                if r.get("pos_metricx"):
                    metricx_p.extend(float(m) for m in r["pos_metricx"] if m is not None)
                if not idx or not r.get("query") or len(r.get("neg") or []) < 10:
                    continue
                gated_rows += 1
                rows_1pos += len(idx) == 1
                subset(r, POS_PARALLEL, idx)
                # reservoir sampling of the candidate superset
                if len(kept) < a.candidates:
                    kept.append(r)
                else:
                    j = rng.randrange(gated_rows)
                    if j < a.candidates:
                        kept[j] = r
        rng.shuffle(kept)
        # pass 2: token lengths of the candidates, length rules, negatives longest first
        q_len = ntok([r["query"] for r in kept])
        p_len = ntok([p for r in kept for p in r["pos"]])
        n_len = ntok([n for r in kept for n in r["neg"]])
        pi = ni = 0
        survivors: list[dict] = []
        drop_pos_long = drop_negs_long = 0
        ratio: list[float] = []
        longer_share: list[float] = []
        for k, r in enumerate(kept):
            pl = p_len[pi:pi + len(r["pos"])]; pi += len(r["pos"])
            nl = n_len[ni:ni + len(r["neg"])]; ni += len(r["neg"])
            keep_pos = [i for i, L in enumerate(pl) if L <= a.max_tokens]
            if not keep_pos:
                drop_pos_long += 1
                continue
            if sum(L > a.max_tokens for L in nl) > a.max_long_negs:
                drop_negs_long += 1
                continue
            if len(keep_pos) < len(pl):
                subset(r, POS_PARALLEL, keep_pos)
                pl = [pl[i] for i in keep_pos]
            order = sorted(range(len(nl)), key=lambda i: -nl[i])
            if order != list(range(len(nl))):
                for key in NEG_PARALLEL:
                    v = r.get(key)
                    if isinstance(v, list) and len(v) == len(nl):
                        r[key] = [v[i] for i in order]
                nl = [nl[i] for i in order]
            r["_q_tok"], r["_pos_tok"], r["_neg_tok"] = q_len[k], pl, nl
            ratio.append(st.median(nl) / max(1, max(pl)))
            longer_share.append(float(np.mean(np.asarray(nl) > max(pl))))
            survivors.append(r)
        cut = survivors[:a.rows]
        pool = survivors[:a.pool]
        stats = {
            "source": str(spec["file"]), "gate_field": spec["gate_field"], "threshold": a.threshold,
            "rows_total": total, "rows_gated_10neg": gated_rows, "rows_gated_share": round(gated_rows / total, 4),
            "positives_total": pos_total, "positives_pass": pos_pass, "positives_pass_share": round(pos_pass / pos_total, 4),
            "gated_rows_with_one_positive_share": round(rows_1pos / max(1, gated_rows), 4),
            "gate_score_percentiles": pct(gate_scores),
            "query_metricx": pct(metricx_q), "pos_metricx": pct(metricx_p),
            "candidates_sampled": len(kept), "dropped_positive_over_512": drop_pos_long,
            "dropped_more_than_%d_long_negatives" % a.max_long_negs: drop_negs_long,
            "survivors": len(survivors), "survivor_share_of_candidates": round(len(survivors) / max(1, len(kept)), 4),
            "cut_rows": len(cut), "cut_training_examples": sum(len(r["pos"]) for r in cut),
            "cut_positives_per_row": dict(sorted(Counter(len(r["pos"]) for r in cut).items())),
            "cut_synthetic_positives": sum(sum(bool(x) for x in (r.get("pos_is_synthetic") or [])) for r in cut),
            "cut_tokens_query": pct([r["_q_tok"] for r in cut]),
            "cut_tokens_pos": pct([L for r in cut for L in r["_pos_tok"]]),
            "cut_tokens_neg": pct([L for r in cut for L in r["_neg_tok"]]),
            "cut_long_negatives_per_row": dict(sorted(Counter(sum(L > a.max_tokens for L in r["_neg_tok"]) for r in cut).items())),
            "cut_median_neg_len_over_max_pos_len": pct(ratio[:len(cut)]),
            "cut_share_negs_longer_than_longest_pos": round(float(np.mean(longer_share[:len(cut)])), 4) if cut else None,
            "cut_gate_score_percentiles": pct([s for r in cut for s in (r.get(spec["gate_field"]) or [])]),
            "pool_rows": len(pool),
        }
        print(f"\n=== {name}")
        print(json.dumps(stats, indent=1, ensure_ascii=False))
        manifest[name] = stats
        if a.dry_run:
            continue
        for r in survivors:
            for k in ("_q_tok", "_pos_tok", "_neg_tok"):
                r.pop(k, None)
        (ARMS / "pool").mkdir(parents=True, exist_ok=True)
        with (ARMS / f"{name}-{a.rows}.jsonl").open("w", encoding="utf-8") as fh:
            for r in cut:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (ARMS / "pool" / f"{name}-pool-{a.pool}.jsonl").open("w", encoding="utf-8") as fh:
            for r in pool:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if fiqa is None:
            fiqa = [json.loads(l) for l in (ARMS / "fiqa-2073.jsonl").open(encoding="utf-8") if l.strip()]
            squad = [json.loads(l) for l in (ARMS / "squadv2-2073.jsonl").open(encoding="utf-8") if l.strip()]
        mix = fiqa + squad + cut
        random.Random(a.seed).shuffle(mix)
        mix_path = ARMS / f"fiqa-squadv2-{name}-{len(mix)}.jsonl"
        with mix_path.open("w", encoding="utf-8") as fh:
            for r in mix:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[name]["mix"] = {"file": mix_path.name, "rows": len(mix),
                                 "training_examples": sum(len(r["pos"]) for r in mix)}
        print(f"wrote {ARMS / f'{name}-{a.rows}.jsonl'} ({len(cut)} rows), pool {len(pool)}, {mix_path.name} ({len(mix)} rows)")
    if not a.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
