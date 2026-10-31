#!/usr/bin/env python
"""Anatomy of the new training files the way PLAN-2026-10-31 §4 measures it on FiQA.

For every input file (pools and cuts of fever / hotpotqa / nq, and the reference files
fiqa-2073, squadv2-2073, msmarco-clean-2073) the script scores every row with two models:

  * the student, base polish-splade (models/base-splade-wrapped, 512 tokens, dot product):
    score of every positive and negative; per row the margin of each negative to the best
    positive (the "tail" = negatives with margin > --tail-margin, default -3, the band that
    hurt on FiQA; "above" = negatives scored above the best positive);
  * the Polish teacher, sdadas/polish-reranker-roberta-v3 (CrossEncoder, bf16, 512 tokens,
    raw logits, the convention of scripts/fiqa_v3_mine.env): every positive of every row, and
    the negatives of the first --neg-rows rows (default 2073 = the cut) to find negatives the
    Polish teacher ranks at or above the row's weakest positive (false-negative check).

The gate scores of the files (gemma-2.5-lightweight on the English texts) are copied next to
the roberta-v3 scores, so the summary reports how the English gate relates to a Polish one:
Pearson r, and the share of gate-passing positives under the roberta-v3 percentiles of the
known-good fiqa-2073-v3 positives (datasets/arms/fiqa-2073-v3.jsonl, pos_scores).

Outputs: --out (summary json) and --per-row-dir/<stem>.jsonl (one record per row with all
scores, so the no-tail / Polish-gate variants can be cut without new scoring).

Usage (in a queue step; GPU):
  .venv/bin/python scripts/speakleash_pool_diag.py --input datasets/arms/pool/nq-pool-8000.jsonl \
      --input datasets/arms/fiqa-2073.jsonl --out results/data/speakleash-diag.json \
      --per-row-dir results/data/speakleash-diag
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

GATE_FIELDS = ("pos_scores_stronger_reranker", "pos_scores")


def load_rows(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("query") and r.get("pos") and r.get("neg"):
                rows.append(r)
            if n and len(rows) >= n:
                break
    return rows


def gate_scores(r: dict) -> list[float | None]:
    for f in GATE_FIELDS:
        v = r.get(f)
        if isinstance(v, list) and len(v) == len(r["pos"]):
            return [None if s is None else float(s) for s in v]
    return [None] * len(r["pos"])


def pct(x, ps=(5, 10, 25, 50, 75, 90, 95)):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    return ({f"p{p}": round(float(np.percentile(x, p)), 3) for p in ps} | {"n": int(len(x)), "mean": round(float(x.mean()), 3)}) if len(x) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, action="append", required=True)
    ap.add_argument("--n", type=int, default=0, help="rows per file (0 = all)")
    ap.add_argument("--neg-rows", type=int, default=2073, help="rows whose negatives the Polish teacher scores")
    ap.add_argument("--student", default="models/base-splade-wrapped")
    ap.add_argument("--teacher", default="sdadas/polish-reranker-roberta-v3")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--teacher-batch-size", type=int, default=32)
    ap.add_argument("--tail-margin", type=float, default=-3.0)
    ap.add_argument("--reference-v3", type=Path, default=Path("datasets/arms/fiqa-2073-v3.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-row-dir", type=Path, required=True)
    ap.add_argument("--no-teacher", action="store_true")
    a = ap.parse_args()
    a.per_row_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sentence_transformers import CrossEncoder
    from sentence_transformers.sparse_encoder import SparseEncoder

    student = SparseEncoder(a.student, device=device)
    student.max_seq_length = a.max_seq_length
    student.eval()
    teacher = None
    if not a.no_teacher:
        teacher = CrossEncoder(a.teacher, device=device, trust_remote_code=True, max_length=a.max_seq_length,
                               model_kwargs={"torch_dtype": torch.bfloat16} if device == "cuda" else {},
                               activation_fn=torch.nn.Identity())

    def enc(texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            return student.encode(texts, batch_size=a.batch_size, convert_to_sparse_tensor=False,
                                  convert_to_tensor=True, show_progress_bar=False).float()

    def teach(pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        out = teacher.predict(pairs, batch_size=a.teacher_batch_size, show_progress_bar=False)
        return [float(s) for s in (out.tolist() if hasattr(out, "tolist") else out)]

    # reference distribution of the Polish teacher on known-good positives
    ref_v3 = []
    if a.reference_v3.exists():
        for line in a.reference_v3.open(encoding="utf-8"):
            if line.strip():
                ref_v3.extend(float(s) for s in (json.loads(line).get("pos_scores") or []) if s is not None)
    ref_pct = pct(ref_v3)
    ref_thr = {k: ref_pct.get(k) for k in ("p5", "p10", "p25")}

    summary = {"student": a.student, "teacher": None if a.no_teacher else a.teacher, "tail_margin": a.tail_margin,
               "reference_v3_positives": {"file": str(a.reference_v3), **ref_pct}, "files": {}}
    for path in a.input:
        t0 = time.time()
        rows = load_rows(path, a.n)
        recs = []
        # student scores in chunks of rows
        CH = 64
        for b in range(0, len(rows), CH):
            batch = rows[b:b + CH]
            q = enc([r["query"] for r in batch])
            docs = [p for r in batch for p in r["pos"]] + [n for r in batch for n in r["neg"]]
            d = enc(docs)
            npos = [len(r["pos"]) for r in batch]
            nneg = [len(r["neg"]) for r in batch]
            pi, ni = 0, sum(npos)
            for i, r in enumerate(batch):
                ps = (q[i] @ d[pi:pi + npos[i]].T).tolist(); pi += npos[i]
                ns = (q[i] @ d[ni:ni + nneg[i]].T).tolist(); ni += nneg[i]
                recs.append({"query_id": str(r.get("query_id", b + i)), "line": b + i,
                             "gate": gate_scores(r), "splade_pos": [round(x, 3) for x in ps],
                             "splade_neg": [round(x, 3) for x in ns],
                             "q_dims": int((q[i] > 0).sum().item())})
            del q, d
        # teacher scores
        if teacher is not None:
            pairs = [(r["query"], p) for r in rows for p in r["pos"]]
            v3 = teach(pairs)
            k = 0
            for r, rec in zip(rows, recs):
                rec["v3_pos"] = [round(x, 3) for x in v3[k:k + len(r["pos"])]]; k += len(r["pos"])
            nrows = rows[:a.neg_rows]
            pairs = [(r["query"], n) for r in nrows for n in r["neg"]]
            v3 = teach(pairs)
            k = 0
            for r, rec in zip(nrows, recs):
                rec["v3_neg"] = [round(x, 3) for x in v3[k:k + len(r["neg"])]]; k += len(r["neg"])
        # summary
        tail_cnt, above_cnt, max_margin, row_tail = [], [], [], 0
        for rec in recs:
            best = max(rec["splade_pos"])
            m = [s - best for s in rec["splade_neg"]]
            tail_cnt.append(sum(x > a.tail_margin for x in m)); above_cnt.append(sum(x > 0 for x in m))
            max_margin.append(max(m)); row_tail += tail_cnt[-1] > 0
        s = {"rows": len(rows), "seconds": round(time.time() - t0),
             "splade_pos": pct([x for rec in recs for x in rec["splade_pos"]]),
             "splade_neg": pct([x for rec in recs for x in rec["splade_neg"]]),
             "splade_max_neg_minus_best_pos": pct(max_margin),
             "tail_negatives_per_row_mean": round(float(np.mean(tail_cnt)), 3),
             "rows_with_tail_share": round(row_tail / len(rows), 4),
             "negatives_above_positive_share": round(float(np.sum(above_cnt)) / sum(len(rec["splade_neg"]) for rec in recs), 4),
             "query_active_dims_mean": round(float(np.mean([rec["q_dims"] for rec in recs])), 1)}
        if teacher is not None:
            v3p = [x for rec in recs for x in rec["v3_pos"]]
            g = [(gg, vv) for rec in recs for gg, vv in zip(rec["gate"], rec["v3_pos"]) if gg is not None]
            s["v3_pos"] = pct(v3p)
            if ref_thr:
                s["v3_pos_below_reference"] = {k: round(float(np.mean(np.asarray(v3p) < t)), 4) for k, t in ref_thr.items() if t is not None}
            if len(g) > 2:
                ga, va = np.asarray([x[0] for x in g]), np.asarray([x[1] for x in g])
                s["gate_vs_v3_pearson"] = round(float(np.corrcoef(ga, va)[0, 1]), 3)
                s["gate_pos"] = pct(ga.tolist())
            fn = tot = rows_fn = 0
            v3n_all = []
            for rec in recs:
                if "v3_neg" not in rec:
                    continue
                weakest = min(rec["v3_pos"])
                bad = sum(x >= weakest for x in rec["v3_neg"])
                fn += bad; tot += len(rec["v3_neg"]); rows_fn += bad > 0
                v3n_all.extend(rec["v3_neg"])
            if tot:
                s["v3_neg"] = pct(v3n_all)
                s["v3_neg_at_or_above_weakest_pos_share"] = round(fn / tot, 4)
                s["rows_with_v3_false_negative_share"] = round(rows_fn / len([r for r in recs if "v3_neg" in r]), 4)
                s["v3_pos_minus_max_v3_neg"] = pct([min(rec["v3_pos"]) - max(rec["v3_neg"]) for rec in recs if "v3_neg" in rec])
        summary["files"][path.stem] = s
        with (a.per_row_dir / f"{path.stem}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{path.stem}: {json.dumps(s, ensure_ascii=False)}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
