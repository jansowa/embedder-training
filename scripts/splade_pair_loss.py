#!/usr/bin/env python
"""Measure how hard a training file is for a SPLADE model, the way training sees it.

Replays the training objective without updating anything: batches of
--batch-size queries, each with pos[0] and the first --negatives negatives, scored
by dot product against every passage in the batch (in-batch negatives included),
cross-entropy with scale 1.0, exactly as SparseMultipleNegativesRankingLoss does.

Reports the mean loss, the mean margin pos - max(neg), the fraction of queries
whose positive is already ranked first, and the active dimensions. A file whose
loss under the base model is far below FiQA's is "too easy": its ranking gradient
vanishes within a few hundred steps and the sparsity regulariser takes over,
which is the collapse seen on trivia / NQ / FEVER.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch


def load_rows(path: Path, n: int, seed: int, negatives: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("query") and r.get("pos") and len(r.get("neg", [])) >= negatives:
                rows.append({"query": r["query"], "pos": r["pos"][0], "neg": r["neg"][:negatives],
                             "query_id": str(r.get("query_id", i)), "line": i})
    random.Random(seed).shuffle(rows)
    return rows[:n] if n > 0 else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/base-splade-wrapped")
    ap.add_argument("--input", required=True, type=Path, action="append", help="JSONL file(s); repeatable")
    ap.add_argument("--n", type=int, default=400, help="rows per file after shuffling; 0 = all rows")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--negatives", type=int, default=6)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--per-row-dir", type=Path, default=None,
                    help="write <dir>/<input stem>.jsonl with one record per row: the in-batch loss the "
                         "training objective sees, plus loss_own/margin_own computed against the row's own "
                         "negatives only (batch-independent, use this to rank rows by difficulty)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from sentence_transformers.sparse_encoder import SparseEncoder

    model = SparseEncoder(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length
    model.eval()

    def encode(texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            return model.encode(texts, batch_size=16, convert_to_sparse_tensor=False, convert_to_tensor=True).float()

    report = {}
    for path in args.input:
        rows = load_rows(path, args.n, args.seed, args.negatives)
        losses, margins, top1, qdims, ddims = [], [], 0, [], []
        per_row = []
        # keep a trailing partial batch when every row is wanted (per-row mode)
        n_use = len(rows) if args.n <= 0 else len(rows) - len(rows) % args.batch_size
        for b in range(0, n_use, args.batch_size):
            batch = rows[b : b + args.batch_size]
            B, K = len(batch), args.negatives
            q = encode([r["query"] for r in batch])
            docs = [r["pos"] for r in batch] + [n for r in batch for n in r["neg"]]
            d = encode(docs)
            scores = q @ d.T  # (B, B + B*negatives)
            target = torch.arange(B, device=scores.device)
            row_loss = torch.nn.functional.cross_entropy(scores, target, reduction="none")
            losses.append(row_loss.mean().item())
            for i in range(B):
                pos = scores[i, i].item()
                others = torch.cat([scores[i, :i], scores[i, i + 1 :]])
                margins.append(pos - others.max().item())
                top1 += int(scores[i].argmax().item() == i)
                # batch-independent view: the positive against the row's own negatives only
                own = torch.cat([scores[i, i : i + 1], scores[i, B + i * K : B + (i + 1) * K]])
                loss_own = torch.nn.functional.cross_entropy(own.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=own.device)).item()
                margin_own = pos - own[1:].max().item() if K > 0 else float("nan")
                per_row.append({
                    "query_id": batch[i]["query_id"], "line": batch[i]["line"],
                    "loss_inbatch": round(row_loss[i].item(), 5), "margin_inbatch": round(margins[-1], 4),
                    "loss_own": round(loss_own, 5), "margin_own": round(margin_own, 4),
                    "pos_score": round(pos, 4),
                    "query_active_dims": int((q[i] > 0).sum().item()), "pos_active_dims": int((d[i] > 0).sum().item()),
                })
            qdims.extend((q > 0).sum(dim=1).tolist())
            ddims.extend((d[:B] > 0).sum(dim=1).tolist())
        if args.per_row_dir is not None:
            args.per_row_dir.mkdir(parents=True, exist_ok=True)
            out = args.per_row_dir / f"{path.stem}.jsonl"
            with out.open("w", encoding="utf-8") as fh:
                for rec in sorted(per_row, key=lambda r: r["line"]):
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"per-row records -> {out} ({len(per_row)} rows)")
        n_q = len(margins)
        rep = {
            "rows": n_q,
            "mean_loss": sum(losses) / len(losses),
            "mean_margin": sum(margins) / n_q,
            "frac_pos_ranked_first": top1 / n_q,
            "query_active_dims_mean": sum(qdims) / len(qdims),
            "doc_active_dims_mean": sum(ddims) / len(ddims),
        }
        report[str(path)] = rep
        print(f"{path}: loss={rep['mean_loss']:.3f} margin={rep['mean_margin']:.2f} top1={rep['frac_pos_ranked_first']:.2f} "
              f"qdims={rep['query_active_dims_mean']:.0f} ddims={rep['doc_active_dims_mean']:.0f} (n={n_q})")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"model": args.model, "batch_size": args.batch_size,
                                                "negatives": args.negatives, "results": report}, indent=2))


if __name__ == "__main__":
    main()
