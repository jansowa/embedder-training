#!/usr/bin/env python
"""Student-mined negatives: the passages a SPLADE model itself ranks highest for a query.

For the gated FiQA-PL queries (inputs of datasets/arms/fiqa-v3-mine: queries, corpus,
relevant and the roberta-v3 scores of the positives) the script encodes the corpus and
the queries with a SparseEncoder, takes the top --top-k non-positive passages per query
by dot product and writes them as negative_candidates.parquet in the schema of
find_negatives.py of ~/work/speakleash/mine-hard-negatives. That project's final rerank
(roberta-v3, two percentile thresholds) and export then decide which of them are safe
negatives: the candidate score is the SPLADE score and every candidate is "selected",
so the adaptive final rerank scores the student's hardest candidates first.

--extra-corpus adds a second passage file (jsonl with id + contents, e.g. the PIRB
fiqa-pl corpus) with --extra-id-prefix on its ids. --smoke N keeps the first N queries
and --corpus-limit passages (their positives first).

Outputs in --out-dir: queries.parquet, corpus.parquet, relevant.parquet,
relevant_with_candidate_score.parquet (positive_candidate_ranking = SPLADE score of the
positive), relevant_with_score.parquet (the source's positive_ranking = v3 score, with
positive_candidate_ranking replaced by the SPLADE score), negative_candidates.parquet
and splade_mining_report.json.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", default="datasets/arms/fiqa-v3-mine")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="models/base-splade-wrapped")
    parser.add_argument("--extra-corpus", default=None, help="jsonl with id + contents (or text) per line")
    parser.add_argument("--extra-id-prefix", default="pirb:")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1024, help="passages encoded and scored per step")
    parser.add_argument("--pooling-chunk", type=int, default=8,
                        help="tokens per SpladePooling chunk (bounds the batch x tokens x vocab logits; 8 GB GPU)")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--smoke", type=int, default=0, help="keep only the first N queries")
    parser.add_argument("--corpus-limit", type=int, default=0, help="keep only this many passages (positives first)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_extra_corpus(path: Path, prefix: str) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("contents") or row.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            ids.append(f"{prefix}{row['id']}")
            texts.append(text)
    return ids, texts


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    return {
        "p10": float(np.quantile(arr, 0.1)),
        "median": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "mean": float(arr.mean()),
    }


def main() -> None:
    args = parse_args()
    import torch
    from sentence_transformers import SparseEncoder

    started = time.time()
    src = Path(args.source_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    queries = pd.read_parquet(src / "queries.parquet")
    corpus = pd.read_parquet(src / "corpus.parquet")
    relevant = pd.read_parquet(src / "relevant.parquet")
    relevant_scored = pd.read_parquet(src / "relevant_with_score.parquet")
    for frame, columns in ((queries, ["id"]), (corpus, ["id"]), (relevant, ["query_id", "document_id"]),
                           (relevant_scored, ["query_id", "document_id"])):
        for column in columns:
            frame[column] = frame[column].astype(str)
    if args.smoke > 0:
        queries = queries.head(args.smoke).reset_index(drop=True)
        keep = set(queries["id"])
        relevant = relevant[relevant["query_id"].isin(keep)].reset_index(drop=True)
        relevant_scored = relevant_scored[relevant_scored["query_id"].isin(keep)].reset_index(drop=True)

    doc_ids = corpus["id"].tolist()
    doc_texts = corpus["text"].tolist()
    doc_source = ["own"] * len(doc_ids)
    n_extra = 0
    if args.extra_corpus:
        own_texts = set(doc_texts)
        own_ids = set(doc_ids)
        extra_ids, extra_texts = load_extra_corpus(Path(args.extra_corpus), args.extra_id_prefix)
        for doc_id, text in zip(extra_ids, extra_texts):
            if doc_id in own_ids or text in own_texts:
                continue
            doc_ids.append(doc_id)
            doc_texts.append(text)
            doc_source.append("extra")
            n_extra += 1
    positive_ids = set(relevant["document_id"])
    if args.corpus_limit and len(doc_ids) > args.corpus_limit:
        rng = random.Random(args.seed)
        keep_idx = [i for i, doc_id in enumerate(doc_ids) if doc_id in positive_ids]
        rest = [i for i, doc_id in enumerate(doc_ids) if doc_id not in positive_ids]
        rng.shuffle(rest)
        keep_idx = sorted(keep_idx + rest[: max(0, args.corpus_limit - len(keep_idx))])
        doc_ids = [doc_ids[i] for i in keep_idx]
        doc_texts = [doc_texts[i] for i in keep_idx]
        doc_source = [doc_source[i] for i in keep_idx]
        n_extra = sum(1 for s in doc_source if s == "extra")
    doc_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    query_index = {query_id: i for i, query_id in enumerate(queries["id"])}

    positives_of_query: dict[int, set[int]] = defaultdict(set)
    queries_of_doc: dict[int, list[int]] = defaultdict(list)
    for query_id, document_id in zip(relevant["query_id"], relevant["document_id"]):
        if query_id not in query_index or document_id not in doc_index:
            continue
        q, d = query_index[query_id], doc_index[document_id]
        positives_of_query[q].add(d)
        queries_of_doc[d].append(q)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SparseEncoder(args.model, device=device)
    model.max_seq_length = args.max_seq_length
    for module in model.modules():
        if type(module).__name__ == "SpladePooling" and hasattr(module, "chunk_size"):
            module.chunk_size = args.pooling_chunk
    encode_kwargs = {
        "batch_size": args.batch_size,
        "convert_to_tensor": True,
        "convert_to_sparse_tensor": False,
        "show_progress_bar": False,
    }
    with torch.no_grad():
        query_embeddings = model.encode(queries["text"].tolist(), **encode_kwargs).float()
    n_queries = len(queries)
    top_k = min(args.top_k, max(1, len(doc_ids) - 1))
    top_scores = torch.full((n_queries, top_k), float("-inf"), device=query_embeddings.device)
    top_index = torch.full((n_queries, top_k), -1, dtype=torch.long, device=query_embeddings.device)
    positive_scores: dict[tuple[int, int], float] = {}

    for start in range(0, len(doc_ids), args.chunk_size):
        end = min(start + args.chunk_size, len(doc_ids))
        with torch.no_grad():
            doc_embeddings = model.encode(doc_texts[start:end], **encode_kwargs).float()
            scores = query_embeddings @ doc_embeddings.T  # (queries, chunk)
        for d in range(start, end):
            for q in queries_of_doc.get(d, []):
                positive_scores[(q, d)] = float(scores[q, d - start])
                scores[q, d - start] = float("-inf")
        chunk_index = torch.arange(start, end, device=scores.device).unsqueeze(0).expand(n_queries, -1)
        merged_scores = torch.cat([top_scores, scores], dim=1)
        merged_index = torch.cat([top_index, chunk_index], dim=1)
        top_scores, order = torch.topk(merged_scores, k=top_k, dim=1)
        top_index = torch.gather(merged_index, 1, order)
        del doc_embeddings, scores, merged_scores, merged_index
    if device == "cuda":
        torch.cuda.empty_cache()

    top_scores_np = top_scores.cpu().numpy()
    top_index_np = top_index.cpu().numpy()
    positive_values = np.sort(np.array(list(positive_scores.values()), dtype=float))

    def percentile(score: float) -> float:
        if positive_values.size == 0:
            return float("nan")
        return float(np.searchsorted(positive_values, score, side="right") / positive_values.size)

    rows = []
    extra_in_top10 = extra_in_topk = total_top10 = total_topk = 0
    for q in range(n_queries):
        query_id = queries["id"].iloc[q]
        rank = 0
        for score, d in zip(top_scores_np[q], top_index_np[q]):
            if d < 0 or not np.isfinite(score):
                continue
            rank += 1
            is_extra = doc_source[d] == "extra"
            total_topk += 1
            extra_in_topk += is_extra
            if rank <= 10:
                total_top10 += 1
                extra_in_top10 += is_extra
            rows.append({
                "query_id": query_id,
                "document_id": doc_ids[d],
                "candidate_ranking": float(score),
                "candidate_percentile": percentile(float(score)),
                "candidate_selected": True,
                "retrieval_rank": rank,
                "retrieval_offset": 0.0,
                "retrieval_score": float(score),
                "retrieval_source": f"splade:{doc_source[d]}",
            })
    candidates = pd.DataFrame(rows)
    candidates.to_parquet(out / "negative_candidates.parquet", index=False)

    queries[["id", "text"]].to_parquet(out / "queries.parquet", index=False)
    pd.DataFrame({"id": doc_ids, "text": doc_texts}).to_parquet(out / "corpus.parquet", index=False)
    relevant[["query_id", "document_id"]].to_parquet(out / "relevant.parquet", index=False)
    positive_frame = pd.DataFrame([
        {"query_id": queries["id"].iloc[q], "document_id": doc_ids[d], "positive_candidate_ranking": score}
        for (q, d), score in positive_scores.items()
    ])
    positive_frame.to_parquet(out / "relevant_with_candidate_score.parquet", index=False)
    scored = relevant_scored.drop(columns=["positive_candidate_ranking"], errors="ignore").merge(
        positive_frame, on=["query_id", "document_id"], how="left"
    )
    scored = scored[["query_id", "document_id", "positive_candidate_ranking", "positive_ranking"]]
    scored.to_parquet(out / "relevant_with_score.parquet", index=False)

    # how the student sees its positives: candidates above the positive, per (query, positive) pair
    above = []
    for (q, d), score in positive_scores.items():
        above.append(int((top_scores_np[q] > score).sum()))
    above_arr = np.array(above) if above else np.zeros(0)
    rank1 = [float(top_scores_np[q][0]) for q in range(n_queries) if np.isfinite(top_scores_np[q][0])]
    rank10 = [float(top_scores_np[q][9]) for q in range(n_queries) if top_k >= 10 and np.isfinite(top_scores_np[q][9])]
    report = {
        "model": args.model,
        "source_dir": str(src),
        "queries": n_queries,
        "corpus_own": len(doc_ids) - n_extra,
        "corpus_extra": n_extra,
        "top_k": top_k,
        "candidate_rows": int(len(candidates)),
        "positive_pairs": len(positive_scores),
        "positive_splade_scores": quantiles(positive_values),
        "candidate_score_at_rank1": quantiles(rank1),
        "candidate_score_at_rank10": quantiles(rank10),
        "positive_is_first_share": float((above_arr == 0).mean()) if above_arr.size else None,
        "positive_in_top10_share": float((above_arr < 10).mean()) if above_arr.size else None,
        "positive_beyond_topk_share": float((above_arr >= top_k).mean()) if above_arr.size else None,
        "candidates_above_positive": quantiles(above_arr) if above_arr.size else {},
        "extra_share_top10": (extra_in_top10 / total_top10) if total_top10 else None,
        "extra_share_topk": (extra_in_topk / total_topk) if total_topk else None,
        "elapsed_s": round(time.time() - started, 1),
    }
    (out / "splade_mining_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
