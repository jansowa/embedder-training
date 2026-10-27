#!/usr/bin/env python
"""Build the three parquet files that ~/work/speakleash/mine-hard-negatives expects
(queries, corpus, relevant) from the gated FiQA-PL rows in datasets/arms/fiqa-2073.jsonl.

The corpus is the union of every positive and every negative passage of those rows
(our own translation, 11.6k passages); the PIRB fiqa-pl corpus is a different and
weaker translation, so it is deliberately left out (27.10 evening). Every positive
of a query (including the synthetic ones already in the rows) becomes a relevant pair.
String ids throughout, as in that project's nfcorpus converter and in
scripts/synth_build_mining_inputs.py.

--smoke N keeps the first N queries and a corpus of their passages plus 2000 random
others, for a quick end-to-end check of the pipeline wiring.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=Path("datasets/arms/fiqa-2073.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/arms/fiqa-v3-mine"))
    ap.add_argument("--smoke", type=int, default=0, help="keep only the first N queries (and a small corpus)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.rows.open(encoding="utf-8") if line.strip()]
    corpus: dict[str, str] = {}
    conflicts = 0
    for r in rows:
        for pid, text in list(zip(r["pos_id"], r["pos"])) + list(zip(r["neg_id"], r["neg"])):
            pid = str(pid)
            if pid in corpus and corpus[pid] != text:
                conflicts += 1
            corpus.setdefault(pid, text)
    if conflicts:
        raise SystemExit(f"{conflicts} passage ids map to more than one text; refusing to build")

    if args.smoke:
        rows = rows[: args.smoke]
        keep = {str(i) for r in rows for i in r["pos_id"] + r["neg_id"]}
        others = [k for k in corpus if k not in keep]
        random.seed(42)
        keep.update(random.sample(others, min(2000, len(others))))
        corpus = {k: v for k, v in corpus.items() if k in keep}

    queries = [{"id": str(r["query_id"]), "text": r["query"]} for r in rows]
    relevant = [{"query_id": str(r["query_id"]), "document_id": str(pid)} for r in rows for pid in r["pos_id"]]
    if len({q["id"] for q in queries}) != len(queries):
        raise SystemExit("duplicate query ids in the rows")

    schema_qc = pa.schema([("id", pa.string()), ("text", pa.string())])
    pq.write_table(pa.Table.from_pylist(queries, schema=schema_qc), args.out_dir / "queries.parquet")
    pq.write_table(
        pa.Table.from_pylist([{"id": k, "text": v} for k, v in corpus.items()], schema=schema_qc),
        args.out_dir / "corpus.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(relevant, schema=pa.schema([("query_id", pa.string()), ("document_id", pa.string())])),
        args.out_dir / "relevant.parquet",
    )
    print(f"queries {len(queries)}, corpus {len(corpus)}, relevant {len(relevant)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
