#!/usr/bin/env python
"""Turn verified synthetic pairs plus the chunk pool into the three parquet files
that ~/work/speakleash/mine-hard-negatives expects: queries (id, text),
corpus (id, text), relevant (query_id, document_id). String ids throughout, as in
that project's nfcorpus converter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True, help="JSONL with chunk_id, query")
    ap.add_argument("--pool", type=Path, default=Path("datasets/synth/chunks_pool.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/synth/mine"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = {}
    for line in args.pool.open(encoding="utf-8"):
        r = json.loads(line)
        corpus[r["chunk_id"]] = r["text"]
    queries, relevant = [], []
    missing = 0
    for i, line in enumerate(args.pairs.open(encoding="utf-8")):
        r = json.loads(line)
        if r["chunk_id"] not in corpus:
            missing += 1
            continue
        qid = f"q-{i:05d}"
        queries.append({"id": qid, "text": r["query"]})
        relevant.append({"query_id": qid, "document_id": r["chunk_id"]})
    schema_qc = pa.schema([("id", pa.string()), ("text", pa.string())])
    pq.write_table(pa.Table.from_pylist(queries, schema=schema_qc), args.out_dir / "queries.parquet")
    pq.write_table(pa.Table.from_pylist([{"id": k, "text": v} for k, v in corpus.items()], schema=schema_qc),
                   args.out_dir / "corpus.parquet")
    pq.write_table(pa.Table.from_pylist(relevant, schema=pa.schema([("query_id", pa.string()), ("document_id", pa.string())])),
                   args.out_dir / "relevant.parquet")
    print(f"queries {len(queries)}, corpus {len(corpus)}, relevant {len(relevant)}, pairs without chunk {missing}")


if __name__ == "__main__":
    main()
