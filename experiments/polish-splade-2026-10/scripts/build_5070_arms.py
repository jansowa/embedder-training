#!/usr/bin/env python
"""Training files for the 5070 Ti night queue (scripts/queue_5070.sh).

  datasets/arms/mix3-6219.jsonl              fiqa-2073 + squadv2-2073 + msmarco-clean-2073, shuffled
  datasets/arms/fiqa-plus-squadv2-10k.jsonl  fiqa-2073 + 8000 fresh gate-passing SQuAD-v2 rows
                                             (stronger reranker > 23.5, 10 negatives, not in squadv2-2073)

Rows keep every field of the source files, so the usual dataset filter
(configs/dataset_filters/splade_positive_score_gt_23_50_no_metricx.yaml) applies
unchanged. Deterministic (seed 42); prints row counts and a manifest to
datasets/arms/manifest_5070.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ARMS = Path("datasets/arms")
SQUAD_POOL = Path("cache/huggingface_datasets/speakleash-squadv2-main-train_pl-7f2f680853ed240b/dataset.jsonl")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def gate_ok(r: dict, thr: float = 23.5) -> bool:
    s = r.get("pos_scores_stronger_reranker") or []
    return bool(s) and s[0] is not None and float(s[0]) > thr


def main() -> None:
    rng = random.Random(42)
    fiqa = read_jsonl(ARMS / "fiqa-2073.jsonl")
    squad = read_jsonl(ARMS / "squadv2-2073.jsonl")
    msm = read_jsonl(ARMS / "msmarco-clean-2073.jsonl")
    manifest = {}

    mix3 = fiqa + squad + msm
    rng.shuffle(mix3)
    write_jsonl(ARMS / "mix3-6219.jsonl", mix3)
    manifest["mix3-6219"] = {"rows": len(mix3), "sources": {"fiqa-2073": len(fiqa), "squadv2-2073": len(squad), "msmarco-clean-2073": len(msm)}}
    print(f"mix3-6219: {len(mix3)} rows")

    used = {str(r["query_id"]) for r in squad}
    pool = []
    for line in SQUAD_POOL.open(encoding="utf-8"):
        r = json.loads(line)
        if str(r["query_id"]) in used or not gate_ok(r) or len(r.get("neg", [])) < 10 or not r.get("query") or not r.get("pos"):
            continue
        pool.append(r)
    rng.shuffle(pool)
    fresh = pool[:8000]
    big = fiqa + fresh
    rng.shuffle(big)
    write_jsonl(ARMS / "fiqa-plus-squadv2-10k.jsonl", big)
    manifest["fiqa-plus-squadv2-10k"] = {"rows": len(big), "sources": {"fiqa-2073": len(fiqa), "squadv2-fresh-gated": len(fresh)}, "squad_pool_gated_unused": len(pool)}
    print(f"fiqa-plus-squadv2-10k: {len(big)} rows ({len(fresh)} fresh SQuAD rows out of {len(pool)} gate-passing unused)")
    (ARMS / "manifest_5070.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
