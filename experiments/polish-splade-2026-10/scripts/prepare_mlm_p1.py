"""Build the judged P1 pool on CPU; preserve every negative and its metadata."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterator


GOOD_POLISH = {"natural", "awkward_but_clear"}


def pair_key(family: str, query_id: Any, passage: str) -> tuple[str, str, bytes]:
    return family, str(query_id), hashlib.sha256(passage.encode("utf-8")).digest()


def rejection_mask(verdict: dict[str, Any]) -> int:
    return (
        int(int(verdict["relevance"]) < 3)
        | (2 if verdict["query_polish"] not in GOOD_POLISH else 0)
        | (4 if verdict["passage_polish"] not in GOOD_POLISH else 0)
    )


def read_rows(path: Path, fingerprints: dict[str, Any]) -> Iterator[dict[str, Any]]:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            digest.update(line)
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed during the build: {path}")
    fingerprints[str(path.resolve())] = {"sha256": digest.hexdigest(), "bytes": after.st_size}


def load_judgments(annotations: Path, old_rejected: Path, fingerprints: dict[str, Any]):
    judgments: dict[tuple[str, str, bytes], int] = {}
    for stem, pair_file in (
        ("pairs_msmarco_full", "pairs_msmarco_full.unique.jsonl"),
        ("pairs_v1", "pairs_v1.jsonl"),
    ):
        # Preserve vetoes even if multiple journal entries exist for one pair.
        masks: dict[str, int] = {}
        for row in read_rows(annotations / f"{stem}.journal.jsonl", fingerprints):
            verdict = row.get("record", row.get("verdict"))
            if verdict and "relevance" in verdict:
                masks[row["key"]] = masks.get(row["key"], 0) | rejection_mask(verdict)
        for row in read_rows(annotations / pair_file, fingerprints):
            source = row["source"]
            if row["side"] != "pos" or not any(s in source for s in ("msmarco", "squadv2")):
                continue
            if row["pair_id"] not in masks:
                continue
            family = "msmarco" if "msmarco" in source else "squadv2"
            key = pair_key(family, row["query_id"], row["passage"])
            judgments[key] = judgments.get(key, 0) | masks[row["pair_id"]]
    for row in read_rows(old_rejected, fingerprints):
        for passage in row["pos"]:
            key = pair_key("msmarco", row["query_id"], passage)
            judgments[key] = judgments.get(key, 0) | 8
    return judgments


def select_row(row: dict[str, Any], family: str, judgments: dict) -> dict[str, Any] | None:
    positives = row["pos"]
    scores = row.get("pos_scores_stronger_reranker", row.get("pos_scores"))
    if not isinstance(scores, list) or len(scores) != len(positives):
        raise ValueError(f"Unaligned reranker scores: {family}:{row.get('query_id')}")
    keep = [i for i, value in enumerate(scores) if value is not None and float(value) > 23.5]
    if keep:
        keep += [i for i, value in enumerate(scores) if value is None]
    if not keep or not row.get("query") or not row.get("neg"):
        return None
    keys = [pair_key(family, row["query_id"], positives[i]) for i in keep]
    if any(judgments.get(key, 0) for key in keys):
        return None
    if len(row["neg"]) != 10:
        raise ValueError(f"Expected exactly ten unchanged negatives: {family}:{row['query_id']}")
    result = dict(row)
    for field, values in row.items():
        positive_field = field == "pos" or field.startswith("pos_") or field.startswith("original_pos_")
        if positive_field and isinstance(values, list):
            if len(values) != len(positives):
                raise ValueError(f"Unaligned positive field {field}: {family}:{row['query_id']}")
            result[field] = [values[i] for i in keep]
    if isinstance(row.get("translation_source"), dict):
        result["translation_source"] = dict(row["translation_source"])
        values = row["translation_source"].get("pos")
        if isinstance(values, list):
            if len(values) != len(positives):
                raise ValueError("Unaligned translation_source.pos")
            result["translation_source"]["pos"] = [values[i] for i in keep]
    result["pos_scores_stronger_reranker"] = [scores[i] for i in keep]
    result["p1_source"] = family
    result["p1_source_query_id"] = row["query_id"]
    result["p1_positive_judgments"] = ["accepted" if key in judgments else "unjudged" for key in keys]
    result["query_id"] = f"{family}:{row['query_id']}"
    if result.get("pos_scores") and all(v is not None for v in result["pos_scores"]):
        mean = sum(result["pos_scores"]) / len(result["pos_scores"])
        result["mean_pos_score"] = mean
        if "mean_neg_score" in result:
            result["difference_between_mean_scores"] = mean - result["mean_neg_score"]
        if "max_neg_score" in result:
            result["difference_between_max_scores"] = max(result["pos_scores"]) - result["max_neg_score"]
    return result


def encode_row(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ab-output", type=Path, required=True)
    parser.add_argument("--ab-rows", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-rows", type=int, default=266404)
    parser.add_argument("--expected-examples", type=int, default=331172)
    args = parser.parse_args()
    started = time.monotonic()
    for target in (args.output, args.ab_output):
        if target.exists() or target.with_suffix(target.suffix + ".partial").exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage("/mnt/c").free < 15_000_000_000:
        raise RuntimeError("C: must have at least 15 GB free before preparing P1")
    fingerprints: dict[str, Any] = {}
    judgments = load_judgments(args.annotations_root, Path("datasets/judged/msmarco-rejected-20000.jsonl"), fingerprints)
    print(f"Loaded {len(judgments)} distinct judged pairs", flush=True)
    rng = random.Random(args.seed)
    reservoir: list[tuple[int, dict[str, Any]]] = []
    seen = set()
    counts: dict[str, Counter] = {}
    digest = hashlib.sha256()
    total = examples = 0
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    with partial.open("xb") as output:
        for family, source in (
            ("msmarco", Path("dataset-full-msmarco-pl/train_pl.jsonl")),
            ("squadv2", Path("datasets/arms/squadv2-v2-all.jsonl")),
        ):
            counts[family] = Counter()
            for original in read_rows(source, fingerprints):
                counts[family]["input_rows"] += 1
                row = select_row(original, family, judgments)
                if row is None:
                    continue
                identity = (row["query_id"], tuple(sorted(hashlib.sha256(t.encode()).digest() for t in row["pos"])))
                if identity in seen:
                    counts[family]["duplicates"] += 1
                    continue
                seen.add(identity)
                assert row["neg"] == original["neg"]
                for field in original:
                    if field.startswith("neg_") or field.startswith("original_neg_"):
                        assert row[field] == original[field], field
                data = encode_row(row)
                output.write(data)
                digest.update(data)
                total += 1
                examples += len(row["pos"])
                counts[family]["rows"] += 1
                counts[family]["examples"] += len(row["pos"])
                if len(reservoir) < args.ab_rows:
                    reservoir.append((total, row))
                else:
                    index = rng.randrange(total)
                    if index < args.ab_rows:
                        reservoir[index] = (total, row)
            print(f"{family}: {dict(counts[family])}", flush=True)
    if (total, examples) != (args.expected_rows, args.expected_examples):
        raise RuntimeError(f"Counts differ from reviewed inventory: {total} rows, {examples} examples; retained {partial}")
    ab_digest = hashlib.sha256()
    ab_examples = 0
    ab_partial = args.ab_output.with_suffix(args.ab_output.suffix + ".partial")
    with ab_partial.open("xb") as output:
        for _, row in sorted(reservoir, key=lambda item: item[0]):
            data = encode_row(row)
            output.write(data)
            ab_digest.update(data)
            ab_examples += len(row["pos"])
    manifest = {
        "rows": total, "examples": examples, "negatives_per_row": 10,
        "sha256": digest.hexdigest(), "bytes": partial.stat().st_size,
        "sources": {key: dict(value) for key, value in counts.items()},
        "inputs": fingerprints, "seed": args.seed,
        "rules": "positive reranker >23.5, then whole-row veto from any positive LLM rejection, then dedup query + positive set; negatives unchanged",
        "llm_gate": "relevance=3 and query_polish/passage_polish natural or awkward_but_clear; older rejected subset also vetoes",
        "ab_subset": {"path": str(args.ab_output), "rows": len(reservoir), "examples": ab_examples, "sha256": ab_digest.hexdigest()},
        "seconds": round(time.monotonic() - started, 2),
    }
    if shutil.disk_usage("/mnt/c").free < 10_000_000_000:
        raise RuntimeError("C: below 10 GB; partial outputs retained")
    partial.replace(args.output)
    ab_partial.replace(args.ab_output)
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.ab_output.with_suffix(args.ab_output.suffix + ".manifest.json").write_text(json.dumps(manifest["ab_subset"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
