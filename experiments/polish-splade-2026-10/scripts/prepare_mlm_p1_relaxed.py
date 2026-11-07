"""Select one best positive per query for the relevance>=1 and >=2 P1 arms."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_mlm_p1 import encode_row, pair_key, read_rows


@dataclass(frozen=True)
class Candidate:
    line: int
    index: int
    relevance: int | None
    query_polish: str | None
    passage_polish: str | None
    gate_score: float | None

    def language_ok(self) -> bool:
        return self.query_polish not in {"broken", "nonsense", "untranslated_parts"} and self.passage_polish not in {"broken", "nonsense"}

    def rank(self) -> tuple:
        return (
            -1 if self.relevance is None else self.relevance,
            self.language_ok(),
            float("-inf") if self.gate_score is None else self.gate_score,
            -self.line, -self.index,
        )

    def eligible(self, minimum: int) -> bool:
        return self.language_ok() and (self.relevance is None or self.relevance >= minimum)


def load_verdicts(annotations: Path, fingerprints: dict):
    verdicts = {}
    conflicts = 0
    for stem, pair_file in (("pairs_msmarco_full", "pairs_msmarco_full.unique.jsonl"), ("pairs_v1", "pairs_v1.jsonl")):
        journal = {}
        for row in read_rows(annotations / f"{stem}.journal.jsonl", fingerprints):
            record = row.get("record", row.get("verdict"))
            if record and "relevance" in record:
                journal[row["key"]] = (int(record["relevance"]), record["query_polish"], record["passage_polish"])
        for row in read_rows(annotations / pair_file, fingerprints):
            if row["side"] != "pos" or row["pair_id"] not in journal:
                continue
            if stem == "pairs_msmarco_full":
                family = "msmarco"
            elif "squadv2" in row["source"]:
                family = "squadv2"
            else:
                continue
            key = pair_key(family, row["query_id"], row["passage"])
            value = journal[row["pair_id"]]
            if key in verdicts and verdicts[key] != value:
                conflicts += 1
                # Identical query/passages can appear more than once in the source.
                # Retain the more conservative Qwen relevance/language verdict.
                old = verdicts[key]
                value = (min(old[0], value[0]),
                         next((v for v in (old[1], value[1]) if v in {"broken", "nonsense", "untranslated_parts"}), value[1]),
                         next((v for v in (old[2], value[2]) if v in {"broken", "nonsense"}), value[2]))
            verdicts[key] = value
        print(f"Loaded {stem}", flush=True)
    return verdicts, conflicts


def candidates(row: dict, family: str, line: int, verdicts: dict):
    scores = row.get("pos_scores_stronger_reranker", row.get("pos_scores"))
    if scores is not None and len(scores) != len(row["pos"]):
        raise ValueError("Reranker scores and positives are not aligned")
    for index, text in enumerate(row["pos"]):
        score = None if scores is None else scores[index]
        if score is not None and float(score) <= 23.5:
            continue
        verdict = verdicts.get(pair_key(family, row["query_id"], text))
        if verdict is None and family == "msmarco":
            raise ValueError(f"Missing full Qwen judgment: {row['query_id']} positive {index}")
        relevance, query_polish, passage_polish = verdict or (None, None, None)
        yield Candidate(line, index, relevance, query_polish, passage_polish, score)


def selected_row(original: dict, family: str, candidate: Candidate) -> dict:
    row = dict(original)
    n = len(original["pos"])
    for field, values in original.items():
        if (field == "pos" or field.startswith("pos_") or field.startswith("original_pos_")) and isinstance(values, list):
            if len(values) != n:
                raise ValueError(f"Unaligned positive field {field}")
            row[field] = [values[candidate.index]]
    if isinstance(original.get("translation_source"), dict):
        row["translation_source"] = dict(original["translation_source"])
        values = original["translation_source"].get("pos")
        if isinstance(values, list):
            if len(values) != n:
                raise ValueError("Unaligned positive translation_source")
            row["translation_source"]["pos"] = [values[candidate.index]]
    if candidate.gate_score is not None:
        row["pos_scores_stronger_reranker"] = [candidate.gate_score]
    row.update(query_id=f"{family}:{original['query_id']}", p1_source=family,
               p1_source_query_id=original["query_id"], p1_source_positive_index=candidate.index,
               p1_qwen={"relevance": candidate.relevance, "query_polish": candidate.query_polish, "passage_polish": candidate.passage_polish})
    if row.get("pos_scores") and row["pos_scores"][0] is not None:
        score = row["pos_scores"][0]
        row["mean_pos_score"] = score
        if "mean_neg_score" in row:
            row["difference_between_mean_scores"] = score - row["mean_neg_score"]
        if "max_neg_score" in row:
            row["difference_between_max_scores"] = score - row["max_neg_score"]
    assert row["neg"] == original["neg"] and len(row["neg"]) == 10
    for field in original:
        if field.startswith("neg_") or field.startswith("original_neg_"):
            assert row[field] == original[field]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/arms"))
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ["mlm-p1-kl-v2.jsonl", "mlm-p1-mnrl-v2.jsonl", "mlm-p1-mnrl-v2-ab40k.jsonl", "p1_kl.ids", "p1_mnrl.ids"]
    for name in names:
        if (args.output_dir / name).exists():
            raise FileExistsError(args.output_dir / name)
    if shutil.disk_usage(args.output_dir).free < 18_000_000_000:
        raise RuntimeError("Need at least 18 GB free before creating the revised pools")
    fingerprints = {}
    verdicts, conflicts = load_verdicts(args.annotations_root, fingerprints)
    sources = [("msmarco", Path("dataset-full-msmarco-pl/train_pl.jsonl")), ("squadv2", Path("datasets/arms/squadv2-v2-all.jsonl"))]
    counts = {}
    output_handles = {name: (args.output_dir / (name + ".partial")).open("xb") for name in names}
    digests = {name: hashlib.sha256() for name in names}
    rng = random.Random(42)
    reservoir = []
    kl_ids, mnrl_ids = set(), set()
    def write(name, data):
        output_handles[name].write(data)
        digests[name].update(data)
    try:
        for family, source in sources:
            best = {}
            stats = Counter()
            for line, row in enumerate(read_rows(source, fingerprints)):
                stats["input_rows"] += 1
                if not row.get("query") or not row.get("pos") or not row.get("neg"):
                    continue
                qid = str(row["query_id"])
                for candidate in candidates(row, family, line, verdicts):
                    previous = best.get(qid)
                    if previous is None or candidate.rank() > previous.rank():
                        best[qid] = candidate
            stats["unique_queries_after_reranker"] = len(best)
            for line, row in enumerate(read_rows(source, fingerprints)):
                candidate = best.get(str(row["query_id"]))
                if candidate is None or candidate.line != line or not candidate.eligible(1):
                    continue
                selected = selected_row(row, family, candidate)
                data = encode_row(selected)
                qid = selected["query_id"]
                assert qid not in kl_ids
                kl_ids.add(qid)
                write("mlm-p1-kl-v2.jsonl", data)
                write("p1_kl.ids", (qid + "\n").encode())
                stats["kl"] += 1
                stats["kl_unjudged"] += candidate.relevance is None
                if not candidate.eligible(2):
                    continue
                mnrl_ids.add(qid)
                write("mlm-p1-mnrl-v2.jsonl", data)
                write("p1_mnrl.ids", (qid + "\n").encode())
                stats["mnrl"] += 1
                stats["mnrl_unjudged"] += candidate.relevance is None
                if len(reservoir) < 40000:
                    reservoir.append((len(mnrl_ids), data))
                else:
                    index = rng.randrange(len(mnrl_ids))
                    if index < 40000:
                        reservoir[index] = (len(mnrl_ids), data)
            counts[family] = dict(stats)
            print(f"{family}: {dict(stats)}", flush=True)
        assert mnrl_ids <= kl_ids
        for _, data in sorted(reservoir):
            write("mlm-p1-mnrl-v2-ab40k.jsonl", data)
    finally:
        for handle in output_handles.values():
            handle.close()
    manifest = {"rules_version": 2, "sources": counts, "inputs": fingerprints,
                "duplicate_pair_verdict_conflicts": conflicts, "negatives_per_row": 10,
                "selection": "reranker >23.5 where present; highest relevance per query, language-safe tie then reranker then source order; check language on chosen positive; no old rejected veto",
                "language": "passage excludes broken/nonsense; query excludes broken/nonsense/untranslated_parts",
                "unjudged_squadv2": "keep source positive; when any Qwen-labeled candidate exists, choose highest labeled relevance",
                "seconds": round(time.monotonic() - started, 2), "seed": 42,
                "files": {name: {"sha256": digests[name].hexdigest(), "bytes": (args.output_dir / (name + ".partial")).stat().st_size} for name in names}}
    for name in names:
        (args.output_dir / (name + ".partial")).replace(args.output_dir / name)
    for kind, ids, name in [("kl", kl_ids, names[0]), ("mnrl", mnrl_ids, names[1])]:
        detail = dict(manifest, rows=len(ids), examples=len(ids), sha256=digests[name].hexdigest(),
                      ab_subset={"path": str(args.output_dir / names[2]), "rows": len(reservoir), "examples": len(reservoir), "sha256": digests[names[2]].hexdigest()})
        (args.output_dir / (name + ".manifest.json")).write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "p1-v2.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kl_rows": len(kl_ids), "mnrl_rows": len(mnrl_ids), "ab_rows": len(reservoir), "seconds": manifest["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
