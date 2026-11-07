import json
from pathlib import Path
import runpy
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "experiments/polish-splade-2026-10/scripts/prepare_mlm_p1_relaxed.py"
MODULE = runpy.run_path(str(SCRIPT))
Candidate = MODULE["Candidate"]


def test_language_rules_allow_untranslated_passage_but_not_query():
    assert Candidate(0, 0, 2, "natural", "untranslated_parts", 25).eligible(2)
    assert not Candidate(0, 0, 2, "untranslated_parts", "natural", 25).eligible(1)
    assert not Candidate(0, 0, 3, "natural", "broken", 25).eligible(1)


def test_highest_relevance_precedes_gate_score_and_language_is_checked_after_selection():
    good = Candidate(0, 0, 2, "natural", "natural", 30)
    high = Candidate(0, 1, 3, "natural", "broken", 25)
    assert max([good, high], key=lambda c: c.rank()) == high
    assert not high.eligible(1)
    tied_clean = Candidate(0, 2, 3, "natural", "natural", 24)
    assert max([high, tied_clean], key=lambda c: c.rank()) == tied_clean


def test_builder_writes_nested_id_sets_and_one_positive_per_query(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for stem in ("pairs_msmarco_full", "pairs_v1"):
        (annotations / f"{stem}.journal.jsonl").write_text("")
    (annotations / "pairs_v1.jsonl").write_text("")
    pairs = []
    journal = []
    for qid, index, text, relevance in [(1, 0, "weak", 1), (1, 1, "best", 2), (2, 0, "only", 1)]:
        pair_id = f"msmarco-full:{qid}:pos{index}"
        pairs.append({"pair_id": pair_id, "source": "msmarco-full", "side": "pos", "query_id": qid, "passage": text})
        journal.append({"key": pair_id, "record": {"relevance": relevance, "query_polish": "natural", "passage_polish": "natural"}})
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(annotations / "pairs_msmarco_full.unique.jsonl", pairs)
    write(annotations / "pairs_msmarco_full.journal.jsonl", journal)
    negatives = [f"n{i}" for i in range(10)]
    write(tmp_path / "dataset-full-msmarco-pl/train_pl.jsonl", [
        {"query_id": 1, "query": "q1", "pos": ["weak", "best"], "pos_scores": [30, 25], "neg": negatives},
        {"query_id": 2, "query": "q2", "pos": ["only"], "pos_scores": [25], "neg": negatives},
    ])
    write(tmp_path / "datasets/arms/squadv2-v2-all.jsonl", [
        {"query_id": 1, "query": "s1", "pos": ["unknown"], "pos_scores_stronger_reranker": [25], "neg": negatives},
    ])
    result = subprocess.run([sys.executable, str(SCRIPT), "--annotations-root", str(annotations)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    out = tmp_path / "datasets/arms"
    kl = set((out / "p1_kl.ids").read_text().splitlines())
    mnrl = set((out / "p1_mnrl.ids").read_text().splitlines())
    assert kl == {"msmarco:1", "msmarco:2", "squadv2:1"}
    assert mnrl == {"msmarco:1", "squadv2:1"}
    rows = [json.loads(line) for line in (out / "mlm-p1-kl-v2.jsonl").read_text().splitlines()]
    assert rows[0]["pos"] == ["best"]
    assert all(len(row["pos"]) == 1 and row["neg"] == negatives for row in rows)
