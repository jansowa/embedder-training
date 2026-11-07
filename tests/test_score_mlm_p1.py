from copy import deepcopy
from pathlib import Path
import json
import runpy

import pytest


MODULE = runpy.run_path(str(Path(__file__).parents[1] / "experiments/polish-splade-2026-10/scripts/score_mlm_p1.py"))
ensemble_row = MODULE["ensemble_row"]


def example():
    return {"query_id": "msmarco:7", "query": "q", "pos": ["p"], "neg": ["n"],
            "pos_scores": [20.0], "neg_scores": [8.0], "teacher_model": "soup-v12"}


def test_ensemble_means_already_scaled_scores_without_multiplying_scale_again():
    sparse = example()
    dense = deepcopy(sparse)
    dense.update(pos_scores=[40.0], neg_scores=[4.0], dense_scale=100.0, teacher_model="dense")
    row = ensemble_row(sparse, dense)
    assert row["pos_scores"] == [30.0]
    assert row["neg_scores"] == [6.0]


def test_matching_ids_and_lengths_do_not_hide_changed_passage_text():
    sparse = example()
    dense = deepcopy(sparse)
    dense["neg"] = ["different"]
    with pytest.raises(ValueError, match="text/order"):
        ensemble_row(sparse, dense)


def test_ensemble_rejects_truncated_or_nonfinite_score_arrays():
    sparse = example()
    dense = deepcopy(sparse)
    dense["pos_scores"] = []
    with pytest.raises(ValueError, match="Unaligned"):
        ensemble_row(sparse, dense)
    dense["pos_scores"] = [float("nan")]
    with pytest.raises(ValueError, match="Non-finite"):
        ensemble_row(sparse, dense)


def test_combine_extracts_full_mnrl_and_ab_subsets_without_rescoring(tmp_path):
    rows = []
    for i in range(3):
        row = example()
        row["query_id"] = f"msmarco:{i}"
        rows.append(row)
    def write(name, values):
        path = tmp_path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in values))
        return path
    sparse = write("sparse.jsonl", rows)
    dense = write("dense.jsonl", rows)
    Path(str(dense) + ".teacher.json").write_text(json.dumps({"scale": 100, "rows": 3}))
    ab_source = write("ab.jsonl", rows[:1])
    mnrl_source = write("mnrl.jsonl", rows[:2])
    (tmp_path / "p1_mnrl.ids").write_text("msmarco:0\nmsmarco:1\n")
    output, ab_output, mnrl_output = [tmp_path / name for name in ("ensemble.jsonl", "ab-scored.jsonl", "mnrl-scored.jsonl")]
    manifest = {"rows": 3, "examples": 3, "ab_subset": {"rows": 1}, "sha256": "pool-hash"}
    MODULE["combine"](sparse, dense, output, ab_output, manifest, ab_source, mnrl_source, mnrl_output)
    assert len(output.read_text().splitlines()) == 3
    assert len(ab_output.read_text().splitlines()) == 1
    assert [json.loads(line)["query_id"] for line in mnrl_output.read_text().splitlines()] == ["msmarco:0", "msmarco:1"]
    side = json.loads(Path(str(output) + ".teacher.json").read_text())
    assert side["dense_scale"] == 100 and side["training_score_scale"] == 1


def test_combine_refuses_silently_truncating_an_incomplete_teacher_file(tmp_path):
    row = example()
    sparse, dense, ab, mnrl = [tmp_path / name for name in ("sparse.jsonl", "dense.jsonl", "ab.jsonl", "mnrl.jsonl")]
    line = json.dumps(row) + "\n"
    sparse.write_text(line * 2)
    for path in (dense, ab, mnrl):
        path.write_text(line)
    Path(str(dense) + ".teacher.json").write_text(json.dumps({"scale": 1, "rows": 2}))
    (tmp_path / "p1_mnrl.ids").write_text(row["query_id"] + "\n")
    output = tmp_path / "ensemble.jsonl"
    with pytest.raises(ValueError, match="different numbers"):
        MODULE["combine"](sparse, dense, output, tmp_path / "ab-scored.jsonl", {"rows": 2, "examples": 2, "ab_subset": {"rows": 1}}, ab, mnrl, tmp_path / "mnrl-scored.jsonl")
    assert not output.exists()
