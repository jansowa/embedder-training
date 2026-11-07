from copy import deepcopy
from pathlib import Path
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
