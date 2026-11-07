from copy import deepcopy
from pathlib import Path
import runpy


MODULE = runpy.run_path(str(Path(__file__).parents[1] / "experiments/polish-splade-2026-10/scripts/prepare_mlm_p1.py"))
select_row = MODULE["select_row"]
pair_key = MODULE["pair_key"]


def example():
    return {
        "query_id": 7, "query": "question", "pos": ["low", "equal", "high"],
        "pos_scores": [22, 23.5, 25], "pos_id": ["a", "b", "c"],
        "pos_is_synthetic": [False, False, True],
        "translation_source": {"pos": ["x", "y", "z"], "neg": ["unchanged"]},
        "neg": [f"negative-{i}" for i in range(10)], "neg_scores": list(range(10)),
    }


def test_strict_gate_precedes_llm_and_preserves_parallel_fields_and_negatives():
    row = example()
    original = deepcopy(row)
    judgments = {pair_key("msmarco", 7, "low"): 1, pair_key("msmarco", 7, "equal"): 8}
    result = select_row(row, "msmarco", judgments)
    assert result["pos"] == ["high"]
    assert result["pos_scores_stronger_reranker"] == [25]
    assert result["pos_is_synthetic"] == [True]
    assert result["pos_id"] == ["c"]
    assert result["translation_source"]["pos"] == ["z"]
    assert result["neg"] == original["neg"]
    assert result["neg_scores"] == original["neg_scores"]
    assert result["translation_source"]["neg"] == original["translation_source"]["neg"]
    assert row == original


def test_any_retained_positive_rejection_vetoes_the_whole_row():
    row = example()
    row["pos_scores"] = [26, 27, 28]
    assert select_row(row, "msmarco", {pair_key("msmarco", "7", "high"): 8}) is None


def test_reranker_backup_takes_precedence_over_teacher_scores():
    row = example()
    row["pos_scores_stronger_reranker"] = [26, 20, 20]
    row["pos_scores"] = [0.1, 0.9, 0.8]
    result = select_row(row, "msmarco", {})
    assert result["pos"] == ["low"]
    assert result["pos_scores"] == [0.1]


def test_judgment_cannot_leak_across_corpora_or_different_positive_texts():
    row = example()
    judgments = {pair_key("squadv2", 7, "high"): 1, pair_key("msmarco", 7, "other"): 1}
    assert select_row(row, "msmarco", judgments)["p1_positive_judgments"] == ["unjudged"]
