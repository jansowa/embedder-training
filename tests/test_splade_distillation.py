"""SPLADE distillation options of the sentence-transformers backend: the KL loss labels,
the teacher score scale and its calibration against the untrained student."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_margin_fit_recovers_the_scale_factor():
    from training.backends.sentence_transformers_backend import _fit_teacher_score_scale

    fit = _fit_teacher_score_scale([2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0])

    assert fit["alpha"] == pytest.approx(2.0)
    assert fit["pearson"] == pytest.approx(1.0)
    assert fit["pairs"] == 4
    assert fit["student_margin_mean"] == pytest.approx(5.0)
    assert fit["teacher_margin_mean"] == pytest.approx(2.5)


def test_margin_fit_keeps_scale_one_without_teacher_signal():
    from training.backends.sentence_transformers_backend import _fit_teacher_score_scale

    fit = _fit_teacher_score_scale([1.0, 2.0], [0.0, 0.0])

    assert fit["alpha"] == 1.0
    assert fit["pearson"] == 0.0


def test_score_scale_calibration_samples_rows_evenly_and_fits_student_margins():
    from training.backends.sentence_transformers_backend import _calibrate_score_scale

    rows = [
        {"anchor": f"q{i}", "positive": f"p{i}", "negative_1": f"n{i}a", "negative_2": f"n{i}b", "label": [1.0, 2.0]}
        for i in range(4)
    ]
    seen = {}

    def score_fn(queries, passages):
        seen["queries"] = queries
        seen["passages"] = passages
        return [[10.0, 7.0, 4.0] for _ in queries]  # student margins 3 and 6 = 3 x teacher

    fit = _calibrate_score_scale(rows, score_fn, sample_rows=2, label_layout="margins")

    assert seen["queries"] == ["q0", "q2"]
    assert seen["passages"] == [["p0", "n0a", "n0b"], ["p2", "n2a", "n2b"]]
    assert fit["alpha"] == pytest.approx(3.0)
    assert fit["sample_rows"] == 2
    assert fit["pairs"] == 4


def test_score_scale_calibration_reads_margins_from_the_scores_layout():
    from training.backends.sentence_transformers_backend import _calibrate_score_scale

    rows = [{"anchor": "q", "positive": "p", "negative_1": "n1", "label": [5.0, 3.0]}]

    fit = _calibrate_score_scale(rows, lambda queries, passages: [[8.0, 4.0]], sample_rows=8, label_layout="scores")

    assert fit["alpha"] == pytest.approx(2.0)
    assert fit["sample_rows"] == 1


def test_score_scale_calibration_rejects_mismatched_margin_counts():
    from training.backends.sentence_transformers_backend import SentenceTransformersConfigError, _calibrate_score_scale

    rows = [{"anchor": "q", "positive": "p", "negative_1": "n1", "label": [1.0, 2.0]}]
    with pytest.raises(SentenceTransformersConfigError):
        _calibrate_score_scale(rows, lambda queries, passages: [[8.0, 4.0]], sample_rows=8, label_layout="margins")


def test_resolve_score_scale_values():
    from training.backends.sentence_transformers_backend import SentenceTransformersConfigError, _resolve_score_scale

    assert _resolve_score_scale({}) is None
    assert _resolve_score_scale({"score_scale": 1}) is None
    assert _resolve_score_scale({"score_scale": "none"}) is None
    assert _resolve_score_scale({"score_scale": "auto"}) == "auto"
    assert _resolve_score_scale({"score_scale": 2.5}) == 2.5
    assert _resolve_score_scale({"score_scale": "0.5"}) == 0.5
    for bad in (0, -1.0, True, "x"):
        with pytest.raises(SentenceTransformersConfigError):
            _resolve_score_scale({"score_scale": bad})


def test_apply_score_scale_handles_lists_and_scalars():
    from training.backends.sentence_transformers_backend import _apply_score_scale

    rows = [{"label": [1.0, 2.0]}, {"label": 3.0}, {"anchor": "no label"}]
    _apply_score_scale(rows, 2.0)

    assert rows == [{"label": [2.0, 4.0]}, {"label": 6.0}, {"anchor": "no label"}]


def test_dataset_loader_scores_layout_keeps_positive_and_negative_scores(tmp_path):
    from training.backends.sentence_transformers_backend import load_flagembedding_jsonl_dataset

    data_file = tmp_path / "dataset.jsonl"
    data_file.write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1", "n2"], "pos_scores": [10.0], "neg_scores": [7.0, 4.0]})
        + "\n",
        encoding="utf-8",
    )
    common = {"negatives_per_query": 2, "query_prefix": "", "passage_prefix": "", "use_score_labels": True}

    raw = load_flagembedding_jsonl_dataset(data_file, score_normalization="none", label_layout="scores", **common)
    scaled = load_flagembedding_jsonl_dataset(
        data_file, score_normalization="per_query_minmax", label_layout="scores", **common
    )

    assert raw[0]["label"] == [10.0, 7.0, 4.0]
    assert scaled[0]["label"] == [1.0, 0.5, 0.0]


def test_loss_name_accepts_kl_aliases():
    from training.backends.sentence_transformers_backend import SentenceTransformersConfigError, _splade_base_loss_name

    assert _splade_base_loss_name({"loss": "kl_div"}) == "sparse_distill_kl_div"
    assert _splade_base_loss_name({"loss": "sparse-distill-kl-div"}) == "sparse_distill_kl_div"
    assert _splade_base_loss_name({"loss": "margin_mse"}) == "sparse_margin_mse"
    with pytest.raises(SentenceTransformersConfigError):
        _splade_base_loss_name({"loss": "sparse_distill"})


def _fake_sparse_stack(calls: dict):
    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            calls["rows"] = rows
            return rows

    class FakeMLMTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            pass

    class FakeSpladePooling:
        def __init__(self, pooling_strategy):
            pass

    class FakeSparseEncoder:
        def __init__(self, modules):
            pass

        def save_pretrained(self, output_path):
            pass

    class FakeArgs:
        def __init__(self, **kwargs):
            pass

    class FakeSparseMarginMSELoss:
        def __init__(self, model):
            calls["base_loss"] = "margin"

    class FakeSparseRankingLoss:
        def __init__(self, model, scale, gather_across_devices):
            calls["base_loss"] = "mnrl"

    class FakeSpladeLoss:
        def __init__(self, model, loss, document_regularizer_weight, query_regularizer_weight):
            pass

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            pass

        def train(self):
            pass

    return (
        FakeDataset,
        FakeSparseEncoder,
        FakeTrainer,
        FakeArgs,
        FakeSparseMarginMSELoss,
        FakeSparseRankingLoss,
        FakeSpladeLoss,
        FakeMLMTransformer,
        FakeSpladePooling,
    )


def _distillation_request(tmp_path, **sentence_transformers):
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1", "n2"], "pos_scores": [10.0], "neg_scores": [7.0, 4.0]})
        + "\n",
        encoding="utf-8",
    )
    return TrainingRequest(
        backend="sentence-transformers",
        training_type="splade",
        config={
            "train_data": str(data_dir),
            "output_dir": str(tmp_path / "out"),
            "sentence_transformers": {"model_name_or_path": "tiny-mlm", "negatives_per_query": 2, **sentence_transformers},
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )


def test_splade_training_builds_kl_loss_with_scores_labels_and_fixed_scale(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend

    calls: dict = {}
    monkeypatch.setattr(sentence_transformers_backend, "_load_sparse_sentence_transformers", lambda: _fake_sparse_stack(calls))

    class FakeKLLoss:
        def __init__(self, model, temperature):
            calls["base_loss"] = ("kl", temperature)

    monkeypatch.setattr(sentence_transformers_backend, "_load_sparse_kl_loss", lambda: FakeKLLoss)
    request = _distillation_request(tmp_path, loss="sparse_distill_kl_div", kl_temperature=3, score_scale=2)

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["base_loss"] == ("kl", 3.0)
    assert calls["rows"][0]["label"] == [20.0, 14.0, 8.0]
    assert json.loads((tmp_path / "out" / "score_scale.json").read_text(encoding="utf-8"))["alpha"] == 2.0


def test_splade_training_calibrates_the_auto_score_scale_on_the_student(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend

    calls: dict = {}
    monkeypatch.setattr(sentence_transformers_backend, "_load_sparse_sentence_transformers", lambda: _fake_sparse_stack(calls))
    monkeypatch.setattr(
        sentence_transformers_backend,
        "_splade_score_fn",
        lambda model, batch_size: (lambda queries, passages: [[16.0, 10.0, 4.0] for _ in queries]),
    )
    request = _distillation_request(tmp_path, loss="sparse_margin_mse", score_scale="auto", score_scale_sample_rows=8)

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["base_loss"] == "margin"
    assert calls["rows"][0]["label"] == [6.0, 12.0]  # teacher margins 3, 6 x alpha 2 (student margins 6, 12)
    fit = json.loads((tmp_path / "out" / "score_scale.json").read_text(encoding="utf-8"))
    assert fit["alpha"] == pytest.approx(2.0)
    assert fit["pearson"] == pytest.approx(1.0)
    assert fit["sample_rows"] == 1


def test_splade_training_rejects_score_scale_without_a_distillation_loss(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend

    calls: dict = {}
    monkeypatch.setattr(sentence_transformers_backend, "_load_sparse_sentence_transformers", lambda: _fake_sparse_stack(calls))
    request = _distillation_request(tmp_path, loss="sparse_multiple_negatives_ranking", score_scale=2)

    with pytest.raises(sentence_transformers_backend.SentenceTransformersConfigError):
        sentence_transformers_backend.run_training(request)


def test_splade_training_without_score_scale_keeps_the_margin_labels(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend

    calls: dict = {}
    monkeypatch.setattr(sentence_transformers_backend, "_load_sparse_sentence_transformers", lambda: _fake_sparse_stack(calls))
    request = _distillation_request(tmp_path, loss="sparse_margin_mse", score_normalization="none")

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["rows"][0]["label"] == [3.0, 6.0]
    assert not (tmp_path / "out" / "score_scale.json").exists()
