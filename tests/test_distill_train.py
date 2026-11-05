"""distill.data and distill.train_kd: batching covers the corpus, a tiny KD run writes a loadable checkpoint."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from distill_fixtures import make_teacher, write_corpus  # noqa: E402

from distill import student as st  # noqa: E402
from distill.data import TextCorpus, length_bucketed_batches  # noqa: E402
from distill.train_kd import lr_at, main  # noqa: E402


def test_length_bucketed_batches_cover_every_index_once():
    lengths = [i % 17 for i in range(103)]
    batches = length_bucketed_batches(lengths, 8, seed=1, bucket_batches=3)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(103))
    assert max(len(b) for b in batches) == 8
    # inside a batch the lengths are close (sorted within the bucket)
    spreads = [max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches if len(b) == 8]
    assert sum(spreads) / len(spreads) < 8
    assert length_bucketed_batches(lengths, 8, seed=1) == length_bucketed_batches(lengths, 8, seed=1)
    assert length_bucketed_batches(lengths, 8, seed=1) != length_bucketed_batches(lengths, 8, seed=2)


def test_text_corpus_filters_kinds_and_samples(tmp_path):
    path = tmp_path / "corpus.jsonl"
    texts = write_corpus(path, 24)
    corpus = TextCorpus([path])
    assert corpus.texts == texts
    assert corpus.stats.by_kind == {"query": 6, "passage": 18}
    queries = TextCorpus([path], kinds={"query"})
    assert len(queries) == 6
    sampled = TextCorpus([path], max_items=5, seed=3)
    assert len(sampled) == 5
    assert all(t in texts for t in sampled.texts)
    assert sum(len(b) for b in corpus.batches(4, 0, 0)) == 24


def test_lr_schedules():
    assert lr_at(0, 100, 10, "constant", 1.0) == pytest.approx(0.1)
    assert lr_at(9, 100, 10, "constant", 1.0) == pytest.approx(1.0)
    assert lr_at(50, 100, 10, "constant", 1.0) == 1.0
    assert lr_at(100, 100, 10, "linear", 1.0) == 0.0
    assert lr_at(55, 100, 10, "cosine", 1.0) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("losses", [["--token-loss", "kl", "--pooled-loss", "none"], ["--token-loss", "none", "--pooled-loss", "mse", "--flops-weight", "0.01"]])
def test_kd_run_writes_a_loadable_checkpoint(tmp_path, losses):
    from transformers import AutoModelForMaskedLM

    teacher_dir = tmp_path / "teacher"
    make_teacher(teacher_dir)
    student_dir = tmp_path / "student"
    student, report = st.build_student(teacher_dir, st.StudentSpec(keep_layers=[0, 2], intermediate_size=12))
    st.save_student(student, teacher_dir, student_dir, report)
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(corpus, 20)
    out = tmp_path / "run"
    argv = [
        "--teacher", str(teacher_dir), "--student", str(student_dir), "--corpus", str(corpus), "--out", str(out),
        "--device", "cpu", "--precision", "fp32", "--batch-size", "4", "--max-steps", "3", "--save-every", "2",
        "--log-every", "1", "--warmup-steps", "1", "--fidelity-texts", str(corpus), "--fidelity-n", "6", "--top-k", "4", "--head-chunk", "3",
        "--max-seq-length", "12",
    ] + losses
    assert main(argv) == 0
    assert (out / "final" / "model.safetensors").is_file()
    assert (out / "final" / "tokenizer.json").is_file()
    assert (out / "final" / "1_SpladePooling" / "config.json").is_file()
    assert (out / "step-2" / "kd_state.json").is_file()
    records = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in records if "loss" in r]
    assert steps == [1, 2, 3]
    assert any("fidelity/all/cosine" in r or "fidelity/passage/cosine" in r for r in records)
    summary = json.loads((out / "kd_summary.json").read_text())
    assert summary["steps"] == 3 and summary["examples"] == 12
    reloaded = AutoModelForMaskedLM.from_pretrained(out / "final")
    assert reloaded.config.num_hidden_layers == 2
