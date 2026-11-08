"""Mined pool for the static query encoder: corpus sampling, candidate merge, end-to-end mining and training (CPU, tiny)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from distill_fixtures import make_teacher, write_corpus  # noqa: E402

from distill.static_mine import kind_of_line, main as mine_main, merge_candidates, sample_corpus  # noqa: E402
from distill.static_pool import load_pool  # noqa: E402
from distill.static_query import StaticSparseQueryEncoder, main as static_main  # noqa: E402
from distill.train_static import main as train_main  # noqa: E402


def test_kind_of_line_reads_only_the_kind_field():
    assert kind_of_line(json.dumps({"text": 'a "kind": "query" b', "kind": "passage"})) == "passage"
    assert kind_of_line(json.dumps({"text": "x", "kind": "query"})) == "query"
    assert kind_of_line(json.dumps({"text": "x"})) is None


def test_sample_corpus_is_seeded_and_respects_the_kinds(tmp_path):
    texts = write_corpus(tmp_path / "c.jsonl", n=40)
    queries_all = {t for i, t in enumerate(texts) if i % 4 == 0}
    passages_all = {t for i, t in enumerate(texts) if i % 4 != 0}
    p1, q1 = sample_corpus(tmp_path / "c.jsonl", 10, 5, seed=1)
    p2, q2 = sample_corpus(tmp_path / "c.jsonl", 10, 5, seed=1)
    assert (p1, q1) == (p2, q2)
    assert len(p1) == 10 and len(q1) == 5
    assert set(p1) <= passages_all and set(q1) <= queries_all
    p_all, q_all = sample_corpus(tmp_path / "c.jsonl", 0, 0)
    assert len(p_all) == 30 and len(q_all) == 10


def test_merge_candidates_sorts_the_union_by_the_teacher_and_caps():
    scores = torch.tensor([0.5, 3.0, 1.0, 2.0, 0.1])
    ids, s = merge_candidates(scores, torch.tensor([1, 3]), [torch.tensor([4, 3]), torch.tensor([0])], max_candidates=3)
    assert ids == [1, 3, 0] and s == [3.0, 2.0, 0.5]


def test_mined_pool_trains_with_a_learned_scale(tmp_path):
    make_teacher(tmp_path / "t")
    write_corpus(tmp_path / "c.jsonl", n=120, seed=3)
    idf = tmp_path / "idf"
    assert static_main(["build", "--init", "idf", "--teacher", str(tmp_path / "t"), "--texts", str(tmp_path / "c.jsonl"), "--kinds", "passage", "--out", str(idf)]) == 0
    pool_dir = tmp_path / "pool"
    assert mine_main(["--doc-model", str(tmp_path / "t"), "--corpus", str(tmp_path / "c.jsonl"), "--out", str(pool_dir), "--student", str(idf),
                      "--teacher-top", "4", "--student-top", "3", "--max-candidates", "6", "--max-seq-length", "32", "--search-batch", "7",
                      "--device", "cpu", "--dtype", "fp32", "--min-chars", "1"]) == 0
    pool = load_pool(pool_dir)
    assert pool.meta["kind"] == "mined" and pool.queries is not None and len(pool.queries) == len(pool.rows) == pool.meta["rows"]
    for row in pool.rows:
        assert row["n_pos"] == 1 and 1 <= len(row["docs"]) <= 6 and len(row["scores"]) == len(row["docs"])
        assert row["scores"] == sorted(row["scores"], reverse=True)
    # the stored scores are the teacher's: q_splade . d on the quantized vectors
    q = pool.queries.dense_rows([0])
    d = pool.docs.dense_rows(pool.rows[0]["docs"])
    assert torch.allclose((q @ d.T)[0], torch.tensor(pool.rows[0]["scores"]), atol=1e-3)
    assert 0.0 <= pool.meta["student_recall_of_teacher_top10"]["idf"] <= 1.0
    out = tmp_path / "run"
    assert train_main(["--pool", str(pool_dir), "--init", str(idf), "--out", str(out), "--target", "rows", "--learn-scale", "--max-steps", "4",
                       "--batch-size", "4", "--val-rows", "6", "--eval-every", "2", "--log-every", "1", "--lr", "5e-2", "--warmup-steps", "0",
                       "--device", "cpu"]) == 0
    summary = json.loads((out / "static_summary.json").read_text())
    assert summary["steps"] == 4 and summary["scale_init"] > 0 and summary["scale_final"] > 0
    trained = StaticSparseQueryEncoder.load(out / "final")
    assert not torch.equal(trained.values, StaticSparseQueryEncoder.load(idf).values)
