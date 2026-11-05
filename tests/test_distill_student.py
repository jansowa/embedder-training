"""distill.student: layer / FFN pruning keeps the teacher's weights, the ALBERT conversion loads and calibrates."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from distill_fixtures import make_teacher, sample_texts  # noqa: E402

from distill import student as st  # noqa: E402
from distill.splade import encode_pooled  # noqa: E402


@pytest.fixture(scope="module")
def teacher_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("teacher")
    make_teacher(path)
    return path


def pooled(model, tokenizer, texts):
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        return encode_pooled(model, enc["input_ids"], enc["attention_mask"])


def test_spec_validation():
    spec = st.StudentSpec(keep_layers=[0, 1])
    spec.validate(3, 16, 24, 16)
    with pytest.raises(st.StudentSpecError):
        st.StudentSpec(keep_layers=[1, 0]).validate(3, 16, 24, 16)
    with pytest.raises(st.StudentSpecError):
        st.StudentSpec(keep_layers=[0, 3]).validate(3, 16, 24, 16)
    with pytest.raises(st.StudentSpecError):
        st.StudentSpec(keep_layers=[0], intermediate_size=25).validate(3, 16, 24, 16)
    with pytest.raises(st.StudentSpecError):
        st.StudentSpec(keep_layers=[0], hidden_size=8).validate(3, 16, 24, 16)
    with pytest.raises(st.StudentSpecError):
        st.StudentSpec(keep_layers=[0], hidden_size=8, fresh=True).validate(3, 16, 24, 16)
    assert st.parse_layers("0,2, 5") == [0, 2, 5]


def test_keeping_every_layer_reproduces_the_teacher(teacher_dir):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_dir)
    student, report = st.build_student(teacher_dir, st.StudentSpec(keep_layers=[0, 1, 2]))
    assert report.architecture == "roberta"
    assert report.params["total"] == report.teacher_params["total"]
    texts = sample_texts(5)
    assert torch.allclose(pooled(student, tokenizer, texts), pooled(teacher, tokenizer, texts), atol=1e-5)
    assert all(v == pytest.approx(1.0) for v in report.ffn_importance_kept.values())


def test_layer_subset_and_ffn_pruning_copy_the_right_weights(teacher_dir):
    from transformers import AutoModelForMaskedLM

    teacher = AutoModelForMaskedLM.from_pretrained(teacher_dir)
    student, report = st.build_student(teacher_dir, st.StudentSpec(keep_layers=[0, 2], intermediate_size=12))
    assert student.config.num_hidden_layers == 2
    assert student.config.intermediate_size == 12
    src = teacher.roberta.encoder.layer[2]
    dst = student.roberta.encoder.layer[1]
    assert torch.equal(dst.attention.self.query.weight, src.attention.self.query.weight)
    scores = st.ffn_importance_weight(src)
    idx = torch.topk(scores, 12).indices.sort().values
    assert torch.equal(dst.intermediate.dense.weight, src.intermediate.dense.weight[idx])
    assert torch.equal(dst.output.dense.weight, src.output.dense.weight[:, idx])
    assert 0 < report.ffn_importance_kept["2"] <= 1.0
    assert report.params["total"] < report.teacher_params["total"]
    assert report.macs_per_token["ratio"] < 1.0
    # the decoder stays tied to the input embedding
    assert student.lm_head.decoder.weight.data_ptr() == student.roberta.embeddings.word_embeddings.weight.data_ptr()


def test_activation_importance_uses_sample_texts(teacher_dir):
    student, report = st.build_student(
        teacher_dir, st.StudentSpec(keep_layers=[0, 1], intermediate_size=8, importance="activation"), sample_texts(8)
    )
    assert student.config.intermediate_size == 8
    with pytest.raises(st.StudentSpecError):
        st.build_student(teacher_dir, st.StudentSpec(keep_layers=[0], importance="activation"))


def test_albert_conversion_at_full_rank_reproduces_the_teacher(teacher_dir):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_dir)
    student, report = st.build_student(teacher_dir, st.StudentSpec(keep_layers=[0, 1, 2], embedding_rank=16, calibrate=False))
    assert report.calibration["embedding_subspace_energy"] == pytest.approx(1.0)
    texts = sample_texts(6, seed=3)
    assert torch.allclose(pooled(student, tokenizer, texts), pooled(teacher, tokenizer, texts), atol=1e-4)


def test_albert_conversion_loads_and_calibration_reduces_the_error(teacher_dir, tmp_path):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_dir)
    spec = st.StudentSpec(keep_layers=[0, 1, 2], embedding_rank=8, calibration_steps=150)
    student, report = st.build_student(teacher_dir, spec, sample_texts(16))
    assert report.architecture == "albert"
    assert student.config.model_type == "albert"
    assert student.config.embedding_size == 8
    assert student.config.num_hidden_groups == 3
    assert 0.0 < report.calibration["embedding_subspace_energy"] < 1.0
    joint = report.calibration["joint_calibration"]
    assert joint["embedding_relative_error_before"] < 0.5  # the projection starts close
    assert joint["embedding_relative_error_after"] <= joint["embedding_relative_error_before"] + 1e-6
    assert joint["splade_relative_error_after"] <= joint["splade_relative_error_before"] + 1e-6
    assert joint["tokens"] > 0
    out = tmp_path / "albert-student"
    st.save_student(student, teacher_dir, out, report)
    assert (out / "tokenizer.json").is_file()
    assert (out / "modules.json").is_file()
    assert (out / "1_SpladePooling" / "config.json").is_file()
    manifest = json.loads((out / "student.json").read_text())
    assert manifest["architecture"] == "albert"
    reloaded = AutoModelForMaskedLM.from_pretrained(out)
    assert reloaded.config.model_type == "albert"
    assert reloaded.get_output_embeddings().weight.data_ptr() == reloaded.get_input_embeddings().weight.data_ptr()
    texts = sample_texts(6, seed=3)
    s = pooled(reloaded, tokenizer, texts)
    t = pooled(teacher, tokenizer, texts)
    assert s.shape == t.shape
    assert torch.isfinite(s).all()


def test_untied_head_keeps_its_own_decoder(teacher_dir, tmp_path):
    from transformers import AutoModelForMaskedLM

    spec = st.StudentSpec(keep_layers=[0, 1], embedding_rank=8, untie_head=True, calibration_steps=20)
    student, report = st.build_student(teacher_dir, spec, sample_texts(8))
    assert report.calibration["untie_head"] is True
    assert 0.0 < report.calibration["head_subspace_energy"] <= 1.0
    assert student.predictions.decoder.weight.data_ptr() != student.albert.embeddings.word_embeddings.weight.data_ptr()
    out = tmp_path / "untied"
    st.save_student(student, teacher_dir, out, report)
    reloaded = AutoModelForMaskedLM.from_pretrained(out)
    assert reloaded.get_output_embeddings().weight.data_ptr() != reloaded.get_input_embeddings().weight.data_ptr()
    assert torch.allclose(reloaded.predictions.decoder.weight, student.predictions.decoder.weight)
    with pytest.raises(st.StudentSpecError):
        st.StudentSpec(keep_layers=[0], untie_head=True).validate(3, 16, 24, 16)


def test_fresh_body_keeps_only_the_vocabulary(teacher_dir):
    from transformers import AutoModelForMaskedLM

    teacher = AutoModelForMaskedLM.from_pretrained(teacher_dir)
    student, report = st.build_student(teacher_dir, st.StudentSpec(keep_layers=[0, 1], embedding_rank=8, hidden_size=8, fresh=True))
    assert student.config.hidden_size == 8
    assert student.config.num_hidden_layers == 2
    e = teacher.roberta.embeddings.word_embeddings.weight
    q, _ = st.embedding_subspace(e, teacher.roberta.embeddings.position_embeddings.weight[2:], 8)
    assert torch.allclose(student.albert.embeddings.word_embeddings.weight, e @ q, atol=1e-5)
    assert "embedding_calibration" not in report.calibration
    enc_ids = torch.tensor([[0, 4, 5, 2]])
    with torch.no_grad():
        assert torch.isfinite(student(input_ids=enc_ids, attention_mask=torch.ones_like(enc_ids)).logits).all()


def test_cli_builds_a_student_folder(teacher_dir, tmp_path):
    out = tmp_path / "student"
    assert st.main(["--teacher", str(teacher_dir), "--out", str(out), "--keep-layers", "0,2", "--intermediate-size", "12"]) == 0
    assert (out / "model.safetensors").is_file()
    assert (out / "student.json").is_file()
    assert st.main(["--teacher", str(teacher_dir), "--out", str(out), "--keep-layers", "0"]) == 2  # refuses to overwrite


def test_read_texts_handles_lists_and_limits(tmp_path):
    path = tmp_path / "texts.jsonl"
    path.write_text('{"text": "a b"}\n{"text": ["c d", "e f"]}\n{"other": 1}\nnot json\n', encoding="utf-8")
    assert sorted(st.read_texts(path, "text", 10, 0)) == ["a b", "c d", "e f"]
    assert len(st.read_texts(path, "text", 2, 0)) == 2


def test_head_layer_reproduces_the_teacher_up_to_the_rank(teacher_dir, tmp_path):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_dir)
    texts = sample_texts(24, seed=5)
    full, report = st.build_student(teacher_dir, st.StudentSpec(keep_layers=[0, 1, 2], embedding_rank=16, head_layer=True), texts)
    assert report.calibration["head_layer"] is True and report.calibration["untie_head"] is True
    assert full.config.num_hidden_layers == 4 and full.config.intermediate_size == 32
    assert report.calibration["head_subspace_energy"] == pytest.approx(1.0, abs=1e-5)
    t = pooled(teacher, tokenizer, texts[:6])
    s_full = pooled(full, tokenizer, texts[:6])
    # exact body + head transform; at full rank only the rank-space LayerNorm (per-token scale) remains
    cos_full = torch.nn.functional.cosine_similarity(s_full, t, dim=-1).mean()
    assert cos_full > 0.95
    low, report = st.build_student(teacher_dir, st.StudentSpec(keep_layers=[0, 1, 2], embedding_rank=8, head_layer=True), texts)
    s_low = pooled(low, tokenizer, texts[:6])
    assert torch.isfinite(s_low).all()
    assert 0.0 < report.calibration["head_subspace_energy"] < 1.0
    out = tmp_path / "head-layer"
    st.save_student(low, teacher_dir, out, report)
    reloaded = AutoModelForMaskedLM.from_pretrained(out)
    assert reloaded.config.num_hidden_layers == 4
    assert torch.allclose(pooled(reloaded, tokenizer, texts[:6]), s_low, atol=1e-5)
    with pytest.raises(st.StudentSpecError):
        st.build_student(teacher_dir, st.StudentSpec(keep_layers=[0], embedding_rank=8, head_layer=True))


def test_weighted_subspace_prefers_the_frequent_tokens(teacher_dir, tmp_path):
    from transformers import AutoModelForMaskedLM

    teacher = AutoModelForMaskedLM.from_pretrained(teacher_dir)
    e = teacher.roberta.embeddings.word_embeddings.weight.detach()
    vocab = e.shape[0]
    weights = torch.zeros(vocab)
    weights[:4] = 1.0  # only four tokens ever occur
    q, energy = st.embedding_subspace(e, None, 3, weights)
    centred = e[:4] - e[:4].mean(dim=-1, keepdim=True)
    assert energy > 0.99  # three directions span four centred rows almost entirely
    assert torch.allclose(q.T @ q, torch.eye(3), atol=1e-5)
    counts = tmp_path / "counts.json"
    counts.write_text(json.dumps({"counts": [5] * vocab}))
    loaded = st.load_token_weights(counts)
    assert loaded.shape == (vocab,) and float(loaded.min()) > 5.0
    with pytest.raises(st.StudentSpecError):
        st.StudentSpec(keep_layers=[0], embedding_rank=8, token_weights=torch.ones(vocab + 1)).validate(3, 16, 24, vocab)
