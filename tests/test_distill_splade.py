"""distill.splade: pooling equals the dense reference, losses vanish on identical models (CPU, tiny models)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from distill_fixtures import make_teacher, sample_texts  # noqa: E402

from distill.splade import (  # noqa: E402
    active_dims,
    encode_hidden,
    flops_loss,
    pooled_distill_loss,
    pooled_vector,
    pooled_vector_dense,
    split_mlm_head,
    teacher_topk_targets,
    token_distill_loss,
    topk_kl,
)


@pytest.fixture(scope="module")
def teacher(tmp_path_factory):
    from transformers import AutoTokenizer

    path = tmp_path_factory.mktemp("teacher")
    model = make_teacher(path)
    return model, AutoTokenizer.from_pretrained(path)


def encode(tokenizer, texts):
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def test_pooled_vector_matches_dense_reference(teacher):
    model, tokenizer = teacher
    input_ids, mask = encode(tokenizer, sample_texts(6))
    with torch.no_grad():
        hidden = encode_hidden(model, input_ids, mask)
        logits = model(input_ids=input_ids, attention_mask=mask).logits
        dense = pooled_vector_dense(logits, mask)
        for chunk in (0, 2):
            assert torch.allclose(pooled_vector(model, hidden, mask, chunk), dense, atol=1e-5)
    assert (dense > 0).any()
    assert active_dims(dense).shape == (6,)


def test_split_head_reproduces_logits(teacher):
    model, tokenizer = teacher
    input_ids, mask = encode(tokenizer, sample_texts(3))
    transform, decoder = split_mlm_head(model)
    with torch.no_grad():
        hidden = encode_hidden(model, input_ids, mask)
        assert torch.allclose(decoder(transform(hidden)), model(input_ids=input_ids, attention_mask=mask).logits, atol=1e-5)


def test_token_losses_vanish_for_the_same_model_and_grow_for_a_perturbed_one(teacher):
    model, tokenizer = teacher
    input_ids, mask = encode(tokenizer, sample_texts(4))
    with torch.no_grad():
        hidden = encode_hidden(model, input_ids, mask)
    for kind in ("kl", "mse"):
        same = token_distill_loss(model, hidden, model, hidden, mask, kind=kind, top_k=5, chunk=3)
        assert float(same) == pytest.approx(0.0, abs=1e-5)
        other = token_distill_loss(model, hidden + 0.5 * torch.randn_like(hidden), model, hidden, mask, kind=kind, top_k=5, chunk=3)
        assert float(other) > 1e-3


def test_token_loss_backward_through_chunks_matches_unchunked(teacher):
    model, tokenizer = teacher
    input_ids, mask = encode(tokenizer, sample_texts(4))
    with torch.no_grad():
        teacher_hidden = encode_hidden(model, input_ids, mask)
    grads = []
    for chunk in (0, 2):
        model.zero_grad(set_to_none=True)
        student_hidden = encode_hidden(model, input_ids, mask)
        student_hidden = student_hidden + 0.3 * torch.randn(student_hidden.shape, generator=torch.Generator().manual_seed(1))
        loss = token_distill_loss(model, student_hidden, model, teacher_hidden, mask, kind="kl", top_k=4, chunk=chunk)
        loss.backward()
        grads.append(model.lm_head.dense.weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_topk_kl_is_zero_for_identical_logits_and_positive_otherwise():
    logits = torch.randn(2, 3, 9)
    indices, probs, rest = teacher_topk_targets(logits, 2.0, 4)
    assert torch.allclose(probs.sum(-1) + rest, torch.ones(2, 3), atol=1e-5)
    assert float(topk_kl(logits, indices, probs, rest, 2.0).abs().max()) < 1e-5
    assert float(topk_kl(logits + torch.randn_like(logits), indices, probs, rest, 2.0).mean()) > 0


def test_pooled_losses_and_flops():
    t = torch.rand(4, 10) * (torch.rand(4, 10) > 0.5)
    for kind in ("mse", "kl"):
        assert float(pooled_distill_loss(t, t, kind)) == pytest.approx(0.0, abs=1e-6)
        assert float(pooled_distill_loss(t * 0.5 + 0.1, t, kind)) > 0
    assert float(flops_loss(t)) > 0
    with pytest.raises(ValueError):
        pooled_distill_loss(t, t, "cosine")
