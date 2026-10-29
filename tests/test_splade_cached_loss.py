"""The GradCache option of the SPLADE trainer (sentence_transformers.cached_loss / mini_batch_size)."""

from __future__ import annotations

import pytest


def test_cached_loss_is_off_by_default():
    from training.backends.sentence_transformers_backend import _splade_cached_loss_mini_batch

    assert _splade_cached_loss_mini_batch({}) is None
    assert _splade_cached_loss_mini_batch({"cached_loss": False, "mini_batch_size": 4}) is None
    assert _splade_cached_loss_mini_batch({"cached_loss": "no"}) is None


def test_cached_loss_reads_the_mini_batch_size():
    from training.backends.sentence_transformers_backend import _splade_cached_loss_mini_batch

    assert _splade_cached_loss_mini_batch({"cached_loss": True}) == 32
    assert _splade_cached_loss_mini_batch({"cached_loss": "true", "mini_batch_size": "4"}) == 4


@pytest.mark.parametrize("value", [0, -2, "four", None])
def test_cached_loss_rejects_a_bad_mini_batch_size(value):
    from training.backends.sentence_transformers_backend import SentenceTransformersConfigError, _splade_cached_loss_mini_batch

    with pytest.raises(SentenceTransformersConfigError):
        _splade_cached_loss_mini_batch({"cached_loss": True, "mini_batch_size": value})
