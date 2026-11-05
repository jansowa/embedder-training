"""Tiny RoBERTa SPLADE teacher and a word-level tokenizer for the distill tests (CPU, milliseconds)."""

from __future__ import annotations

import json
from pathlib import Path

WORDS = ["ala", "ma", "kota", "pies", "szczeka", "na", "koty", "i", "psy", "dom", "jest", "duży"]
SPECIALS = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}


def make_tokenizer(out_dir: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = dict(SPECIALS)
    for word in WORDS:
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    fast.save_pretrained(out_dir)
    return fast


def make_teacher(out_dir: Path, layers: int = 3, hidden: int = 16, intermediate: int = 24, seed: int = 0):
    import torch
    from transformers import RobertaConfig, RobertaForMaskedLM

    torch.manual_seed(seed)
    config = RobertaConfig(
        vocab_size=len(SPECIALS) + len(WORDS),
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=2,
        intermediate_size=intermediate,
        max_position_embeddings=40,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        layer_norm_eps=1e-5,
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config)
    with torch.no_grad():  # a spread of logits so that relu / max pooling has something to do
        model.lm_head.bias.normal_(0.0, 2.0)
        model.lm_head.decoder.weight.mul_(4.0)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    make_tokenizer(out_dir)
    (out_dir / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.sparse_encoder.models.MLMTransformer"},
                {"idx": 1, "name": "1", "path": "1_SpladePooling", "type": "sentence_transformers.sparse_encoder.models.SpladePooling"},
            ]
        )
    )
    (out_dir / "1_SpladePooling").mkdir(exist_ok=True)
    (out_dir / "1_SpladePooling" / "config.json").write_text(json.dumps({"pooling_strategy": "max", "activation_function": "relu"}))
    (out_dir / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 32, "do_lower_case": False}))
    return model


def sample_texts(n: int = 24, seed: int = 0) -> list[str]:
    import random

    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 9))) for _ in range(n)]


def write_corpus(path: Path, n: int = 24, seed: int = 0) -> list[str]:
    texts = sample_texts(n, seed)
    with path.open("w", encoding="utf-8") as handle:
        for i, text in enumerate(texts):
            handle.write(json.dumps({"text": text, "kind": "query" if i % 4 == 0 else "passage", "source": "tiny"}) + "\n")
    return texts
