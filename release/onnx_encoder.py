"""SPLADE vectors from an ONNX export (``release.onnx_splade``) on the CPU, with onnxruntime and a tokenizer only.

An export directory holds ``model.onnx``, the tokenizer files and ``onnx_splade.json`` (see ``META_FILE``).
Two graph kinds exist:

* ``splade``: inputs ``input_ids`` / ``attention_mask``, output ``sparse_vector`` [batch, output_dim], the
  pooled ``max_t log(1 + relu(logits)) * mask`` exactly as PIRB's SpladeEncoder computes it;
* ``mlm``: output ``logits`` [batch, seq, vocab] (the layout sentence-transformers / optimum load as
  ``onnx/model.onnx``); the pooling is then done here in numpy.

No torch import: the tokenizer is the ``tokenizers`` backend read from ``tokenizer.json`` (what
``AutoTokenizer`` wraps for a fast tokenizer; ``transformers`` would pull in torch, ~0.7 GB of RSS), with
``AutoTokenizer`` only as a fallback for checkpoints without ``tokenizer.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

META_FILE = "onnx_splade.json"
MODEL_FILE = "model.onnx"


def read_meta(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / META_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: not an ONNX SPLADE export")
    return json.loads(path.read_text(encoding="utf-8"))


def is_onnx_splade_dir(path: str | Path | None) -> bool:
    return path is not None and (Path(path) / META_FILE).is_file()


def pool_logits(logits: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """SPLADE max pooling of MLM logits [b, t, v] in numpy (same formula as the ``splade`` graph)."""
    weights = np.log1p(np.maximum(logits, 0.0, dtype=np.float32))
    weights *= attention_mask[:, :, None].astype(np.float32)
    return weights.max(axis=1)


def session_options(threads: int | None = None, arena: bool = True, optimization: str = "all"):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = int(threads)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_cpu_mem_arena = bool(arena)
    opts.graph_optimization_level = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[optimization]
    opts.log_severity_level = 3
    return opts


def _token_content(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("content")
    return value


class FastTokenizer:
    """``tokenizer.json`` through the ``tokenizers`` library, padded to the longest row and truncated like
    ``AutoTokenizer(texts, padding="longest", truncation=True, max_length=...)``."""

    def __init__(self, model_dir: Path, max_length: int):
        from tokenizers import Tokenizer

        self.backend = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        config: dict[str, Any] = {}
        for name in ("special_tokens_map.json", "tokenizer_config.json"):
            path = model_dir / name
            if path.is_file():
                config.update(json.loads(path.read_text(encoding="utf-8")))
        pad = _token_content(config.get("pad_token"))
        pad_id = self.backend.token_to_id(pad) if pad else None
        self.padding_side = config.get("padding_side", "right")
        self.backend.no_truncation()
        self.backend.enable_truncation(max_length=max_length, direction=config.get("truncation_side", "right"))
        if pad_id is None:
            pad, pad_id = "<pad>", 0
        self.backend.enable_padding(pad_id=pad_id, pad_token=pad, direction=self.padding_side)

    def __call__(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        encodings = self.backend.encode_batch(list(texts), add_special_tokens=True)
        ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        return ids, mask

    def get_vocab(self) -> dict[str, int]:
        return self.backend.get_vocab(with_added_tokens=True)


class _HFTokenizer:
    def __init__(self, model_dir: Path, max_length: int):
        from transformers import AutoTokenizer

        self.inner = AutoTokenizer.from_pretrained(str(model_dir))
        self.max_length = max_length
        self.padding_side = getattr(self.inner, "padding_side", "right")

    def __call__(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        enc = self.inner(list(texts), padding="longest", truncation=True, add_special_tokens=True,
                         max_length=self.max_length, return_tensors="np")
        return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)

    def get_vocab(self) -> dict[str, int]:
        return self.inner.get_vocab()


def load_tokenizer(model_dir: str | Path, max_length: int, backend: str = "auto"):
    model_dir = Path(model_dir)
    if backend == "transformers" or (backend == "auto" and not (model_dir / "tokenizer.json").is_file()):
        return _HFTokenizer(model_dir, max_length)
    return FastTokenizer(model_dir, max_length)


class OnnxSplade:
    """An ONNX SPLADE encoder: ``encode(texts) -> float32 [n, output_dim]``.

    ``max_batch_tokens`` caps rows x padded length of one session run. The graph materialises the MLM
    logits [rows, seq, vocab] in fp32 (50 k vocab: 200 kB per token), so 4096 tokens is ~0.8 GB per
    intermediate tensor; a batch above the cap is run in slices, each trimmed to its own longest row.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        threads: int | None = None,
        max_length: int | None = None,
        max_batch_tokens: int = 4096,
        arena: bool = True,
        optimization: str = "all",
        model_file: str | None = None,
        tokenizer_backend: str = "auto",
    ):
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.meta = read_meta(self.model_dir)
        self.model_path = self.model_dir / (model_file or self.meta.get("model_file", MODEL_FILE))
        self.session = ort.InferenceSession(
            str(self.model_path), session_options(threads, arena, optimization), providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.graph = self.meta.get("graph", "splade")
        self.max_length = int(max_length or self.meta.get("max_length") or 512)
        self.tokenizer = load_tokenizer(self.model_dir, self.max_length, tokenizer_backend)
        self.output_dim = int(self.meta["output_dim"])
        self.max_batch_tokens = max(1, int(max_batch_tokens))
        self._right_padded = getattr(self.tokenizer, "padding_side", "right") == "right"

    def tokenize(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        ids, mask = self.tokenizer(texts)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
        return feeds

    def run(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """One session run on already tokenized inputs (no slicing)."""
        out = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0]
        if out.ndim == 3:
            out = pool_logits(out, feeds["attention_mask"])
        return np.ascontiguousarray(out[:, : self.output_dim], dtype=np.float32)

    def run_sliced(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        n, seq = feeds["input_ids"].shape
        rows = max(1, self.max_batch_tokens // max(1, seq))
        if rows >= n:
            return self.run(feeds)
        parts = []
        for start in range(0, n, rows):
            part = {k: v[start:start + rows] for k, v in feeds.items()}
            if self._right_padded:
                width = max(1, int(part["attention_mask"].sum(axis=1).max()))
                part = {k: v[:, :width] for k, v in part.items()}
            parts.append(self.run(part))
        return np.concatenate(parts, axis=0)

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        texts = list(texts)
        out = np.zeros((len(texts), self.output_dim), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            out[start:start + batch_size] = self.run_sliced(self.tokenize(texts[start:start + batch_size]))
        return out

    def term_weights(self, vector: np.ndarray, quantization_factor: int | None = None) -> dict[str, float | int]:
        """The vector as {token: weight}; with a factor, integer impacts as PIRB indexes them (round(w * f) > 0)."""
        idx = np.nonzero(vector)[0]
        vocab = getattr(self, "_id_to_token", None)
        if vocab is None:
            vocab = self._id_to_token = {i: t for t, i in self.tokenizer.get_vocab().items()}
        if quantization_factor is None:
            return {vocab[int(i)]: float(vector[i]) for i in idx if int(i) in vocab}
        q = np.rint(vector[idx] * quantization_factor).astype(np.int64)
        return {vocab[int(i)]: int(v) for i, v in zip(idx, q) if v > 0 and int(i) in vocab}


def active_mask(vectors: np.ndarray, quantization_factor: int = 100) -> np.ndarray:
    """Dimensions PIRB would index: round(w * factor) > 0."""
    return np.rint(vectors * quantization_factor) > 0
