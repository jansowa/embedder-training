"""Sentence Transformers modules for a static sparse query encoder with expansion (inference-free query side).

``SparseStaticExpansion`` is the query route of a ``SparseEncoder``: every token id maps to up to ``k`` (dimension,
weight) slots and a query is the max over its tokens of those slots,

    q[v] = max_{l : mask[l] = 1} relu(values[ids[l], j])   over the slots j with indices[ids[l], j] = v

so a token can expand into other vocabulary dimensions (sentence_transformers' own ``SparseStaticEmbedding`` holds one
weight per token, its own dimension only). With ``k = 1`` and ``indices[t] = t`` it is that module up to the special
tokens. The table is the ``static-sparse-query/1`` format of ``distill.static_query`` (``static_query.safetensors``:
``indices`` int32 [R, k], ``values`` float32 [R, k], ``slot_mask`` bool [R, k]; ``static_query.json``); this file does
not import ``distill`` so it can travel with a model on the Hugging Face Hub.

``StaticQueryRouter`` is ``sentence_transformers.models.Router`` that also resolves ``SparseStaticExpansion`` when the
model is loaded with ``trust_remote_code=True`` (ST's Router imports its sub-module classes by name only, so a class
shipped as remote code has to be found by the router that came with it). ``release.st_export`` writes such a model:

    from sentence_transformers import SparseEncoder
    model = SparseEncoder("path-or-repo", trust_remote_code=True)
    q = model.encode_query(["jak działa splade?"])          # lookup table, no transformer
    d = model.encode_document(["SPLADE to rzadki ..."])     # the SPLADE document encoder
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from sentence_transformers.models import Router
from sentence_transformers.models.InputModule import InputModule
from sentence_transformers.util import import_from_string
from transformers import AutoTokenizer

TABLE_FILE = "static_query.safetensors"
TABLE_CONFIG = "static_query.json"


class SparseStaticExpansion(InputModule):
    config_keys: list[str] = ["output_dim", "max_length", "add_special_tokens", "pooling"]
    save_in_root: bool = False

    def __init__(
        self,
        tokenizer,
        indices: torch.Tensor,
        values: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
        output_dim: int | None = None,
        max_length: int = 128,
        add_special_tokens: bool = True,
        pooling: str = "max",
        table_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        if pooling not in ("max", "sum"):
            raise ValueError(f"pooling must be max or sum, not {pooling}")
        self.tokenizer = tokenizer
        if slot_mask is None:
            slot_mask = torch.ones_like(indices, dtype=torch.bool)
        # relu and the slot mask are folded into the stored weights: inference is one gather and one scatter
        weights = torch.relu(values.float()) * slot_mask.float()
        self.register_buffer("indices", indices.long().contiguous())
        self.register_buffer("weights", weights.contiguous())
        self.output_dim = int(output_dim if output_dim is not None else int(indices.max()) + 1)
        self.max_length = int(max_length)
        self.max_seq_length = self.max_length
        self.add_special_tokens = bool(add_special_tokens)
        self.pooling = pooling
        self.table_config = dict(table_config or {})

    # ------------------------------------------------------------------------------------------- encode
    def tokenize(self, texts: list[str], padding: str | bool = True, **kwargs) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(list(texts), padding=padding, truncation=True, max_length=self.max_length,
                             add_special_tokens=self.add_special_tokens, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids, attention_mask = features["input_ids"], features["attention_mask"]
        batch = input_ids.shape[0]
        idx = self.indices[input_ids].reshape(batch, -1)  # [B, L * k]
        val = (self.weights[input_ids] * attention_mask.unsqueeze(-1).to(self.weights.dtype)).reshape(batch, -1)
        # dense [B, output_dim] on purpose: ST's own options (max_active_dims, dense outputs) expect a dense tensor here,
        # and a sparse return would save ~0.15 ms of the ~1 ms a query costs inside SparseEncoder.encode
        out = val.new_zeros(batch, self.output_dim)
        if self.pooling == "max":  # weights are >= 0, so the zero start is the empty max
            out = out.scatter_reduce(1, idx, val, reduce="amax", include_self=True)
        else:
            out = out.scatter_add(1, idx, val)
        features["sentence_embedding"] = out
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.output_dim

    # ------------------------------------------------------------------------------------------- save / load
    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        os.makedirs(output_path, exist_ok=True)
        self.save_tokenizer(output_path)
        save_file({"indices": self.indices.to(torch.int32).cpu().contiguous(), "values": self.weights.float().cpu().contiguous()},
                  os.path.join(output_path, TABLE_FILE))
        config = dict(self.table_config, format="static-sparse-query/1", rows=int(self.indices.shape[0]), k=int(self.indices.shape[1]),
                      output_dim=self.output_dim, max_length=self.max_length, add_special_tokens=self.add_special_tokens,
                      pooling=self.pooling, active_slots=int((self.weights > 0).sum()))
        with open(os.path.join(output_path, TABLE_CONFIG), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
        self.save_config(output_path)

    @classmethod
    def from_table_dir(cls, table_dir: str, tokenizer=None) -> "SparseStaticExpansion":
        """Reads a ``static-sparse-query/1`` directory (``distill.static_query`` output or a saved module)."""
        with open(os.path.join(table_dir, TABLE_CONFIG), encoding="utf-8") as handle:
            config = json.load(handle)
        tensors = load_file(os.path.join(table_dir, TABLE_FILE))
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(table_dir)
        keep = {k: config[k] for k in ("format", "meta") if k in config}
        return cls(tokenizer, tensors["indices"], tensors["values"], tensors.get("slot_mask"), output_dim=config["output_dim"],
                   max_length=config.get("max_length", 128), add_special_tokens=config.get("add_special_tokens", True),
                   pooling=config.get("pooling", "max"), table_config=keep)

    @classmethod
    def load(cls, model_name_or_path: str, subfolder: str = "", token: bool | str | None = None, cache_folder: str | None = None,
             revision: str | None = None, local_files_only: bool = False, **kwargs) -> "SparseStaticExpansion":
        from sentence_transformers.util import load_dir_path

        local = load_dir_path(model_name_or_path=model_name_or_path, subfolder=subfolder, token=token, cache_folder=cache_folder,
                              revision=revision, local_files_only=local_files_only)
        return cls.from_table_dir(local)

    def __repr__(self) -> str:
        return (f"SparseStaticExpansion(rows={self.indices.shape[0]}, k={self.indices.shape[1]}, dim={self.output_dim}, "
                f"pooling={self.pooling}, active_slots={int((self.weights > 0).sum())})")


class StaticQueryRouter(Router):
    """``Router`` that resolves ``SparseStaticExpansion`` itself (it may live in a remote-code module of any name)."""

    @classmethod
    def load(cls, model_name_or_path: str, subfolder: str = "", token: bool | str | None = None, cache_folder: str | None = None,
             revision: str | None = None, local_files_only: bool = False, **kwargs):
        from pathlib import Path

        hub = {"token": token, "cache_folder": cache_folder, "revision": revision, "local_files_only": local_files_only}
        config = cls.load_config(model_name_or_path=model_name_or_path, subfolder=subfolder, **hub)
        modules = {}
        for model_id, model_type in config["types"].items():
            module_class = SparseStaticExpansion if model_type.rsplit(".", 1)[-1] == "SparseStaticExpansion" else import_from_string(model_type)
            modules[model_id] = module_class.load(model_name_or_path, subfolder=Path(subfolder, model_id).as_posix(), **hub, **kwargs)
        structure = {route: [modules[m] for m in ids] for route, ids in config["structure"].items()}
        return cls(structure, **config["parameters"])
