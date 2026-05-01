"""PyLate training backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from training.backends.registry import BackendDependencyError, TrainingRequest
from training.backends.sentence_transformers_backend import (
    COMMON_TRAINING_KEYS,
    SentenceTransformersConfigError,
    _dict_section,
    _resolve_value,
    _training_args,
    load_flagembedding_jsonl_dataset,
)


def _require_pylate() -> None:
    if importlib.util.find_spec("pylate") is None:
        raise BackendDependencyError(
            "Backend 'pylate' requires the pylate dependency. "
            "Install it with: pip install -r requirements/requirements-pylate.txt"
        )


def _load_pylate_training_stack():
    _require_pylate()
    try:
        from datasets import Dataset
        from pylate import losses, models
        from pylate.utils import ColBERTCollator
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    except ModuleNotFoundError as exc:
        raise BackendDependencyError(
            "Backend 'pylate' requires the PyLate training stack. "
            "Install it with: pip install -r requirements/requirements-pylate.txt"
        ) from exc
    return Dataset, models.ColBERT, losses.Contrastive, ColBERTCollator, SentenceTransformerTrainer, SentenceTransformerTrainingArguments


def _backend_config(config: dict[str, Any]) -> dict[str, Any]:
    common = {key: config[key] for key in COMMON_TRAINING_KEYS if key in config}
    return common | _dict_section(config, "backend_config") | _dict_section(config, "pylate")


def _resolve_train_data_path(config: dict[str, Any], backend_config: dict[str, Any]) -> Path:
    train_data = _resolve_value(config, backend_config, "train_data")
    if not train_data:
        raise SentenceTransformersConfigError("PyLate training requires 'train_data'.")

    path = Path(str(train_data))
    if path.is_dir():
        for filename in ("dataset.jsonl", "mixed_dataset.jsonl"):
            candidate = path / filename
            if candidate.exists():
                return candidate
        jsonl_files = sorted(path.glob("*.jsonl"))
        if jsonl_files:
            return jsonl_files[0]
        raise SentenceTransformersConfigError(f"No JSONL training file found under '{path}'.")
    if not path.exists():
        raise SentenceTransformersConfigError(f"Training data path '{path}' does not exist.")
    return path


def _model_kwargs(backend_config: dict[str, Any]) -> dict[str, Any]:
    model_kwargs = backend_config.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise SentenceTransformersConfigError("'pylate.model_kwargs' must be a mapping when provided.")
    return dict(model_kwargs)


def run_colbert_training(request: TrainingRequest) -> int:
    (
        Dataset,
        ColBERT,
        Contrastive,
        ColBERTCollator,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    ) = _load_pylate_training_stack()

    config = request.config
    backend_config = _backend_config(config)
    model_name_or_path = _resolve_value(
        config,
        backend_config,
        "model_name_or_path",
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
    )
    output_dir = Path(str(_resolve_value(config, backend_config, "output_dir", "runs/pylate-colbert")))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = _resolve_train_data_path(config, backend_config)
    negatives_per_query = backend_config.get("negatives_per_query", 1)
    if negatives_per_query is not None:
        negatives_per_query = int(negatives_per_query)
    rows = load_flagembedding_jsonl_dataset(
        data_path,
        negatives_per_query=negatives_per_query,
        query_prefix=str(backend_config.get("query_prefix", "") or ""),
        passage_prefix=str(backend_config.get("passage_prefix", "") or ""),
    )
    train_dataset = Dataset.from_list(rows)

    colbert_kwargs: dict[str, Any] = {
        "model_name_or_path": str(model_name_or_path),
        "model_kwargs": _model_kwargs(backend_config),
    }
    for key in (
        "cache_folder",
        "trust_remote_code",
        "revision",
        "local_files_only",
        "token",
        "truncate_dim",
        "embedding_size",
        "bias",
        "query_prefix",
        "document_prefix",
        "add_special_tokens",
        "truncation",
        "query_length",
        "document_length",
        "do_query_expansion",
        "attend_to_expansion_tokens",
        "skiplist_words",
        "tokenizer_kwargs",
        "config_kwargs",
    ):
        if key in backend_config:
            colbert_kwargs[key] = backend_config[key]

    model = ColBERT(**colbert_kwargs)
    args = _training_args(output_dir, backend_config | {"remove_unused_columns": False}, SentenceTransformerTrainingArguments)
    if hasattr(args, "remove_unused_columns"):
        args.remove_unused_columns = False
    loss = Contrastive(
        model=model,
        gather_across_devices=bool(backend_config.get("gather_across_devices", False)),
        temperature=float(backend_config.get("temperature", 1.0)),
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        data_collator=ColBERTCollator(tokenize_fn=model.tokenize),
    )

    print(
        "[INFO] Training PyLate ColBERT "
        f"with {len(rows)} examples from {data_path} and model {model_name_or_path}.",
        flush=True,
    )
    trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    print(f"[INFO] Saved final PyLate ColBERT model to: {final_dir}", flush=True)
    return 0


def run_training(request: TrainingRequest) -> int:
    if request.training_type in {"colbert", "late-interaction"}:
        return run_colbert_training(request)

    raise NotImplementedError(f"Training type '{request.training_type}' is not implemented for backend 'pylate'.")
