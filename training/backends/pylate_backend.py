"""PyLate training backend."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from training.benchmarks import resolve_benchmark_settings
from training.backends.registry import BackendDependencyError, TrainingRequest
from training.backends.sentence_transformers_backend import (
    COMMON_TRAINING_KEYS,
    SentenceTransformersConfigError,
    _dict_section,
    _load_training_rows,
    _model_cache_dir,
    _resolve_value,
    _training_args,
    _training_args_config_for_rows,
)
from training.checkpoints import build_epoch_checkpoint_callback, resolve_resume_checkpoint, train_with_resume
from training.distributed import barrier_if_distributed, is_main_process


LOGGER = logging.getLogger(__name__)


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


def _model_kwargs(backend_config: dict[str, Any]) -> dict[str, Any]:
    model_kwargs = backend_config.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise SentenceTransformersConfigError("'pylate.model_kwargs' must be a mapping when provided.")
    return dict(model_kwargs)


def _trainer_is_main_process(trainer: Any) -> bool:
    is_world_process_zero = getattr(trainer, "is_world_process_zero", None)
    if callable(is_world_process_zero):
        try:
            return bool(is_world_process_zero())
        except TypeError:
            pass
    return is_main_process()


def _warn_if_pylate_benchmarks_requested(
    config: dict[str, Any],
    backend_config: dict[str, Any],
    request: TrainingRequest,
) -> None:
    settings = resolve_benchmark_settings(
        config,
        backend_config,
        request.cli_args,
        default_query_instruction=str(backend_config.get("query_prefix", "") or ""),
    )
    if settings.enabled and is_main_process():
        LOGGER.warning(
            "PyLate post-training benchmarks are not wired yet; benchmark.checkpoints uses the shared "
            "training.benchmarks resolver, but this backend still needs a benchmark runner integration."
        )


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
    resume_from_checkpoint = resolve_resume_checkpoint(output_dir, config, backend_config, request.cli_args)

    loaded = _load_training_rows(
        config,
        backend_config,
        config_path=request.config_path,
        default_negatives_per_query=1,
    )
    train_dataset = Dataset.from_list(loaded.rows)

    colbert_kwargs: dict[str, Any] = {
        "model_name_or_path": str(model_name_or_path),
        "model_kwargs": _model_kwargs(backend_config),
    }
    model_cache_dir = _model_cache_dir(backend_config)
    if model_cache_dir is not None:
        colbert_kwargs["cache_folder"] = model_cache_dir
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
    args_config = _training_args_config_for_rows(backend_config, loaded) | {"remove_unused_columns": False}
    args = _training_args(output_dir, args_config, SentenceTransformerTrainingArguments)
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
    epoch_checkpoint_callback = build_epoch_checkpoint_callback(output_dir, config, backend_config)
    if is_main_process() and epoch_checkpoint_callback is not None and hasattr(trainer, "add_callback"):
        trainer.add_callback(epoch_checkpoint_callback)

    print(
        "[INFO] Training PyLate ColBERT "
        f"with {len(loaded.rows)} examples from {loaded.source_label} and model {model_name_or_path}.",
        flush=True,
    )
    train_with_resume(trainer, resume_from_checkpoint)

    final_dir = output_dir / "final"
    if _trainer_is_main_process(trainer):
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        print(f"[INFO] Saved final PyLate ColBERT model to: {final_dir}", flush=True)
    barrier_if_distributed()
    _warn_if_pylate_benchmarks_requested(config, backend_config, request)
    return 0


def run_training(request: TrainingRequest) -> int:
    if request.training_type in {"colbert", "late-interaction"}:
        return run_colbert_training(request)

    raise NotImplementedError(f"Training type '{request.training_type}' is not implemented for backend 'pylate'.")
