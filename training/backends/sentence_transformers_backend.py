"""SentenceTransformers training backend."""

from __future__ import annotations

import gc
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

from training.benchmarks import (
    resolve_benchmark_settings,
    resolve_benchmark_targets,
    run_benchmarks_for_model,
    run_benchmarks_for_targets,
)
from training.backends.registry import BackendDependencyError, TrainingRequest
from training.checkpoints import (
    build_epoch_checkpoint_callback,
    build_step_checkpoint_callback,
    resolve_resume_checkpoint,
    should_skip_training_for_final,
    train_with_resume,
)
from training.distributed import barrier_if_distributed, is_main_process, is_torchrun_child, wait_for_files
from training.multi_dataset import (
    LoadedTrainingRows,
    PROPORTIONAL_BATCH_BEST_EFFORT,
    ProportionalNoDuplicatesBatchSamplerFactory,
    dedupe_values_from_training_row,
    normalize_dataset_mix_strategy,
    resolve_dataset_sources,
)
from training.wandb_tracking import reports_to_wandb, wandb_run_environment


class SentenceTransformersConfigError(ValueError):
    """Raised when a SentenceTransformers config cannot drive training."""


COMMON_TRAINING_KEYS = {
    "batch_sampler",
    "bf16",
    "dataloader_drop_last",
    "dataloader_num_workers",
    "dataloader_persistent_workers",
    "dataloader_pin_memory",
    "dataloader_prefetch_factor",
    "dataset_mix_strategy",
    "epoch_checkpoint_dir",
    "fp16",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "gradient_checkpointing_kwargs",
    "keep_epoch_checkpoints",
    "keep_step_checkpoints",
    "learning_rate",
    "loss",
    "logging_steps",
    "max_seq_length",
    "max_steps",
    "cache_dir",
    "cache_folder",
    "model_cache_dir",
    "model_kwargs",
    "model_name_or_path",
    "num_train_epochs",
    "optim",
    "output_dir",
    "overwrite_output_dir",
    "per_device_train_batch_size",
    "processor_kwargs",
    "query_instruction_for_retrieval",
    "report_to",
    "resume",
    "resume_from_checkpoint",
    "run_name",
    "save_steps",
    "save_strategy",
    "save_total_limit",
    "seed",
    "splade_activation_stats",
    "step_checkpoint_dir",
    "tokenizer_args",
    "tokenizer_name_or_path",
    "tf32",
    "torch_compile",
    "torch_compile_backend",
    "torch_compile_mode",
    "train_batch_size",
    "train_data",
    "trust_remote_code",
    "warmup_ratio",
    "wandb_run_id",
    "weight_decay",
}

BATCH_SAMPLER_ALIASES = {
    "batch_sampler": "batch_sampler",
    "default": "batch_sampler",
    "group_by_label": "group_by_label",
    "group-by-label": "group_by_label",
    "no_duplicate": "no_duplicates",
    "no-duplicate": "no_duplicates",
    "no_duplicates": "no_duplicates",
    "no-duplicates": "no_duplicates",
    "no_duplicates_hashed": "no_duplicates_hashed",
    "no-duplicates-hashed": "no_duplicates_hashed",
}


def _require_sentence_transformers() -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        raise BackendDependencyError(
            "Backend 'sentence-transformers' requires the sentence-transformers dependency. "
            "Install it with: pip install -r requirements/requirements-sentence-transformers.txt"
        )


def _load_sentence_transformers():
    _require_sentence_transformers()
    try:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        try:
            from sentence_transformers.sentence_transformer.losses import (
                MatryoshkaLoss,
                MultipleNegativesRankingLoss,
            )
        except ModuleNotFoundError:
            from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    except ModuleNotFoundError as exc:
        raise BackendDependencyError(
            "Backend 'sentence-transformers' requires the sentence-transformers training stack. "
            "Install it with: pip install -r requirements/requirements-sentence-transformers.txt"
        ) from exc
    return (
        Dataset,
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        MultipleNegativesRankingLoss,
        MatryoshkaLoss,
    )


def _load_sparse_sentence_transformers():
    _require_sentence_transformers()
    try:
        from datasets import Dataset
        from sentence_transformers.sparse_encoder import (
            SparseEncoder,
            SparseEncoderTrainer,
            SparseEncoderTrainingArguments,
        )
        from sentence_transformers.sparse_encoder.losses import (
            SparseMarginMSELoss,
            SparseMultipleNegativesRankingLoss,
            SpladeLoss,
        )
        try:
            from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
        except (ImportError, ModuleNotFoundError):
            from sentence_transformers.sparse_encoder.modules import (
                SpladePooling,
                Transformer as MLMTransformer,
            )
    except ModuleNotFoundError as exc:
        raise BackendDependencyError(
            "Backend 'sentence-transformers' with training type 'splade' requires the sparse encoder stack. "
            "Install it with: pip install -r requirements/requirements-sentence-transformers.txt"
        ) from exc
    return (
        Dataset,
        SparseEncoder,
        SparseEncoderTrainer,
        SparseEncoderTrainingArguments,
        SparseMarginMSELoss,
        SparseMultipleNegativesRankingLoss,
        SpladeLoss,
        MLMTransformer,
        SpladePooling,
    )


def _dict_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _backend_config(config: dict[str, Any]) -> dict[str, Any]:
    common = {key: config[key] for key in COMMON_TRAINING_KEYS if key in config}
    return common | _dict_section(config, "backend_config") | _dict_section(config, "sentence_transformers")


def _resolve_value(config: dict[str, Any], backend_config: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in backend_config:
        return backend_config[key]
    return config.get(key, default)


def _accepts_kwarg(callable_obj: Any, key: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    return key in parameters or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())


def _normalize_batch_sampler(value: Any) -> str:
    if callable(value):
        return "custom"
    raw_value = getattr(value, "value", value)
    normalized = str(raw_value).strip().lower()
    normalized = normalized.replace("batchsamplers.", "").replace("batchsampler.", "")
    normalized = normalized.replace(" ", "_")
    batch_sampler = BATCH_SAMPLER_ALIASES.get(normalized)
    if batch_sampler is None:
        raise SentenceTransformersConfigError(
            "'sentence_transformers.batch_sampler' must be one of: batch_sampler, no_duplicates, "
            "no_duplicates_hashed, group_by_label."
        )
    return batch_sampler


def _resolve_train_data_sources(
    config: dict[str, Any],
    backend_config: dict[str, Any],
    *,
    config_path: str | None = None,
):
    train_data = _resolve_value(config, backend_config, "train_data")
    if not train_data:
        raise SentenceTransformersConfigError("SentenceTransformers training requires 'train_data'.")

    return resolve_dataset_sources(
        train_data,
        config,
        backend_config,
        config_path=config_path,
    )


def _ensure_text_list(value: Any, field_name: str, line_no: int) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return value
    raise SentenceTransformersConfigError(f"Line {line_no}: field '{field_name}' must be a non-empty string or list of strings.")


def _ensure_float_list(value: Any, field_name: str, line_no: int) -> list[float]:
    if not isinstance(value, list) or not value:
        raise SentenceTransformersConfigError(f"Line {line_no}: field '{field_name}' must be a non-empty list of numbers.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise SentenceTransformersConfigError(f"Line {line_no}: field '{field_name}' must contain only numbers.") from exc


def _prefix_text(prefix: str, text: str) -> str:
    return f"{prefix}{text}" if prefix else text


def _normalize_teacher_scores(pos_score: float, neg_scores: list[float], normalization: str) -> tuple[float, list[float]]:
    if normalization == "none":
        return pos_score, neg_scores
    values = [pos_score, *neg_scores]
    if normalization == "per_query_minmax":
        min_value, max_value = min(values), max(values)
        span = max_value - min_value
        if span == 0:
            return 0.0, [0.0 for _ in neg_scores]
        normalized = [(value - min_value) / span for value in values]
        return normalized[0], normalized[1:]
    if normalization == "per_query_zscore":
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = variance ** 0.5
        if std == 0:
            return 0.0, [0.0 for _ in neg_scores]
        normalized = [(value - mean) / std for value in values]
        return normalized[0], normalized[1:]
    raise SentenceTransformersConfigError(
        "'sentence_transformers.score_normalization' must be one of: none, per_query_minmax, per_query_zscore."
    )


def _splade_base_loss_name(backend_config: dict[str, Any]) -> str:
    loss_name = str(backend_config.get("loss", "sparse_multiple_negatives_ranking")).strip().lower().replace("-", "_")
    aliases = {
        "mnrl": "sparse_multiple_negatives_ranking",
        "multiple_negatives_ranking": "sparse_multiple_negatives_ranking",
        "sparse_multiple_negatives_ranking_loss": "sparse_multiple_negatives_ranking",
        "sparse_margin_mse_loss": "sparse_margin_mse",
        "margin_mse": "sparse_margin_mse",
        "distil_margin_mse": "sparse_margin_mse",
    }
    loss_name = aliases.get(loss_name, loss_name)
    if loss_name not in {"sparse_multiple_negatives_ranking", "sparse_margin_mse"}:
        raise SentenceTransformersConfigError(
            "'sentence_transformers.loss' must be one of: sparse_multiple_negatives_ranking, sparse_margin_mse."
        )
    return loss_name


def _min_negatives_in_flagembedding_jsonl(data_path: Path) -> int:
    min_negatives: int | None = None
    with data_path.open(encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SentenceTransformersConfigError(f"Line {line_no}: invalid JSON: {exc}") from exc
            negatives = _ensure_text_list(item.get("neg"), "neg", line_no)
            if not negatives:
                raise SentenceTransformersConfigError(f"Line {line_no}: at least one negative is required.")
            min_negatives = len(negatives) if min_negatives is None else min(min_negatives, len(negatives))
    if min_negatives is None:
        raise SentenceTransformersConfigError(f"Training data file '{data_path}' did not yield any examples.")
    return min_negatives


def load_flagembedding_jsonl_dataset(
    data_path: Path,
    *,
    negatives_per_query: int | None,
    query_prefix: str,
    passage_prefix: str,
    use_score_labels: bool = False,
    score_normalization: str = "none",
) -> list[dict[str, Any]]:
    parsed_items: list[tuple[int, str, list[str], list[str], list[float] | None, list[float] | None]] = []
    min_negatives: int | None = None

    with data_path.open(encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SentenceTransformersConfigError(f"Line {line_no}: invalid JSON: {exc}") from exc

            query = item.get("query")
            if not isinstance(query, str) or not query:
                raise SentenceTransformersConfigError(f"Line {line_no}: field 'query' must be a non-empty string.")
            positives = _ensure_text_list(item.get("pos"), "pos", line_no)
            negatives = _ensure_text_list(item.get("neg"), "neg", line_no)
            if not negatives:
                raise SentenceTransformersConfigError(f"Line {line_no}: at least one negative is required.")
            pos_scores = neg_scores = None
            if use_score_labels:
                pos_scores = _ensure_float_list(item.get("pos_scores"), "pos_scores", line_no)
                neg_scores = _ensure_float_list(item.get("neg_scores"), "neg_scores", line_no)
                if len(pos_scores) not in {1, len(positives)}:
                    raise SentenceTransformersConfigError(
                        f"Line {line_no}: field 'pos_scores' must have length 1 or match the number of positives."
                    )
                if len(neg_scores) < len(negatives):
                    raise SentenceTransformersConfigError(
                        f"Line {line_no}: field 'neg_scores' must have at least as many scores as negatives."
                    )

            parsed_items.append((line_no, query, positives, negatives, pos_scores, neg_scores))
            min_negatives = len(negatives) if min_negatives is None else min(min_negatives, len(negatives))

    if not parsed_items:
        raise SentenceTransformersConfigError(f"Training data file '{data_path}' did not yield any examples.")

    effective_negatives = min_negatives if negatives_per_query is None else negatives_per_query
    if effective_negatives is None or effective_negatives <= 0:
        raise SentenceTransformersConfigError("'negatives_per_query' must be greater than zero.")

    rows: list[dict[str, Any]] = []
    for line_no, query, positives, negatives, pos_scores, neg_scores in parsed_items:
        if len(negatives) < effective_negatives:
            raise SentenceTransformersConfigError(
                f"Line {line_no}: expected at least {effective_negatives} negatives, got {len(negatives)}."
            )

        selected_negatives = negatives[:effective_negatives]
        for positive_idx, positive in enumerate(positives):
            row = {
                "anchor": _prefix_text(query_prefix, query),
                "positive": _prefix_text(passage_prefix, positive),
            }
            for idx, negative in enumerate(selected_negatives, start=1):
                row[f"negative_{idx}"] = _prefix_text(passage_prefix, negative)
            if use_score_labels:
                assert pos_scores is not None and neg_scores is not None
                pos_score = pos_scores[positive_idx] if len(pos_scores) > 1 else pos_scores[0]
                selected_neg_scores = neg_scores[:effective_negatives]
                pos_score, selected_neg_scores = _normalize_teacher_scores(
                    pos_score,
                    selected_neg_scores,
                    score_normalization,
                )
                labels = [pos_score - neg_score for neg_score in selected_neg_scores]
                row["label"] = labels[0] if len(labels) == 1 else labels
            rows.append(row)

    return rows


def _training_args(output_dir: Path, backend_config: dict[str, Any], training_arguments_cls: Any):
    kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": bool(backend_config.get("overwrite_output_dir", True)),
        "do_train": True,
        "max_steps": int(backend_config.get("max_steps", -1)),
        "num_train_epochs": float(backend_config.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(
            backend_config.get("per_device_train_batch_size", backend_config.get("train_batch_size", 8))
        ),
        "gradient_accumulation_steps": int(backend_config.get("gradient_accumulation_steps", 1)),
        "gradient_checkpointing": bool(backend_config.get("gradient_checkpointing", False)),
        "learning_rate": float(backend_config.get("learning_rate", 5e-5)),
        "weight_decay": float(backend_config.get("weight_decay", 0.0)),
        "warmup_ratio": float(backend_config.get("warmup_ratio", 0.0)),
        "logging_steps": int(backend_config.get("logging_steps", 10)),
        "save_strategy": str(backend_config.get("save_strategy", "steps")),
        "save_steps": int(backend_config.get("save_steps", 500)),
        "save_total_limit": backend_config.get("save_total_limit"),
        "seed": int(backend_config.get("seed", 42)),
        "bf16": bool(backend_config.get("bf16", False)),
        "fp16": bool(backend_config.get("fp16", False)),
        "dataloader_drop_last": bool(backend_config.get("dataloader_drop_last", False)),
        "dataloader_num_workers": int(backend_config.get("dataloader_num_workers", 0)),
        "report_to": backend_config.get("report_to", []),
    }
    optional_training_args: dict[str, Any] = {
        "dataloader_persistent_workers": _as_bool,
        "dataloader_pin_memory": _as_bool,
        "dataloader_prefetch_factor": int,
        "gradient_checkpointing_kwargs": dict,
        "optim": str,
        "tf32": _as_bool,
        "torch_compile": _as_bool,
        "torch_compile_backend": str,
        "torch_compile_mode": str,
    }
    for key, converter in optional_training_args.items():
        if key not in backend_config:
            continue
        if not _accepts_kwarg(training_arguments_cls, key):
            raise SentenceTransformersConfigError(
                f"'sentence_transformers.{key}' requires a SentenceTransformers/Transformers version "
                f"whose training arguments support {key}."
            )
        value = backend_config[key]
        if key == "gradient_checkpointing_kwargs" and not isinstance(value, dict):
            raise SentenceTransformersConfigError(
                "'sentence_transformers.gradient_checkpointing_kwargs' must be a mapping."
            )
        kwargs[key] = converter(value)
    if backend_config.get("batch_sampler") is not None:
        if not _accepts_kwarg(training_arguments_cls, "batch_sampler"):
            raise SentenceTransformersConfigError(
                "'sentence_transformers.batch_sampler' requires a SentenceTransformers version "
                "whose training arguments support batch_sampler."
            )
        batch_sampler = backend_config["batch_sampler"]
        kwargs["batch_sampler"] = batch_sampler if callable(batch_sampler) else _normalize_batch_sampler(batch_sampler)
    if backend_config.get("run_name") is not None:
        kwargs["run_name"] = str(backend_config["run_name"])
    return training_arguments_cls(**kwargs)


def _training_args_config_for_rows(backend_config: dict[str, Any], loaded: LoadedTrainingRows) -> dict[str, Any]:
    strategy = normalize_dataset_mix_strategy(backend_config.get("dataset_mix_strategy"))
    if strategy != PROPORTIONAL_BATCH_BEST_EFFORT:
        return backend_config
    if not loaded.is_multi_source:
        return backend_config
    configured_batch_sampler = backend_config.get("batch_sampler")
    if configured_batch_sampler is None or _normalize_batch_sampler(configured_batch_sampler) != "no_duplicates":
        raise SentenceTransformersConfigError(
            "'dataset_mix_strategy: proportional_batch_best_effort' requires 'batch_sampler: no_duplicates'."
        )
    updated = dict(backend_config)
    updated["batch_sampler"] = ProportionalNoDuplicatesBatchSamplerFactory(
        loaded.dataset_ids,
        loaded.dedupe_values,
    )
    return updated


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _finish_wandb_run(backend_config: dict[str, Any]) -> None:
    if not is_main_process():
        return
    if not reports_to_wandb(backend_config.get("report_to", [])):
        return
    try:
        import wandb
    except ModuleNotFoundError:
        return
    if getattr(wandb, "run", None) is not None:
        wandb.finish()


def _add_checkpoint_callbacks(trainer: Any, output_dir: Path, config: dict[str, Any], backend_config: dict[str, Any]) -> None:
    if not is_main_process():
        return
    if not hasattr(trainer, "add_callback"):
        return
    for callback in (
        build_epoch_checkpoint_callback(output_dir, config, backend_config),
        build_step_checkpoint_callback(output_dir, config, backend_config),
    ):
        if callback is not None:
            trainer.add_callback(callback)


def _trainer_is_main_process(trainer: Any) -> bool:
    is_world_process_zero = getattr(trainer, "is_world_process_zero", None)
    if callable(is_world_process_zero):
        try:
            return bool(is_world_process_zero())
        except TypeError:
            pass
    return is_main_process()


def _save_final_model(trainer: Any, model: Any, final_dir: Path, *, label: str) -> None:
    if _trainer_is_main_process(trainer):
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        print(f"[INFO] Saved final {label} model to: {final_dir}", flush=True)
    barrier_if_distributed()


def _run_sentence_transformers_post_training_benchmarks(
    output_dir: Path,
    config: dict[str, Any],
    backend_config: dict[str, Any],
    request: TrainingRequest,
    *,
    trainer: Any = None,
    model: Any = None,
) -> None:
    default_query_instruction = str(
        backend_config.get(
            "query_instruction_for_retrieval",
            backend_config.get("query_prefix", ""),
        )
        or ""
    )
    settings = resolve_benchmark_settings(
        config,
        backend_config,
        request.cli_args,
        default_query_instruction=default_query_instruction,
    )
    if not settings.enabled:
        return

    _release_training_gpu_memory(trainer, model)
    trainer_state = getattr(trainer, "state", None)
    raw_final_step = getattr(trainer_state, "global_step", None)
    try:
        final_step = int(raw_final_step) if raw_final_step is not None else None
    except (TypeError, ValueError):
        final_step = None
    targets = resolve_benchmark_targets(output_dir, settings, final_step=final_step)
    distributed = is_torchrun_child()
    marker = output_dir / ".post-training-benchmarks.complete"
    main_process = is_main_process()
    if distributed:
        if main_process:
            marker.unlink(missing_ok=True)
        barrier_if_distributed()

    benchmark_error: BaseException | None = None
    try:
        if main_process:
            for target in targets:
                print(f"[INFO] Running post-training benchmarks for {target.label}: {target.path}.", flush=True)
            run_benchmarks_for_targets(targets, settings, runner=run_benchmarks_for_model)
        elif distributed:
            wait_for_files([marker])
    except BaseException as exc:
        benchmark_error = exc
    finally:
        if distributed:
            if main_process:
                marker.touch()
            barrier_if_distributed()
            if main_process:
                marker.unlink(missing_ok=True)
    if benchmark_error is not None:
        raise benchmark_error


def _skip_completed_training(
    request: TrainingRequest,
    *,
    default_output_dir: str,
) -> bool:
    config = request.config
    backend_config = _backend_config(config)
    output_dir = Path(str(_resolve_value(config, backend_config, "output_dir", default_output_dir)))
    if not should_skip_training_for_final(output_dir, config, backend_config, request.cli_args):
        return False

    if is_main_process():
        print(
            f"[INFO] Final model already exists at {output_dir / 'final'}; skipping completed training.",
            flush=True,
        )
    _run_sentence_transformers_post_training_benchmarks(
        output_dir,
        config,
        backend_config,
        request,
    )
    return True


def _release_training_gpu_memory(trainer: Any, model: Any) -> None:
    """Release training allocations before PIRB subprocesses claim the GPUs."""
    if trainer is not None:
        accelerator = getattr(trainer, "accelerator", None)
        free_memory = getattr(accelerator, "free_memory", None)
        if callable(free_memory):
            try:
                free_memory()
            except Exception:
                pass
        for attr in ("optimizer", "lr_scheduler"):
            if hasattr(trainer, attr):
                try:
                    setattr(trainer, attr, None)
                except Exception:
                    pass

    move_to = getattr(model, "to", None)
    if callable(move_to):
        try:
            move_to("cpu")
        except Exception:
            pass

    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unique_texts(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _even_sample(values: list[str], sample_size: int) -> list[str]:
    values = _unique_texts(values)
    if len(values) <= sample_size:
        return values
    if sample_size == 1:
        return [values[0]]
    step = (len(values) - 1) / (sample_size - 1)
    return [values[round(idx * step)] for idx in range(sample_size)]


def _activation_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"samples": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    ordered = sorted(values)
    return {
        "samples": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(sorted_values: list[int], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


class SpladeActivationStatsCallback:
    def __init__(
        self,
        *,
        model: Any,
        query_texts: list[str],
        document_texts: list[str],
        batch_size: int,
        interval_steps: int,
        quantization_factor: int,
        log_on_train_begin: bool,
        log_on_train_end: bool,
        prefix: str,
    ) -> None:
        self.model = model
        self.query_texts = query_texts
        self.document_texts = document_texts
        self.batch_size = batch_size
        self.interval_steps = interval_steps
        self.quantization_factor = quantization_factor
        self.log_on_train_begin = log_on_train_begin
        self.log_on_train_end = log_on_train_end
        self.prefix = prefix.rstrip("/")
        self._last_logged_step: int | None = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("on_"):
            def _noop(args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                return control

            return _noop
        raise AttributeError(name)

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.log_on_train_begin:
            self._log(getattr(state, "global_step", 0))
        return control

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        step = int(getattr(state, "global_step", 0) or 0)
        if step > 0 and step % self.interval_steps == 0:
            metrics = self._log(step)
            if logs is not None:
                logs.update(metrics)
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.log_on_train_end:
            self._log(getattr(state, "global_step", 0))
        return control

    def _log(self, step: int) -> dict[str, float | int]:
        step = int(step or 0)
        if self._last_logged_step == step:
            return {}
        self._last_logged_step = step
        metrics = self._compute_metrics()
        if not metrics:
            return metrics
        metrics[f"{self.prefix}/quantization_factor"] = self.quantization_factor
        self._log_to_wandb(metrics, step)
        return metrics

    def _compute_metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {}
        for kind, texts, encoder_name in (
            ("query", self.query_texts, "encode_query"),
            ("document", self.document_texts, "encode_document"),
        ):
            if not texts:
                continue
            counts = self._count_active_dims(texts, encoder_name)
            for stat_name, value in _activation_stats(counts).items():
                metrics[f"{self.prefix}/{kind}_active_dims_{stat_name}"] = value
        return metrics

    def _count_active_dims(self, texts: list[str], encoder_name: str) -> list[int]:
        import torch

        counts: list[int] = []
        was_training = bool(getattr(self.model, "training", False))
        if hasattr(self.model, "eval"):
            self.model.eval()
        try:
            for batch in _batch_texts(texts, self.batch_size):
                encoder = getattr(self.model, encoder_name, None) or getattr(self.model, "encode")
                embeddings = encoder(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    convert_to_sparse_tensor=False,
                    save_to_cpu=False,
                )
                tensor = _as_dense_tensor(embeddings, torch)
                # Keep the dense vocabulary-sized embeddings on the accelerator. Moving only
                # the final per-example counts avoids transferring and quantizing roughly
                # sample_size * vocab_size floats on the CPU for every stats interval.
                quantized = torch.round(tensor.float() * self.quantization_factor)
                counts.extend((quantized > 0).sum(dim=-1).cpu().tolist())
                del embeddings, tensor, quantized
        finally:
            if was_training and hasattr(self.model, "train"):
                self.model.train()
        return [int(value) for value in counts]

    def _log_to_wandb(self, metrics: dict[str, float | int], step: int) -> None:
        try:
            import wandb
        except ModuleNotFoundError:
            return
        if getattr(wandb, "run", None) is None:
            return
        wandb.log(metrics, step=step)


def _batch_texts(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[idx : idx + batch_size] for idx in range(0, len(values), batch_size)]


def _as_dense_tensor(embeddings: Any, torch: Any) -> Any:
    if isinstance(embeddings, list):
        tensors = [_as_dense_tensor(value, torch) for value in embeddings]
        return torch.stack(tensors)
    if not isinstance(embeddings, torch.Tensor):
        return torch.as_tensor(embeddings)
    if embeddings.is_sparse:
        return embeddings.to_dense()
    return embeddings


def _splade_activation_stats_config(backend_config: dict[str, Any]) -> dict[str, Any] | None:
    raw_config = backend_config.get("splade_activation_stats")
    if raw_config is None:
        return None
    if isinstance(raw_config, bool):
        if not raw_config:
            return None
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise SentenceTransformersConfigError("'sentence_transformers.splade_activation_stats' must be a mapping or boolean.")
    if not _as_bool(raw_config.get("enabled", True)):
        return None

    config = dict(raw_config)
    config["sample_size"] = int(config.get("sample_size", 128))
    config["batch_size"] = int(config.get("batch_size", 16))
    config["interval_steps"] = int(config.get("interval_steps", 500))
    config["quantization_factor"] = int(config.get("quantization_factor", 100))
    config["include_negatives"] = _as_bool(config.get("include_negatives", True))
    config["log_on_train_begin"] = _as_bool(config.get("log_on_train_begin", True))
    config["log_on_train_end"] = _as_bool(config.get("log_on_train_end", True))
    config["prefix"] = str(config.get("prefix", "train/splade_activation_stats"))
    for key in ("sample_size", "batch_size", "interval_steps", "quantization_factor"):
        if config[key] <= 0:
            raise SentenceTransformersConfigError(f"'sentence_transformers.splade_activation_stats.{key}' must be positive.")
    return config


def _build_splade_activation_stats_callback(
    model: Any,
    rows: list[dict[str, str]],
    backend_config: dict[str, Any],
) -> SpladeActivationStatsCallback | None:
    if not is_main_process():
        return None
    stats_config = _splade_activation_stats_config(backend_config)
    if stats_config is None:
        return None

    sample_size = int(stats_config["sample_size"])
    query_texts = _even_sample([row["anchor"] for row in rows if row.get("anchor")], sample_size)
    document_candidates = [row["positive"] for row in rows if row.get("positive")]
    if stats_config["include_negatives"]:
        for row in rows:
            for key, value in row.items():
                if key.startswith("negative_") and value:
                    document_candidates.append(value)
    document_texts = _even_sample(document_candidates, sample_size)
    return SpladeActivationStatsCallback(
        model=model,
        query_texts=query_texts,
        document_texts=document_texts,
        batch_size=int(stats_config["batch_size"]),
        interval_steps=int(stats_config["interval_steps"]),
        quantization_factor=int(stats_config["quantization_factor"]),
        log_on_train_begin=bool(stats_config["log_on_train_begin"]),
        log_on_train_end=bool(stats_config["log_on_train_end"]),
        prefix=str(stats_config["prefix"]),
    )


def _model_kwargs(backend_config: dict[str, Any]) -> dict[str, Any]:
    model_kwargs = backend_config.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise SentenceTransformersConfigError("'sentence_transformers.model_kwargs' must be a mapping when provided.")
    model_kwargs = dict(model_kwargs)
    if "trust_remote_code" in backend_config:
        model_kwargs.setdefault("trust_remote_code", bool(backend_config["trust_remote_code"]))
    return model_kwargs


def _model_cache_dir(backend_config: dict[str, Any]) -> str | None:
    for key in ("model_cache_dir", "cache_dir", "cache_folder"):
        value = backend_config.get(key)
        if value is not None:
            return str(value)
    return None


def _sentence_transformer_model_kwargs(backend_config: dict[str, Any]) -> dict[str, Any]:
    model_kwargs = _model_kwargs(backend_config)
    cache_dir = _model_cache_dir(backend_config)
    if cache_dir is not None:
        model_kwargs.setdefault("cache_folder", cache_dir)
    return model_kwargs


def _kwargs_mapping(backend_config: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = backend_config.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            joined = " / ".join(f"sentence_transformers.{item}" for item in keys)
            raise SentenceTransformersConfigError(f"'{joined}' must be a mapping when provided.")
        return dict(value)
    return {}


def _splade_pooling_kwargs(backend_config: dict[str, Any]) -> dict[str, Any]:
    pooling_kwargs: dict[str, Any] = {"pooling_strategy": str(backend_config.get("pooling_strategy", "max"))}
    if "splade_pooling_chunk_size" in backend_config:
        pooling_kwargs["chunk_size"] = int(backend_config["splade_pooling_chunk_size"])
    elif "chunk_size" in backend_config:
        pooling_kwargs["chunk_size"] = int(backend_config["chunk_size"])
    return pooling_kwargs


def _build_mlm_transformer(MLMTransformer: Any, model_name_or_path: str, mlm_kwargs: dict[str, Any]):
    try:
        from transformers import AutoProcessor, AutoTokenizer
    except ModuleNotFoundError:
        return MLMTransformer(str(model_name_or_path), **mlm_kwargs)

    original_from_pretrained = AutoProcessor.from_pretrained

    def from_pretrained_with_tokenizer_fallback(pretrained_model_name_or_path: str, *args: Any, **kwargs: Any):
        try:
            return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except ValueError as exc:
            if "Unrecognized processing class" not in str(exc):
                raise
            return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoProcessor.from_pretrained = from_pretrained_with_tokenizer_fallback
    try:
        return MLMTransformer(str(model_name_or_path), **mlm_kwargs)
    finally:
        AutoProcessor.from_pretrained = original_from_pretrained


def _load_training_rows(
    config: dict[str, Any],
    backend_config: dict[str, Any],
    *,
    config_path: str | None = None,
    use_score_labels: bool = False,
    score_normalization: str = "none",
    default_negatives_per_query: int | None = None,
) -> LoadedTrainingRows:
    sources = _resolve_train_data_sources(config, backend_config, config_path=config_path)
    negatives_per_query = backend_config.get("negatives_per_query", default_negatives_per_query)
    if negatives_per_query is not None:
        negatives_per_query = int(negatives_per_query)
    elif len(sources) > 1:
        negatives_per_query = min(_min_negatives_in_flagembedding_jsonl(source.path) for source in sources)
    query_prefix = str(
        backend_config.get(
            "query_prefix",
            _resolve_value(config, backend_config, "query_instruction_for_retrieval", ""),
        )
        or ""
    )
    passage_prefix = str(backend_config.get("passage_prefix", "") or "")

    rows: list[dict[str, Any]] = []
    dataset_ids: list[str] = []
    dedupe_values: list[set[str]] = []
    for source in sources:
        source_rows = load_flagembedding_jsonl_dataset(
            source.path,
            negatives_per_query=negatives_per_query,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            use_score_labels=use_score_labels,
            score_normalization=score_normalization,
        )
        rows.extend(source_rows)
        dataset_ids.extend([source.name] * len(source_rows))
        dedupe_values.extend(dedupe_values_from_training_row(row) for row in source_rows)
    return LoadedTrainingRows(sources=sources, rows=rows, dataset_ids=dataset_ids, dedupe_values=dedupe_values)


def _build_dense_model(SentenceTransformer: Any, model_name_or_path: str, backend_config: dict[str, Any]):
    model = SentenceTransformer(str(model_name_or_path), **_sentence_transformer_model_kwargs(backend_config))
    max_seq_length = backend_config.get("max_seq_length")
    if max_seq_length is not None:
        model.max_seq_length = int(max_seq_length)
    return model


def _build_splade_model(
    SparseEncoder: Any,
    MLMTransformer: Any,
    SpladePooling: Any,
    model_name_or_path: str,
    backend_config: dict[str, Any],
):
    model_kwargs = _model_kwargs(backend_config)
    processor_kwargs = _kwargs_mapping(backend_config, "processor_kwargs", "tokenizer_args")
    config_kwargs = _kwargs_mapping(backend_config, "config_kwargs", "config_args")
    cache_dir = _model_cache_dir(backend_config)
    if cache_dir is not None:
        model_kwargs.setdefault("cache_dir", cache_dir)
        processor_kwargs.setdefault("cache_dir", cache_dir)
        config_kwargs.setdefault("cache_dir", cache_dir)
    max_seq_length = backend_config.get("max_seq_length")
    signature = inspect.signature(MLMTransformer)
    parameters = signature.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    mlm_kwargs: dict[str, Any] = {}
    if model_kwargs and "model_args" in parameters:
        mlm_kwargs["model_args"] = model_kwargs
    elif model_kwargs and "model_kwargs" in parameters:
        mlm_kwargs["model_kwargs"] = model_kwargs
    elif model_kwargs and accepts_kwargs:
        mlm_kwargs["model_kwargs"] = model_kwargs
    if processor_kwargs and "tokenizer_args" in parameters:
        mlm_kwargs["tokenizer_args"] = processor_kwargs
    elif processor_kwargs and "processor_kwargs" in parameters:
        mlm_kwargs["processor_kwargs"] = processor_kwargs
    elif processor_kwargs and accepts_kwargs:
        mlm_kwargs["processor_kwargs"] = processor_kwargs
    if config_kwargs and "config_args" in parameters:
        mlm_kwargs["config_args"] = config_kwargs
    elif config_kwargs and "config_kwargs" in parameters:
        mlm_kwargs["config_kwargs"] = config_kwargs
    elif config_kwargs and accepts_kwargs:
        mlm_kwargs["config_kwargs"] = config_kwargs
    if "transformer_task" in parameters:
        mlm_kwargs["transformer_task"] = "fill-mask"
    if max_seq_length is not None and ("max_seq_length" in parameters or accepts_kwargs):
        mlm_kwargs["max_seq_length"] = int(max_seq_length)
    if "tokenizer_name_or_path" in backend_config and ("tokenizer_name_or_path" in parameters or accepts_kwargs):
        mlm_kwargs["tokenizer_name_or_path"] = backend_config["tokenizer_name_or_path"]

    mlm_transformer = _build_mlm_transformer(MLMTransformer, str(model_name_or_path), mlm_kwargs)
    model = SparseEncoder(modules=[mlm_transformer, SpladePooling(**_splade_pooling_kwargs(backend_config))])
    if max_seq_length is not None:
        model.max_seq_length = int(max_seq_length)
    return model


def _resolve_matryoshka_dims(backend_config: dict[str, Any], model: Any) -> list[int]:
    configured_dims = backend_config.get("matryoshka_dims")
    if configured_dims is None:
        embedding_dim = None
        if hasattr(model, "get_sentence_embedding_dimension"):
            embedding_dim = model.get_sentence_embedding_dimension()
        if embedding_dim is None:
            return [768, 384, 192]

        embedding_dim = int(embedding_dim)
        candidates = [embedding_dim, max(1, embedding_dim // 2), max(1, embedding_dim // 4)]
        return list(dict.fromkeys(candidates))

    if not isinstance(configured_dims, list):
        raise SentenceTransformersConfigError("'sentence_transformers.matryoshka_dims' must be a list of integers.")

    dims = [int(dim) for dim in configured_dims]
    if not dims or any(dim <= 0 for dim in dims):
        raise SentenceTransformersConfigError("'sentence_transformers.matryoshka_dims' must contain positive integers.")

    if hasattr(model, "get_sentence_embedding_dimension"):
        embedding_dim = model.get_sentence_embedding_dimension()
        if embedding_dim is not None and any(dim > int(embedding_dim) for dim in dims):
            raise SentenceTransformersConfigError(
                "'sentence_transformers.matryoshka_dims' cannot exceed the model embedding dimension "
                f"({embedding_dim})."
            )

    return dims


def run_embedder_training(request: TrainingRequest) -> int:
    if _skip_completed_training(request, default_output_dir="runs/sentence-transformers"):
        return 0

    (
        Dataset,
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        MultipleNegativesRankingLoss,
        _,
    ) = _load_sentence_transformers()

    config = request.config
    backend_config = _backend_config(config)
    model_name_or_path = _resolve_value(
        config,
        backend_config,
        "model_name_or_path",
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
    )
    output_dir = Path(str(_resolve_value(config, backend_config, "output_dir", "runs/sentence-transformers")))
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_from_checkpoint = resolve_resume_checkpoint(output_dir, config, backend_config, request.cli_args)

    loaded = _load_training_rows(config, backend_config, config_path=request.config_path)
    train_dataset = Dataset.from_list(loaded.rows)

    model = _build_dense_model(SentenceTransformer, str(model_name_or_path), backend_config)
    args = _training_args(output_dir, _training_args_config_for_rows(backend_config, loaded), SentenceTransformerTrainingArguments)
    loss = MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    _add_checkpoint_callbacks(trainer, output_dir, config, backend_config)

    print(
        "[INFO] Training SentenceTransformers embedder "
        f"with {len(loaded.rows)} examples from {loaded.source_label} and model {model_name_or_path}.",
        flush=True,
    )
    try:
        with wandb_run_environment(
            output_dir,
            report_to=backend_config.get("report_to", []),
            resume_from_checkpoint=resume_from_checkpoint,
            configured_run_id=backend_config.get("wandb_run_id"),
        ):
            train_with_resume(trainer, resume_from_checkpoint)
            _save_final_model(trainer, model, output_dir / "final", label="SentenceTransformer")
            _run_sentence_transformers_post_training_benchmarks(
                output_dir,
                config,
                backend_config,
                request,
                trainer=trainer,
                model=model,
            )
    finally:
        _finish_wandb_run(backend_config)

    return 0


def run_matryoshka_training(request: TrainingRequest) -> int:
    if _skip_completed_training(request, default_output_dir="runs/sentence-transformers-matryoshka"):
        return 0

    (
        Dataset,
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        MultipleNegativesRankingLoss,
        MatryoshkaLoss,
    ) = _load_sentence_transformers()

    config = request.config
    backend_config = _backend_config(config)
    model_name_or_path = _resolve_value(
        config,
        backend_config,
        "model_name_or_path",
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
    )
    output_dir = Path(str(_resolve_value(config, backend_config, "output_dir", "runs/sentence-transformers-matryoshka")))
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_from_checkpoint = resolve_resume_checkpoint(output_dir, config, backend_config, request.cli_args)

    loaded = _load_training_rows(config, backend_config, config_path=request.config_path)
    train_dataset = Dataset.from_list(loaded.rows)
    model = _build_dense_model(SentenceTransformer, str(model_name_or_path), backend_config)
    args = _training_args(output_dir, _training_args_config_for_rows(backend_config, loaded), SentenceTransformerTrainingArguments)

    matryoshka_kwargs: dict[str, Any] = {
        "matryoshka_dims": _resolve_matryoshka_dims(backend_config, model),
    }
    if "matryoshka_weights" in backend_config:
        matryoshka_kwargs["matryoshka_weights"] = backend_config["matryoshka_weights"]
    if "n_dims_per_step" in backend_config:
        matryoshka_kwargs["n_dims_per_step"] = int(backend_config["n_dims_per_step"])

    base_loss = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(model, base_loss, **matryoshka_kwargs)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    _add_checkpoint_callbacks(trainer, output_dir, config, backend_config)

    print(
        "[INFO] Training SentenceTransformers matryoshka "
        f"with {len(loaded.rows)} examples from {loaded.source_label}, model {model_name_or_path}, "
        f"and dims {matryoshka_kwargs['matryoshka_dims']}.",
        flush=True,
    )
    try:
        with wandb_run_environment(
            output_dir,
            report_to=backend_config.get("report_to", []),
            resume_from_checkpoint=resume_from_checkpoint,
            configured_run_id=backend_config.get("wandb_run_id"),
        ):
            train_with_resume(trainer, resume_from_checkpoint)
            _save_final_model(trainer, model, output_dir / "final", label="SentenceTransformer")
            _run_sentence_transformers_post_training_benchmarks(
                output_dir,
                config,
                backend_config,
                request,
                trainer=trainer,
                model=model,
            )
    finally:
        _finish_wandb_run(backend_config)

    return 0


def run_splade_training(request: TrainingRequest) -> int:
    if _skip_completed_training(request, default_output_dir="runs/sentence-transformers-splade"):
        return 0

    (
        Dataset,
        SparseEncoder,
        SparseEncoderTrainer,
        SparseEncoderTrainingArguments,
        SparseMarginMSELoss,
        SparseMultipleNegativesRankingLoss,
        SpladeLoss,
        MLMTransformer,
        SpladePooling,
    ) = _load_sparse_sentence_transformers()

    config = request.config
    backend_config = _backend_config(config)
    model_name_or_path = _resolve_value(
        config,
        backend_config,
        "model_name_or_path",
        "hf-internal-testing/tiny-random-BertForMaskedLM",
    )
    output_dir = Path(str(_resolve_value(config, backend_config, "output_dir", "runs/sentence-transformers-splade")))
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_from_checkpoint = resolve_resume_checkpoint(output_dir, config, backend_config, request.cli_args)

    loss_name = _splade_base_loss_name(backend_config)
    loaded = _load_training_rows(
        config,
        backend_config,
        config_path=request.config_path,
        use_score_labels=loss_name == "sparse_margin_mse",
        score_normalization=str(backend_config.get("score_normalization", "none")),
    )
    train_dataset = Dataset.from_list(loaded.rows)
    model = _build_splade_model(SparseEncoder, MLMTransformer, SpladePooling, str(model_name_or_path), backend_config)

    args = _training_args(output_dir, _training_args_config_for_rows(backend_config, loaded), SparseEncoderTrainingArguments)
    if loss_name == "sparse_margin_mse":
        ranking_loss = SparseMarginMSELoss(model)
    else:
        ranking_loss = SparseMultipleNegativesRankingLoss(
            model,
            scale=float(backend_config.get("scale", 1.0)),
            gather_across_devices=bool(backend_config.get("gather_across_devices", False)),
        )
    loss = SpladeLoss(
        model,
        loss=ranking_loss,
        document_regularizer_weight=float(backend_config.get("document_regularizer_weight", 3e-5)),
        query_regularizer_weight=float(backend_config.get("query_regularizer_weight", 5e-5)),
    )
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    activation_stats_callback = _build_splade_activation_stats_callback(model, loaded.rows, backend_config)
    if activation_stats_callback is not None and hasattr(trainer, "add_callback"):
        trainer.add_callback(activation_stats_callback)
    _add_checkpoint_callbacks(trainer, output_dir, config, backend_config)

    print(
        "[INFO] Training SentenceTransformers SPLADE "
        f"with {len(loaded.rows)} examples from {loaded.source_label} and model {model_name_or_path}.",
        flush=True,
    )
    try:
        with wandb_run_environment(
            output_dir,
            report_to=backend_config.get("report_to", []),
            resume_from_checkpoint=resume_from_checkpoint,
            configured_run_id=backend_config.get("wandb_run_id"),
        ):
            train_with_resume(trainer, resume_from_checkpoint)
            final_dir = output_dir / "final"
            _save_final_model(trainer, model, final_dir, label="SparseEncoder")
            _run_sentence_transformers_post_training_benchmarks(
                output_dir,
                config,
                backend_config,
                request,
                trainer=trainer,
                model=model,
            )
    finally:
        _finish_wandb_run(backend_config)

    return 0


def run_training(request: TrainingRequest) -> int:
    if request.training_type == "embedder":
        return run_embedder_training(request)
    if request.training_type == "matryoshka":
        return run_matryoshka_training(request)
    if request.training_type == "splade":
        return run_splade_training(request)

    _require_sentence_transformers()
    raise NotImplementedError(
        f"Training type '{request.training_type}' for backend 'sentence-transformers' is registered "
        "as an extension point, but this project does not implement its training loop yet."
    )
