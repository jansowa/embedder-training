"""SentenceTransformers training backend."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

from training.backends.registry import BackendDependencyError, TrainingRequest
from training.dataset_filters import apply_dataset_filter_if_configured


class SentenceTransformersConfigError(ValueError):
    """Raised when a SentenceTransformers config cannot drive training."""


COMMON_TRAINING_KEYS = {
    "bf16",
    "dataloader_drop_last",
    "dataloader_num_workers",
    "fp16",
    "gradient_accumulation_steps",
    "learning_rate",
    "logging_steps",
    "max_seq_length",
    "max_steps",
    "model_kwargs",
    "model_name_or_path",
    "num_train_epochs",
    "output_dir",
    "overwrite_output_dir",
    "per_device_train_batch_size",
    "processor_kwargs",
    "query_instruction_for_retrieval",
    "report_to",
    "run_name",
    "save_steps",
    "save_strategy",
    "save_total_limit",
    "seed",
    "tokenizer_args",
    "tokenizer_name_or_path",
    "train_batch_size",
    "train_data",
    "trust_remote_code",
    "warmup_ratio",
    "weight_decay",
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


def _resolve_train_data_path(config: dict[str, Any], backend_config: dict[str, Any], *, config_path: str | None = None) -> Path:
    train_data = _resolve_value(config, backend_config, "train_data")
    if not train_data:
        raise SentenceTransformersConfigError("SentenceTransformers training requires 'train_data'.")

    filter_result = apply_dataset_filter_if_configured(
        train_data,
        config,
        backend_config,
        config_path=config_path,
    )
    if filter_result is not None:
        return filter_result.output_path

    path = Path(str(train_data))
    if path.is_dir():
        preferred_files = ("dataset.jsonl", "mixed_dataset.jsonl")
        for filename in preferred_files:
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


def _ensure_text_list(value: Any, field_name: str, line_no: int) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return value
    raise SentenceTransformersConfigError(f"Line {line_no}: field '{field_name}' must be a non-empty string or list of strings.")


def _prefix_text(prefix: str, text: str) -> str:
    return f"{prefix}{text}" if prefix else text


def load_flagembedding_jsonl_dataset(
    data_path: Path,
    *,
    negatives_per_query: int | None,
    query_prefix: str,
    passage_prefix: str,
) -> list[dict[str, str]]:
    parsed_items: list[tuple[int, str, list[str], list[str]]] = []
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

            parsed_items.append((line_no, query, positives, negatives))
            min_negatives = len(negatives) if min_negatives is None else min(min_negatives, len(negatives))

    if not parsed_items:
        raise SentenceTransformersConfigError(f"Training data file '{data_path}' did not yield any examples.")

    effective_negatives = min_negatives if negatives_per_query is None else negatives_per_query
    if effective_negatives is None or effective_negatives <= 0:
        raise SentenceTransformersConfigError("'negatives_per_query' must be greater than zero.")

    rows: list[dict[str, str]] = []
    for line_no, query, positives, negatives in parsed_items:
        if len(negatives) < effective_negatives:
            raise SentenceTransformersConfigError(
                f"Line {line_no}: expected at least {effective_negatives} negatives, got {len(negatives)}."
            )

        selected_negatives = negatives[:effective_negatives]
        for positive in positives:
            row = {
                "anchor": _prefix_text(query_prefix, query),
                "positive": _prefix_text(passage_prefix, positive),
            }
            for idx, negative in enumerate(selected_negatives, start=1):
                row[f"negative_{idx}"] = _prefix_text(passage_prefix, negative)
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
    if backend_config.get("run_name") is not None:
        kwargs["run_name"] = str(backend_config["run_name"])
    return training_arguments_cls(**kwargs)


def _reports_to_wandb(report_to: Any) -> bool:
    if report_to is None:
        return False
    if isinstance(report_to, str):
        values = [report_to]
    else:
        try:
            values = list(report_to)
        except TypeError:
            values = [report_to]
    return any(str(value).lower() in {"all", "wandb"} for value in values)


def _finish_wandb_run(backend_config: dict[str, Any]) -> None:
    if not _reports_to_wandb(backend_config.get("report_to", [])):
        return
    try:
        import wandb
    except ModuleNotFoundError:
        return
    if getattr(wandb, "run", None) is not None:
        wandb.finish()


def _model_kwargs(backend_config: dict[str, Any]) -> dict[str, Any]:
    model_kwargs = backend_config.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise SentenceTransformersConfigError("'sentence_transformers.model_kwargs' must be a mapping when provided.")
    model_kwargs = dict(model_kwargs)
    if "trust_remote_code" in backend_config:
        model_kwargs.setdefault("trust_remote_code", bool(backend_config["trust_remote_code"]))
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
) -> tuple[Path, list[dict[str, str]]]:
    data_path = _resolve_train_data_path(config, backend_config, config_path=config_path)
    negatives_per_query = backend_config.get("negatives_per_query")
    if negatives_per_query is not None:
        negatives_per_query = int(negatives_per_query)
    query_prefix = str(
        backend_config.get(
            "query_prefix",
            _resolve_value(config, backend_config, "query_instruction_for_retrieval", ""),
        )
        or ""
    )
    passage_prefix = str(backend_config.get("passage_prefix", "") or "")

    rows = load_flagembedding_jsonl_dataset(
        data_path,
        negatives_per_query=negatives_per_query,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
    return data_path, rows


def _build_dense_model(SentenceTransformer: Any, model_name_or_path: str, backend_config: dict[str, Any]):
    model = SentenceTransformer(str(model_name_or_path), **_model_kwargs(backend_config))
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

    data_path, rows = _load_training_rows(config, backend_config, config_path=request.config_path)
    train_dataset = Dataset.from_list(rows)

    model = _build_dense_model(SentenceTransformer, str(model_name_or_path), backend_config)
    args = _training_args(output_dir, backend_config, SentenceTransformerTrainingArguments)
    loss = MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    print(
        "[INFO] Training SentenceTransformers embedder "
        f"with {len(rows)} examples from {data_path} and model {model_name_or_path}.",
        flush=True,
    )
    try:
        trainer.train()
    finally:
        _finish_wandb_run(backend_config)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    print(f"[INFO] Saved final SentenceTransformer model to: {final_dir}", flush=True)
    return 0


def run_matryoshka_training(request: TrainingRequest) -> int:
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

    data_path, rows = _load_training_rows(config, backend_config, config_path=request.config_path)
    train_dataset = Dataset.from_list(rows)
    model = _build_dense_model(SentenceTransformer, str(model_name_or_path), backend_config)
    args = _training_args(output_dir, backend_config, SentenceTransformerTrainingArguments)

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

    print(
        "[INFO] Training SentenceTransformers matryoshka "
        f"with {len(rows)} examples from {data_path}, model {model_name_or_path}, "
        f"and dims {matryoshka_kwargs['matryoshka_dims']}.",
        flush=True,
    )
    try:
        trainer.train()
    finally:
        _finish_wandb_run(backend_config)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    print(f"[INFO] Saved final SentenceTransformer model to: {final_dir}", flush=True)
    return 0


def run_splade_training(request: TrainingRequest) -> int:
    (
        Dataset,
        SparseEncoder,
        SparseEncoderTrainer,
        SparseEncoderTrainingArguments,
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

    data_path, rows = _load_training_rows(config, backend_config, config_path=request.config_path)
    train_dataset = Dataset.from_list(rows)
    model = _build_splade_model(SparseEncoder, MLMTransformer, SpladePooling, str(model_name_or_path), backend_config)

    args = _training_args(output_dir, backend_config, SparseEncoderTrainingArguments)
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

    print(
        "[INFO] Training SentenceTransformers SPLADE "
        f"with {len(rows)} examples from {data_path} and model {model_name_or_path}.",
        flush=True,
    )
    try:
        trainer.train()
    finally:
        _finish_wandb_run(backend_config)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    print(f"[INFO] Saved final SparseEncoder model to: {final_dir}", flush=True)
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
