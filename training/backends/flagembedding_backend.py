"""FlagEmbedding training backend."""

from __future__ import annotations

import importlib.util
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from time import time
from typing import Any

from training.benchmarks import resolve_benchmark_settings, run_benchmarks_for_model
from training.backends.registry import BackendDependencyError, TrainingRequest
from training.checkpoints import resolve_resume_checkpoint
from training.dataset_filters import apply_dataset_filter_if_configured
from training.dataset_sources import HF_DATASET_CACHE_CONFIG_KEYS, resolve_train_data_entry
from training.distributed import resolve_distributed_config
from training.multi_dataset import (
    PROPORTIONAL_BATCH_BEST_EFFORT,
    is_multi_train_data,
    materialize_mixed_jsonl_dataset,
    normalize_dataset_mix_strategy,
    normalize_train_data_entries,
)


DEFAULT_GRID = {
    "architectures": [
        "answerdotai/ModernBERT-base",
        "answerdotai/ModernBERT-large",
    ],
    "hparams": [
        {"learning_rate": 2.5e-5, "num_train_epochs": 2},
    ],
}

QUERY_INSTRUCTION_FOR_RETRIEVAL_DEFAULT = "query: "
DATASET_FILTER_ARG_KEYS = {"dataset_filter", "dataset_filter_cache_dir"}
MIXED_DATASET_ARG_KEYS = {"dataset_mix_strategy", "mixed_dataset_cache_dir"}
TRAINING_CONTROL_ARG_KEYS = (
    DATASET_FILTER_ARG_KEYS
    | MIXED_DATASET_ARG_KEYS
    | HF_DATASET_CACHE_CONFIG_KEYS
    | {"distributed", "gpus", "num_gpus"}
)

STATIC_ARGS = {
    "cache_dir": "./cache/model",
    "train_data": "./dataset-no_in_batch_neg",
    "cache_path": "./cache/data",
    "train_group_size": 6,
    "query_max_len": 512,
    "passage_max_len": 512,
    "pad_to_multiple_of": 8,
    "query_instruction_for_retrieval": QUERY_INSTRUCTION_FOR_RETRIEVAL_DEFAULT,
    "query_instruction_format": "{}{}",
    "knowledge_distillation": True,
    "fp16": False,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "dataloader_drop_last": True,
    "warmup_ratio": 0.1,
    "gradient_checkpointing": True,
    "deepspeed": "./ds_stage0.json",
    "logging_steps": 1,
    "save_strategy": "epoch",
    "negatives_cross_device": True,
    "temperature": 0.02,
    "sentence_pooling_method": "cls",
    "normalize_embeddings": True,
    "kd_loss_type": "kl_div",
}

RESERVED_CONFIG_KEYS = {
    "architectures",
    "backend",
    "backend_config",
    "benchmark",
    "benchmark_batch_size",
    "benchmark_name",
    "benchmark_output_dir",
    "benchmark_query_instruction",
    "distributed",
    "dataset_mix_strategy",
    "epoch_checkpoint_dir",
    "flagembedding",
    "grid_architecture",
    "grid_hparams",
    "grid_train_data_group",
    "hparams",
    "hf_dataset_cache_dir",
    "huggingface_dataset_cache_dir",
    "keep_epoch_checkpoints",
    "model_name_or_path",
    "mixed_dataset_cache_dir",
    "output_dir",
    "pylate",
    "pirb_batch_size",
    "pirb_max_seq_length",
    "pirb_scope",
    "pirb_threads",
    "remove_checkpoints",
    "resume",
    "resume_from_checkpoint",
    "run_name",
    "run_mteb",
    "run_pirb",
    "runs_dir",
    "sentence_transformers",
    "train_data_groups",
    "training-type",
    "training_type",
    "wandb_project",
}


def _require_flagembedding() -> None:
    if importlib.util.find_spec("FlagEmbedding") is None:
        raise BackendDependencyError(
            "Backend 'flagembedding' requires the FlagEmbedding dependency. "
            "Install it with: pip install -r requirements/requirements-flagembedding.txt"
        )


def _load_wandb():
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise BackendDependencyError(
            "FlagEmbedding training requires wandb for the current pipeline. "
            "Install it with: pip install -r requirements/requirements-flagembedding.txt"
        ) from exc
    return wandb


def _load_mteb():
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise BackendDependencyError(
            "MTEB benchmarking requires mteb. "
            "Install it with: pip install -r requirements/requirements-flagembedding.txt"
        ) from exc
    return mteb


def _dict_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _safe_slug(value: Any, *, max_length: int = 80) -> str:
    raw_value = str(value)
    allowed = []
    for char in raw_value.strip().replace("\\", "/"):
        if char.isalnum() or char in {"_", "-"}:
            allowed.append(char)
        else:
            allowed.append("-")
    slug = "".join(allowed).strip("-_") or "run"
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(raw_value.encode("utf-8")).hexdigest()[:10]
    return f"{slug[: max_length - 11].rstrip('-_')}-{digest}"


def _grid_from_config(config: dict[str, Any]) -> dict[str, Any]:
    backend_config = _dict_section(config, "backend_config") | _dict_section(config, "flagembedding")
    configured_model = config.get("model_name_or_path", backend_config.get("model_name_or_path"))
    configured_architectures = config.get("architectures", backend_config.get("architectures"))
    configured_hparams = config.get("hparams", backend_config.get("hparams"))

    if configured_architectures is None and configured_model:
        configured_architectures = [configured_model]
    if configured_hparams is None and configured_model:
        configured_hparams = [{}]

    return {
        "architectures": config.get(
            "architectures",
            configured_architectures or DEFAULT_GRID["architectures"],
        ),
        "hparams": config.get(
            "hparams",
            configured_hparams or DEFAULT_GRID["hparams"],
        ),
    }


def _base_training_args(config: dict[str, Any]) -> dict[str, Any]:
    root_overrides = {key: value for key, value in config.items() if key not in RESERVED_CONFIG_KEYS}
    backend_overrides = _dict_section(config, "backend_config") | _dict_section(config, "flagembedding")
    backend_overrides = {
        key: value
        for key, value in backend_overrides.items()
        if key not in RESERVED_CONFIG_KEYS
    }
    return {**STATIC_ARGS, **root_overrides, **backend_overrides}


def _append_training_arg(cmd: list[str], key: str, value: Any) -> None:
    flag = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _benchmark_model(
    st_model_dir: str,
    epoch_idx: int,
    query_instruction: str,
    request: TrainingRequest,
    full_args: dict[str, Any],
    *,
    label: str,
) -> None:
    prefix = f"epoch{epoch_idx}/" if epoch_idx is not None else ""
    settings = resolve_benchmark_settings(
        request.config,
        full_args,
        request.cli_args,
        default_query_instruction=query_instruction,
    )
    run_benchmarks_for_model(
        st_model_dir,
        settings,
        metric_prefix=prefix,
        step=epoch_idx or 0,
        label=label,
    )


def _benchmark_checkpoints(output_dir: Path, arch: str, full_args: dict[str, Any], request: TrainingRequest) -> None:
    from convert_utils import convert_to_sentence_transformer

    ckpt_dirs = sorted(output_dir.glob("checkpoint-*"), key=lambda path: path.stat().st_mtime)
    query_instruction = full_args.get(
        "query_instruction_for_retrieval",
        QUERY_INSTRUCTION_FOR_RETRIEVAL_DEFAULT,
    )
    pooling_method = full_args.get("sentence_pooling_method")

    if not ckpt_dirs:
        st_dir = output_dir / "base-st"
        convert_to_sentence_transformer(arch, str(st_dir), pooling_method=pooling_method)
        _benchmark_model(
            str(st_dir.resolve()),
            epoch_idx=0,
            query_instruction=query_instruction,
            request=request,
            full_args=full_args,
            label="base",
        )
        return

    for idx, ckpt in enumerate(ckpt_dirs, start=1):
        st_dir = ckpt.with_name(f"{ckpt.name}-st")
        convert_to_sentence_transformer(str(ckpt), str(st_dir), pooling_method=pooling_method)
        _benchmark_model(
            str(st_dir.resolve()),
            epoch_idx=idx,
            query_instruction=query_instruction,
            request=request,
            full_args=full_args,
            label=f"epoch{idx}",
        )


def run_training(request: TrainingRequest) -> int:
    if request.training_type != "embedder":
        raise NotImplementedError(
            f"Training type '{request.training_type}' is registered for FlagEmbedding but is not implemented yet."
        )

    _require_flagembedding()
    wandb = _load_wandb()

    config = request.config
    grid = _grid_from_config(config)
    base_args = _base_training_args(config)
    distributed_config = resolve_distributed_config(
        backend=request.backend,
        config=config,
        cli_args=request.cli_args,
    )
    runs_dir = Path(config.get("runs_dir") or _dict_section(config, "flagembedding").get("runs_dir") or "runs")
    runs_dir.mkdir(exist_ok=True)
    wandb_project = config.get("wandb_project") or os.getenv("WANDB_PROJECT", "mining-tests")

    for arch in grid["architectures"]:
        for hparams in grid["hparams"]:
            full_args = {**base_args, **hparams}
            model_cache_dir = full_args.pop("model_cache_dir", None)
            if model_cache_dir is not None:
                full_args["cache_dir"] = str(model_cache_dir)
            mix_strategy = normalize_dataset_mix_strategy(
                full_args.get(
                    "dataset_mix_strategy",
                    config.get(
                        "dataset_mix_strategy",
                        _dict_section(config, "backend_config").get(
                            "dataset_mix_strategy",
                            _dict_section(config, "flagembedding").get("dataset_mix_strategy"),
                        ),
                    ),
                )
            )
            if mix_strategy == PROPORTIONAL_BATCH_BEST_EFFORT:
                raise NotImplementedError(
                    "'dataset_mix_strategy: proportional_batch_best_effort' is not supported for backend "
                    "'flagembedding' because it runs training in an external process."
                )

            train_data_entries = normalize_train_data_entries(full_args.get("train_data"))
            if is_multi_train_data(full_args.get("train_data")):
                grid_group = config.get("grid_train_data_group") if isinstance(config.get("grid_train_data_group"), dict) else {}
                mixed_result = materialize_mixed_jsonl_dataset(
                    full_args.get("train_data"),
                    full_args,
                    config,
                    config_path=request.config_path,
                    cache_dir=full_args.get(
                        "mixed_dataset_cache_dir",
                        config.get(
                            "mixed_dataset_cache_dir",
                            _dict_section(config, "backend_config").get(
                                "mixed_dataset_cache_dir",
                                _dict_section(config, "flagembedding").get("mixed_dataset_cache_dir"),
                            ),
                        ),
                    ),
                    group_name=str(grid_group.get("slug") or config.get("run_name") or "mixed"),
                )
                full_args["train_data"] = str(mixed_result.output_dir)
            else:
                full_args["train_data"] = str(resolve_train_data_entry(train_data_entries[0], full_args, config))
                filter_result = apply_dataset_filter_if_configured(
                    full_args.get("train_data"),
                    full_args,
                    config_path=request.config_path,
                )
                if filter_result is not None:
                    full_args["train_data"] = str(filter_result.output_dir)
            lr = full_args.get("learning_rate")
            epochs = full_args.get("num_train_epochs")
            dataset_path = full_args.get("train_data")
            safe_arch = _safe_slug(arch)
            safe_dataset = _safe_slug(dataset_path)
            configured_output_dir = config.get("output_dir")
            if configured_output_dir is not None:
                output_dir = Path(str(configured_output_dir))
                run_name = str(config.get("run_name") or output_dir.name)
            else:
                run_name = str(config.get("run_name") or f"{safe_arch}-{lr}lr-{epochs}ep-{safe_dataset}-{int(time())}")
                output_dir = runs_dir / run_name
            backend_resume_config = (
                {key: config[key] for key in ("resume", "resume_from_checkpoint", "epoch_checkpoint_dir") if key in config}
                | _dict_section(config, "backend_config")
                | _dict_section(config, "flagembedding")
            )
            resume_from_checkpoint = resolve_resume_checkpoint(
                output_dir,
                config,
                backend_resume_config,
                request.cli_args,
            )

            run = wandb.init(project=wandb_project, name=run_name, config={**full_args, "arch": arch})
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "torchrun",
                "--nproc_per_node",
                str(distributed_config.nproc_per_node),
                "-m",
                "FlagEmbedding.finetune.embedder.encoder_only.base",
                "--model_name_or_path",
                arch,
                "--output_dir",
                str(output_dir),
                "--report_to",
                "wandb",
                "--run_name",
                run_name,
                "--trust_remote_code",
                "True",
            ]
            if resume_from_checkpoint is None:
                cmd.append("--overwrite_output_dir")
            else:
                cmd.extend(["--resume-from-checkpoint", resume_from_checkpoint])
            for key, value in full_args.items():
                if key in TRAINING_CONTROL_ARG_KEYS:
                    continue
                _append_training_arg(cmd, key, value)

            print(">>> LAUNCH:", " ".join(cmd), flush=True)
            env = os.environ.copy()
            env.update(
                {
                    "WANDB_PROJECT": wandb_project,
                    "WANDB_NAME": run_name,
                    "WANDB_RUN_GROUP": safe_arch,
                }
            )
            if distributed_config.cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = distributed_config.cuda_visible_devices
            subprocess.run(cmd, check=True, env=env)

            benchmark_settings = resolve_benchmark_settings(
                config,
                full_args,
                request.cli_args,
                default_query_instruction=str(
                    full_args.get("query_instruction_for_retrieval", QUERY_INSTRUCTION_FOR_RETRIEVAL_DEFAULT)
                    or ""
                ),
            )
            if benchmark_settings.enabled:
                _benchmark_checkpoints(output_dir, arch, full_args, request)

            if request.cli_args.remove_checkpoints:
                print(f"[INFO] Removing checkpoints from {output_dir}...", flush=True)
                for ckpt_dir in output_dir.glob("checkpoint-*"):
                    shutil.rmtree(ckpt_dir)

            run.finish()

    return 0
