"""FlagEmbedding training backend."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
from time import time
from typing import Any

from training.backends.registry import BackendDependencyError, TrainingRequest


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
    "benchmark_name",
    "flagembedding",
    "hparams",
    "output_dir",
    "pylate",
    "pirb_scope",
    "remove_checkpoints",
    "run_mteb",
    "run_pirb",
    "runs_dir",
    "sentence_transformers",
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


def _grid_from_config(config: dict[str, Any]) -> dict[str, Any]:
    backend_config = _dict_section(config, "backend_config") | _dict_section(config, "flagembedding")
    return {
        "architectures": config.get(
            "architectures",
            backend_config.get("architectures", DEFAULT_GRID["architectures"]),
        ),
        "hparams": config.get(
            "hparams",
            backend_config.get("hparams", DEFAULT_GRID["hparams"]),
        ),
    }


def _base_training_args(config: dict[str, Any]) -> dict[str, Any]:
    root_overrides = {key: value for key, value in config.items() if key not in RESERVED_CONFIG_KEYS}
    backend_overrides = _dict_section(config, "backend_config") | _dict_section(config, "flagembedding")
    backend_overrides = {
        key: value
        for key, value in backend_overrides.items()
        if key not in {"architectures", "hparams", "runs_dir", "wandb_project"}
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


def _benchmark_model(st_model_dir: str, epoch_idx: int, query_instruction: str, request: TrainingRequest, tasks) -> None:
    from convert_utils import run_mteb, run_pirb

    metrics_to_log: dict[str, float] = {}
    prefix = f"epoch{epoch_idx}/" if epoch_idx is not None else ""

    if request.cli_args.run_mteb:
        metrics_mteb = run_mteb(st_model_dir, tasks)
        metrics_to_log.update({f"{prefix}{key}": value for key, value in metrics_mteb.items()})

    if request.cli_args.run_pirb:
        metrics_pirb = run_pirb(
            st_model_dir,
            query_instruction_for_retrieval=query_instruction,
            scope=request.cli_args.pirb_scope,
        )
        metrics_to_log.update({f"{prefix}{key}": value for key, value in metrics_pirb.items()})

    if metrics_to_log:
        wandb = _load_wandb()
        wandb.log(metrics_to_log, step=epoch_idx or 0)


def _benchmark_checkpoints(output_dir: Path, arch: str, full_args: dict[str, Any], request: TrainingRequest) -> None:
    from convert_utils import convert_to_sentence_transformer

    tasks = None
    if request.cli_args.run_mteb:
        tasks = _load_mteb().get_benchmarks(names=[request.cli_args.benchmark_name])

    ckpt_dirs = sorted(output_dir.glob("checkpoint-*"), key=lambda path: path.stat().st_mtime)
    query_instruction = full_args.get(
        "query_instruction_for_retrieval",
        QUERY_INSTRUCTION_FOR_RETRIEVAL_DEFAULT,
    )
    pooling_method = full_args.get("sentence_pooling_method")

    if not ckpt_dirs:
        st_dir = output_dir / "base-st"
        convert_to_sentence_transformer(arch, str(st_dir), pooling_method=pooling_method)
        _benchmark_model(str(st_dir.resolve()), epoch_idx=0, query_instruction=query_instruction, request=request, tasks=tasks)
        return

    for idx, ckpt in enumerate(ckpt_dirs, start=1):
        st_dir = ckpt.with_name(f"{ckpt.name}-st")
        convert_to_sentence_transformer(str(ckpt), str(st_dir), pooling_method=pooling_method)
        _benchmark_model(str(st_dir.resolve()), epoch_idx=idx, query_instruction=query_instruction, request=request, tasks=tasks)


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
    runs_dir = Path(config.get("runs_dir") or _dict_section(config, "flagembedding").get("runs_dir") or "runs")
    runs_dir.mkdir(exist_ok=True)
    wandb_project = config.get("wandb_project") or os.getenv("WANDB_PROJECT", "mining-tests")

    for arch in grid["architectures"]:
        for hparams in grid["hparams"]:
            full_args = {**base_args, **hparams}
            lr = full_args.get("learning_rate")
            epochs = full_args.get("num_train_epochs")
            dataset_path = full_args.get("train_data")
            safe_arch = arch.replace("/", "_").replace(".", "_")
            run_name = f"{safe_arch}-{lr}lr-{epochs}ep-{dataset_path}-{int(time())}"

            run = wandb.init(project=wandb_project, name=run_name, config={**full_args, "arch": arch})
            output_dir = runs_dir / run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "torchrun",
                "--nproc_per_node",
                "1",
                "-m",
                "FlagEmbedding.finetune.embedder.encoder_only.base",
                "--model_name_or_path",
                arch,
                "--overwrite_output_dir",
                "--output_dir",
                str(output_dir),
                "--report_to",
                "wandb",
                "--run_name",
                run_name,
                "--trust_remote_code",
                "True",
            ]
            for key, value in full_args.items():
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
            subprocess.run(cmd, check=True, env=env)

            if request.cli_args.run_mteb or request.cli_args.run_pirb:
                _benchmark_checkpoints(output_dir, arch, full_args, request)

            if request.cli_args.remove_checkpoints:
                print(f"[INFO] Removing checkpoints from {output_dir}...", flush=True)
                for ckpt_dir in output_dir.glob("checkpoint-*"):
                    shutil.rmtree(ckpt_dir)

            run.finish()

    return 0
