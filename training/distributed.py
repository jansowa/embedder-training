"""Distributed launch and GPU selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import subprocess
import time
from typing import Any, Sequence

from training.backends.registry import TrainingCliError


AUTO_DISTRIBUTED_BACKENDS = {"sentence-transformers", "pylate"}
RUN_TIMESTAMP_ENV = "EMBEDDER_TRAINING_RUN_TIMESTAMP"


class DistributedConfigError(TrainingCliError):
    """Raised when distributed/GPU configuration is invalid."""


@dataclass(frozen=True)
class DistributedLaunchConfig:
    enabled: bool
    nproc_per_node: int
    cuda_visible_devices: str | None
    launcher: str = "torchrun"
    standalone: bool = True


def _dict_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _merged_distributed_section(config: dict[str, Any], backend: str) -> dict[str, Any]:
    backend_section_key = {
        "flagembedding": "flagembedding",
        "sentence-transformers": "sentence_transformers",
        "pylate": "pylate",
    }.get(backend)
    merged = dict(_dict_section(config, "distributed"))
    backend_config = _dict_section(config, "backend_config")
    if isinstance(backend_config.get("distributed"), dict):
        merged.update(backend_config["distributed"])
    if backend_section_key:
        backend_section = _dict_section(config, backend_section_key)
        if isinstance(backend_section.get("distributed"), dict):
            merged.update(backend_section["distributed"])
    return merged


def _as_bool_or_auto(value: Any) -> bool | str:
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise DistributedConfigError("'distributed.enabled' must be one of: auto, true, false.")


def _visible_cuda_devices_from_env() -> list[str] | None:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    if not raw_value or raw_value == "-1":
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def cuda_device_count() -> int:
    visible_devices = _visible_cuda_devices_from_env()
    if visible_devices is not None:
        return len(visible_devices)

    try:
        import torch
    except Exception:
        return 0

    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _first_visible_devices(count: int) -> list[str]:
    if count < 0:
        raise DistributedConfigError("'distributed.num_gpus' must be greater than or equal to zero.")
    visible_devices = _visible_cuda_devices_from_env()
    if visible_devices is not None:
        if count > len(visible_devices):
            raise DistributedConfigError(
                f"Requested {count} GPUs, but CUDA_VISIBLE_DEVICES exposes only {len(visible_devices)}."
            )
        return visible_devices[:count]
    return [str(index) for index in range(count)]


def _parse_gpu_ids(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() == "auto":
            return None
        return [item.strip() for item in normalized.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        gpu_ids = [str(item).strip() for item in value if str(item).strip()]
        return gpu_ids
    raise DistributedConfigError(
        "'distributed.gpus' must be 'auto', a comma-separated string, or a YAML list of GPU ids. "
        "Use 'distributed.num_gpus' for a GPU count."
    )


def _parse_num_gpus(value: Any) -> int | None:
    if value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise DistributedConfigError("'distributed.num_gpus' must be an integer.") from exc
    if count < 0:
        raise DistributedConfigError("'distributed.num_gpus' must be greater than or equal to zero.")
    return count


def _resolve_gpu_selection(section: dict[str, Any], cli_args: Any) -> tuple[int, str | None]:
    cli_gpus = getattr(cli_args, "gpus", None)
    cli_num_gpus = getattr(cli_args, "num_gpus", None)
    config_gpus = section.get("gpus", section.get("gpu_ids"))
    config_num_gpus = section.get("num_gpus")

    if cli_gpus is not None and cli_num_gpus is not None:
        raise DistributedConfigError("Use either --gpus or --num-gpus, not both.")
    if cli_gpus is None and cli_num_gpus is None and config_gpus is not None and config_num_gpus is not None:
        raise DistributedConfigError("Use either distributed.gpus or distributed.num_gpus, not both.")

    if cli_gpus is not None:
        gpu_ids = _parse_gpu_ids(cli_gpus)
        if gpu_ids is None:
            return cuda_device_count(), None
        return len(gpu_ids), ",".join(gpu_ids)

    if cli_num_gpus is not None:
        count = _parse_num_gpus(cli_num_gpus) or 0
        return count, ",".join(_first_visible_devices(count)) if count else None

    if config_gpus is not None:
        gpu_ids = _parse_gpu_ids(config_gpus)
        if gpu_ids is None:
            return cuda_device_count(), None
        return len(gpu_ids), ",".join(gpu_ids)

    if config_num_gpus is not None:
        count = _parse_num_gpus(config_num_gpus) or 0
        return count, ",".join(_first_visible_devices(count)) if count else None

    return cuda_device_count(), None


def resolve_distributed_config(
    *,
    backend: str,
    config: dict[str, Any],
    cli_args: Any,
) -> DistributedLaunchConfig:
    section = _merged_distributed_section(config, backend)
    enabled_value = _as_bool_or_auto(section.get("enabled", "auto"))
    if bool(getattr(cli_args, "no_distributed", False)):
        enabled_value = False

    requested_processes, cuda_visible_devices = _resolve_gpu_selection(section, cli_args)
    if enabled_value is False:
        return DistributedLaunchConfig(
            enabled=False,
            nproc_per_node=1,
            cuda_visible_devices=cuda_visible_devices,
            standalone=bool(section.get("standalone", True)),
        )

    if enabled_value is True and requested_processes <= 0:
        raise DistributedConfigError("Distributed training was enabled, but no CUDA GPUs were detected or selected.")

    nproc_per_node = max(1, requested_processes)
    return DistributedLaunchConfig(
        enabled=True,
        nproc_per_node=nproc_per_node,
        cuda_visible_devices=cuda_visible_devices,
        standalone=bool(section.get("standalone", True)),
    )


def is_torchrun_child() -> bool:
    return "LOCAL_RANK" in os.environ or "RANK" in os.environ or "WORLD_SIZE" in os.environ


def process_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def is_main_process() -> bool:
    return process_rank() == 0


def barrier_if_distributed() -> None:
    if not is_torchrun_child():
        return
    try:
        import torch.distributed as dist
    except Exception:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _timestamp_output_dir_enabled(config: dict[str, Any], backend: str) -> bool:
    backend_section_key = {
        "flagembedding": "flagembedding",
        "sentence-transformers": "sentence_transformers",
        "pylate": "pylate",
    }.get(backend)
    for source in (
        _dict_section(config, backend_section_key) if backend_section_key else {},
        _dict_section(config, "backend_config"),
        config,
    ):
        if "timestamp_output_dir" in source:
            value = source["timestamp_output_dir"]
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def maybe_launch_distributed_training(
    *,
    backend: str,
    config: dict[str, Any],
    cli_args: Any,
    argv: Sequence[str],
) -> int | None:
    if backend not in AUTO_DISTRIBUTED_BACKENDS or is_torchrun_child():
        return None

    launch_config = resolve_distributed_config(backend=backend, config=config, cli_args=cli_args)
    if not launch_config.enabled or launch_config.nproc_per_node <= 1:
        return None

    cmd = [launch_config.launcher]
    if launch_config.standalone:
        cmd.append("--standalone")
    cmd.extend(["--nproc_per_node", str(launch_config.nproc_per_node), "-m", "training.train", *argv])

    env = os.environ.copy()
    if launch_config.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = launch_config.cuda_visible_devices
    if _timestamp_output_dir_enabled(config, backend):
        env.setdefault(RUN_TIMESTAMP_ENV, _run_timestamp())

    visible_text = env.get("CUDA_VISIBLE_DEVICES", "all visible")
    print(
        "[INFO] Relaunching distributed training: "
        f"backend={backend} nproc_per_node={launch_config.nproc_per_node} CUDA_VISIBLE_DEVICES={visible_text}",
        flush=True,
    )
    completed = subprocess.run(cmd, check=False, env=env)
    return int(completed.returncode)


def wait_for_files(paths: Sequence[str | os.PathLike[str]], *, timeout_seconds: float = 3600.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if all(os.path.exists(path) for path in paths):
            return
        if time.monotonic() >= deadline:
            joined = ", ".join(str(path) for path in paths)
            raise DistributedConfigError(f"Timed out waiting for distributed rank 0 to create: {joined}")
        time.sleep(1.0)


def argv_from_args(args: Any) -> list[str]:
    raw_argv = getattr(args, "_raw_argv", None)
    if raw_argv is not None:
        return list(raw_argv)

    argv: list[str] = ["--config", str(args.config)]
    if getattr(args, "backend", None) is not None:
        argv.extend(["--backend", str(args.backend)])
    if getattr(args, "training_type", None) is not None:
        argv.extend(["--training-type", str(args.training_type)])
    if getattr(args, "benchmark_name", None) is not None:
        argv.extend(["--benchmark-name", str(args.benchmark_name)])
    if getattr(args, "run_mteb", False):
        argv.append("--run-mteb")
    if getattr(args, "run_pirb", False):
        argv.append("--run-pirb")
    if getattr(args, "pirb_scope", None) is not None:
        argv.extend(["--pirb-scope", str(args.pirb_scope)])
    if getattr(args, "remove_checkpoints", False):
        argv.append("--remove-checkpoints")
    if getattr(args, "resume", False):
        argv.append("--resume")
    if getattr(args, "resume_from_checkpoint", None):
        argv.extend(["--resume-from-checkpoint", str(args.resume_from_checkpoint)])
    if getattr(args, "gpus", None) is not None:
        argv.extend(["--gpus", str(args.gpus)])
    if getattr(args, "num_gpus", None) is not None:
        argv.extend(["--num-gpus", str(args.num_gpus)])
    if getattr(args, "no_distributed", False):
        argv.append("--no-distributed")
    return argv
