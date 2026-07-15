"""Shared post-training benchmark helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Sequence

from training.checkpoints import DEFAULT_STEP_CHECKPOINT_DIR
from training.distributed import visible_cuda_devices


DEFAULT_BENCHMARK_NAME = "NanoBEIR"
DEFAULT_PIRB_SCOPE = "tiny"
DEFAULT_BENCHMARK_BATCH_SIZE = 64
DEFAULT_PIRB_MAX_SEQ_LENGTH = 512
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkTarget:
    label: str
    path: Path
    step: int | None = None


@dataclass(frozen=True)
class BenchmarkSettings:
    run_mteb: bool
    run_pirb: bool
    benchmark_name: str
    pirb_scope: str
    batch_size: int
    pirb_max_seq_length: int
    query_instruction_for_retrieval: str
    output_dir: Path | None = None
    log_to_wandb: bool = True
    checkpoints: tuple[Any, ...] | None = None
    step_checkpoint_dir: str = DEFAULT_STEP_CHECKPOINT_DIR
    parallel_checkpoints: bool = True
    parallel_checkpoint_workers: int | None = None

    @property
    def enabled(self) -> bool:
        return self.run_mteb or self.run_pirb


def _dict_section(config: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _first_value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _benchmark_sections(config: dict[str, Any], backend_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return _dict_section(config, "benchmark"), _dict_section(backend_config, "benchmark")


def _config_value(config: dict[str, Any], backend_config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    benchmark_config, backend_benchmark_config = _benchmark_sections(config, backend_config)
    for key in keys:
        for source in (benchmark_config, backend_benchmark_config, backend_config, config):
            if key in source:
                return source[key]
    return default


def resolve_benchmark_settings(
    config: dict[str, Any],
    backend_config: dict[str, Any],
    cli_args: Any,
    *,
    default_query_instruction: str = "",
) -> BenchmarkSettings:
    run_mteb = bool(getattr(cli_args, "run_mteb", False)) or _as_bool(
        _config_value(config, backend_config, "run_mteb", "mteb", default=False)
    )
    run_pirb = bool(getattr(cli_args, "run_pirb", False)) or _as_bool(
        _config_value(config, backend_config, "run_pirb", "pirb", default=False)
    )
    benchmark_name = str(
        _first_value(
            getattr(cli_args, "benchmark_name", None),
            _config_value(config, backend_config, "benchmark_name", "name"),
            default=DEFAULT_BENCHMARK_NAME,
        )
    )
    pirb_scope = str(
        _first_value(
            getattr(cli_args, "pirb_scope", None),
            _config_value(config, backend_config, "pirb_scope", "scope"),
            default=DEFAULT_PIRB_SCOPE,
        )
    )
    batch_size = int(
        _first_value(
            getattr(cli_args, "benchmark_batch_size", None),
            _config_value(config, backend_config, "benchmark_batch_size", "batch_size", "mteb_batch_size"),
            default=DEFAULT_BENCHMARK_BATCH_SIZE,
        )
    )
    pirb_max_seq_length = int(
        _first_value(
            getattr(cli_args, "pirb_max_seq_length", None),
            _config_value(config, backend_config, "pirb_max_seq_length", "max_seq_length"),
            default=DEFAULT_PIRB_MAX_SEQ_LENGTH,
        )
    )
    query_instruction = str(
        _first_value(
            getattr(cli_args, "benchmark_query_instruction", None),
            _config_value(
                config,
                backend_config,
                "benchmark_query_instruction",
                "query_instruction_for_retrieval",
                "query_instruction",
            ),
            default=default_query_instruction,
        )
        or ""
    )
    benchmark_config, backend_benchmark_config = _benchmark_sections(config, backend_config)
    output_dir = _first_value(
        getattr(cli_args, "benchmark_output_dir", None),
        benchmark_config.get("output_dir"),
        backend_benchmark_config.get("output_dir"),
        backend_config.get("benchmark_output_dir"),
        config.get("benchmark_output_dir"),
    )
    log_to_wandb = _as_bool(_config_value(config, backend_config, "log_to_wandb", default=True), default=True)
    checkpoints = _config_value(config, backend_config, "checkpoints")
    if checkpoints is not None:
        if isinstance(checkpoints, (list, tuple)):
            checkpoints = tuple(checkpoints)
        else:
            checkpoints = (checkpoints,)
    step_checkpoint_dir = str(
        backend_config.get("step_checkpoint_dir", config.get("step_checkpoint_dir")) or DEFAULT_STEP_CHECKPOINT_DIR
    )
    parallel_checkpoints = _as_bool(
        _config_value(config, backend_config, "parallel_checkpoints", default=True),
        default=True,
    )
    parallel_checkpoint_workers_value = _config_value(
        config,
        backend_config,
        "parallel_checkpoint_workers",
        "checkpoint_workers",
    )
    parallel_checkpoint_workers = None
    if parallel_checkpoint_workers_value is not None:
        parallel_checkpoint_workers = int(parallel_checkpoint_workers_value)
        if parallel_checkpoint_workers <= 0:
            raise ValueError("'benchmark.parallel_checkpoint_workers' must be greater than zero.")

    return BenchmarkSettings(
        run_mteb=run_mteb,
        run_pirb=run_pirb,
        benchmark_name=benchmark_name,
        pirb_scope=pirb_scope,
        batch_size=batch_size,
        pirb_max_seq_length=pirb_max_seq_length,
        query_instruction_for_retrieval=query_instruction,
        output_dir=Path(str(output_dir)) if output_dir is not None else None,
        log_to_wandb=log_to_wandb,
        checkpoints=checkpoints,
        step_checkpoint_dir=step_checkpoint_dir,
        parallel_checkpoints=parallel_checkpoints,
        parallel_checkpoint_workers=parallel_checkpoint_workers,
    )


def _safe_label(value: str) -> str:
    allowed = []
    for char in value.strip().replace("\\", "/"):
        if char.isalnum() or char in {"_", "-"}:
            allowed.append(char)
        else:
            allowed.append("-")
    return "".join(allowed).strip("-_") or "benchmark"


def _positive_int(value: Any, *, target_type: str) -> int | None:
    if isinstance(value, bool):
        LOGGER.warning("Ignoring benchmark %s target with non-integer value %r.", target_type, value)
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        LOGGER.warning("Ignoring benchmark %s target with non-integer value %r.", target_type, value)
        return None
    if normalized <= 0:
        LOGGER.warning("Ignoring benchmark %s target with non-positive value %r.", target_type, value)
        return None
    return normalized


def _epoch_checkpoint_step(path: Path) -> int | None:
    marker = "-step-"
    if marker not in path.name:
        return None
    try:
        return int(path.name.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _latest_epoch_checkpoint(output_dir: Path, epoch: int) -> Path | None:
    epoch_dir = output_dir / "epoch-checkpoints"
    pattern = f"epoch-{epoch:04d}-step-*"
    candidates = [path for path in epoch_dir.glob(pattern) if path.is_dir()]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (
            _epoch_checkpoint_step(path) is not None,
            _epoch_checkpoint_step(path) or -1,
            path.name,
        ),
    )


def _warn_missing_target(label: str, expected: Path | str) -> None:
    LOGGER.warning("Selected benchmark checkpoint '%s' was not found at %s; skipping.", label, expected)


def _resolve_benchmark_target(
    output_dir: Path,
    spec: Any,
    *,
    step_checkpoint_dir: str = DEFAULT_STEP_CHECKPOINT_DIR,
) -> BenchmarkTarget | None:
    if isinstance(spec, str):
        normalized = spec.strip().lower()
        if normalized == "final":
            path = output_dir / "final"
            if not path.is_dir():
                _warn_missing_target("final", path)
                return None
            return BenchmarkTarget(label="final", path=path, step=0)
        LOGGER.warning("Ignoring unsupported benchmark checkpoint selector %r.", spec)
        return None

    if isinstance(spec, dict):
        selected_keys = [key for key in ("epoch", "step") if key in spec]
        if len(selected_keys) != 1:
            LOGGER.warning("Ignoring unsupported benchmark checkpoint selector %r.", spec)
            return None

        key = selected_keys[0]
        number = _positive_int(spec[key], target_type=key)
        if number is None:
            return None

        if key == "epoch":
            label = f"epoch-{number:04d}"
            path = _latest_epoch_checkpoint(output_dir, number)
            if path is None:
                _warn_missing_target(label, output_dir / "epoch-checkpoints" / f"{label}-step-*")
                return None
            return BenchmarkTarget(label=label, path=path, step=_epoch_checkpoint_step(path))

        label = f"step-{number}"
        preserved_path = output_dir / step_checkpoint_dir / label
        regular_path = output_dir / f"checkpoint-{number}"
        path = preserved_path if preserved_path.is_dir() else regular_path
        if not path.is_dir():
            _warn_missing_target(label, f"{preserved_path} or {regular_path}")
            return None
        return BenchmarkTarget(label=label, path=path, step=number)

    LOGGER.warning("Ignoring unsupported benchmark checkpoint selector %r.", spec)
    return None


def resolve_benchmark_targets(output_dir: Path, settings: BenchmarkSettings) -> list[BenchmarkTarget]:
    specs = settings.checkpoints if settings.checkpoints is not None else ("final",)
    targets: list[BenchmarkTarget] = []
    for spec in specs:
        target = _resolve_benchmark_target(
            output_dir,
            spec,
            step_checkpoint_dir=settings.step_checkpoint_dir,
        )
        if target is not None:
            targets.append(target)
    return targets


def _load_mteb_tasks(benchmark_name: str):
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MTEB benchmarking requires mteb. Install it with the matching backend requirements file."
        ) from exc
    return mteb.get_benchmarks(names=[benchmark_name])


def _log_to_wandb(metrics: dict[str, float], *, step: int | None) -> None:
    try:
        import wandb
    except ModuleNotFoundError:
        return
    if getattr(wandb, "run", None) is None:
        return
    wandb.log(metrics, step=step)


def run_benchmarks_for_model(
    model_dir: str,
    settings: BenchmarkSettings,
    *,
    metric_prefix: str = "",
    step: int | None = None,
    label: str = "final",
    pirb_cuda_visible_device: str | None = None,
) -> dict[str, float]:
    if not settings.enabled:
        return {}

    from convert_utils import run_mteb, run_pirb

    output_root = settings.output_dir / _safe_label(label) if settings.output_dir is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, float] = {}
    if settings.run_mteb:
        mteb_output = output_root / "mteb" if output_root is not None else None
        if mteb_output is not None:
            mteb_output.mkdir(parents=True, exist_ok=True)
        tasks = _load_mteb_tasks(settings.benchmark_name)
        metrics_mteb = run_mteb(
            model_dir,
            tasks,
            batch_size=settings.batch_size,
            output_folder=str(mteb_output) if mteb_output is not None else None,
        )
        metrics.update({f"{metric_prefix}{key}": value for key, value in metrics_mteb.items()})

    if settings.run_pirb:
        pirb_output = output_root / "pirb" if output_root is not None else None
        metrics_pirb = run_pirb(
            model_dir,
            query_instruction_for_retrieval=settings.query_instruction_for_retrieval,
            max_seq_length=settings.pirb_max_seq_length,
            scope=settings.pirb_scope,
            output_dir=str(pirb_output) if pirb_output is not None else None,
            cuda_visible_device=pirb_cuda_visible_device,
        )
        metrics.update({f"{metric_prefix}{key}": value for key, value in metrics_pirb.items()})

    if output_root is not None:
        (output_root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if metrics and settings.log_to_wandb:
        _log_to_wandb(metrics, step=step)

    return metrics


BenchmarkRunner = Callable[..., dict[str, float]]


def _parallel_checkpoint_devices(settings: BenchmarkSettings, target_count: int) -> list[str]:
    if not settings.parallel_checkpoints or target_count <= 1:
        return []
    if not settings.run_pirb or settings.run_mteb:
        return []

    devices = visible_cuda_devices()
    worker_limit = settings.parallel_checkpoint_workers
    if worker_limit is not None:
        devices = devices[:worker_limit]
    return devices[:target_count] if len(devices) > 1 else []


def run_benchmarks_for_targets(
    targets: Sequence[BenchmarkTarget],
    settings: BenchmarkSettings,
    *,
    runner: BenchmarkRunner = run_benchmarks_for_model,
    prepare_pirb: Callable[[], None] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate checkpoint targets, using one independent PIRB process per GPU."""
    selected_targets = list(targets)
    if not selected_targets:
        return {}

    devices = _parallel_checkpoint_devices(settings, len(selected_targets))
    if not devices:
        return {
            target.label: runner(
                str(target.path.resolve()),
                settings,
                metric_prefix=f"{target.label}/",
                step=target.step,
                label=target.label,
            )
            for target in selected_targets
        }

    if prepare_pirb is None:
        from convert_utils import prepare_pirb_data

        prepare_pirb = prepare_pirb_data
    prepare_pirb()

    print(
        "[INFO] Running checkpoint benchmarks in parallel: "
        f"targets={len(selected_targets)} workers={len(devices)} GPUs={','.join(devices)}.",
        flush=True,
    )
    available_devices: Queue[str] = Queue()
    for device in devices:
        available_devices.put(device)

    worker_settings = replace(settings, log_to_wandb=False)

    def run_target(target: BenchmarkTarget) -> dict[str, float]:
        device = available_devices.get()
        try:
            print(f"[INFO] Running benchmark for {target.label} on GPU {device}.", flush=True)
            return runner(
                str(target.path.resolve()),
                worker_settings,
                metric_prefix=f"{target.label}/",
                step=target.step,
                label=target.label,
                pirb_cuda_visible_device=device,
            )
        finally:
            available_devices.put(device)

    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="pirb-checkpoint") as executor:
        futures = [executor.submit(run_target, target) for target in selected_targets]
        ordered_metrics = [future.result() for future in futures]

    results = {target.label: metrics for target, metrics in zip(selected_targets, ordered_metrics)}
    if settings.log_to_wandb:
        for target, metrics in zip(selected_targets, ordered_metrics):
            if metrics:
                _log_to_wandb(metrics, step=target.step)
    return results
