"""Shared post-training benchmark helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Iterable, Sequence

from training.checkpoints import DEFAULT_STEP_CHECKPOINT_DIR, checkpoint_step
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
class PirbTaskGroup:
    """PIRB tasks which must stay together because they share an index cache."""

    cache_name: str
    task_ids: tuple[str, ...]
    size_bytes: int


@dataclass(frozen=True)
class PirbTaskChunk:
    index: int
    groups: tuple[PirbTaskGroup, ...]
    size_bytes: int

    @property
    def scope(self) -> str:
        return ",".join(task_id for group in self.groups for task_id in group.task_ids)


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
    parallel_pirb_tasks: bool = True
    pirb_jobs_per_worker: int = 2

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
    parallel_pirb_tasks = _as_bool(
        _config_value(
            config,
            backend_config,
            "parallel_pirb_tasks",
            "parallel_datasets",
            default=True,
        ),
        default=True,
    )
    pirb_jobs_per_worker = int(
        _config_value(
            config,
            backend_config,
            "pirb_jobs_per_worker",
            "dataset_groups_per_worker",
            default=2,
        )
    )
    if pirb_jobs_per_worker <= 0:
        raise ValueError("'benchmark.pirb_jobs_per_worker' must be greater than zero.")

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
        parallel_pirb_tasks=parallel_pirb_tasks,
        pirb_jobs_per_worker=pirb_jobs_per_worker,
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


def _latest_known_training_step(output_dir: Path, *, step_checkpoint_dir: str) -> int | None:
    checkpoint_dirs = [
        *(path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        *(path for path in (output_dir / "epoch-checkpoints").glob("epoch-*-step-*") if path.is_dir()),
        *(path for path in (output_dir / step_checkpoint_dir).glob("step-*") if path.is_dir()),
    ]
    steps = []
    for path in checkpoint_dirs:
        step = checkpoint_step(path)
        if step < 0 and path.name.startswith("step-"):
            try:
                step = int(path.name.removeprefix("step-"))
            except ValueError:
                pass
        if step >= 0:
            steps.append(step)
    return max(steps) if steps else None


def _warn_missing_target(label: str, expected: Path | str) -> None:
    LOGGER.warning("Selected benchmark checkpoint '%s' was not found at %s; skipping.", label, expected)


def _resolve_benchmark_target(
    output_dir: Path,
    spec: Any,
    *,
    step_checkpoint_dir: str = DEFAULT_STEP_CHECKPOINT_DIR,
    final_step: int | None = None,
) -> BenchmarkTarget | None:
    if isinstance(spec, str):
        normalized = spec.strip().lower()
        if normalized == "final":
            path = output_dir / "final"
            if not path.is_dir():
                _warn_missing_target("final", path)
                return None
            return BenchmarkTarget(label="final", path=path, step=final_step)
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


def resolve_benchmark_targets(
    output_dir: Path,
    settings: BenchmarkSettings,
    *,
    final_step: int | None = None,
) -> list[BenchmarkTarget]:
    if final_step is None:
        final_step = _latest_known_training_step(
            output_dir,
            step_checkpoint_dir=settings.step_checkpoint_dir,
        )
    specs = settings.checkpoints if settings.checkpoints is not None else ("final",)
    targets: list[BenchmarkTarget] = []
    for spec in specs:
        target = _resolve_benchmark_target(
            output_dir,
            spec,
            step_checkpoint_dir=settings.step_checkpoint_dir,
            final_step=final_step,
        )
        if target is not None:
            targets.append(target)
    return sorted(
        targets,
        key=lambda target: (
            target.step is None,
            target.step if target.step is not None else 0,
            target.label == "final",
        ),
    )


def _load_mteb_tasks(benchmark_name: str):
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MTEB benchmarking requires mteb. Install it with the matching backend requirements file."
        ) from exc
    return mteb.get_benchmarks(names=[benchmark_name])


def _log_to_wandb(metrics: dict[str, Any], *, step: int | None, label: str) -> None:
    try:
        import wandb
    except ModuleNotFoundError:
        return
    if getattr(wandb, "run", None) is None:
        return
    payload = {
        **metrics,
        "benchmark/checkpoint_label": label,
    }
    if step is not None:
        payload["benchmark/checkpoint_step"] = step
    # Benchmark checkpoints are evaluated after training, when W&B's internal
    # step has already advanced past their training steps. Passing ``step=``
    # here would make W&B reject historical checkpoint results as out of order.
    wandb.log(payload)


def run_benchmarks_for_model(
    model_dir: str,
    settings: BenchmarkSettings,
    *,
    metric_prefix: str = "",
    step: int | None = None,
    label: str = "final",
    pirb_cuda_visible_device: str | None = None,
) -> dict[str, Any]:
    if not settings.enabled:
        return {}

    from convert_utils import run_mteb, run_pirb

    output_root = settings.output_dir / _safe_label(label) if settings.output_dir is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {}
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
            benchmark_label=metric_prefix.removesuffix("/") or label,
        )
        metrics.update({f"{metric_prefix}{key}": value for key, value in metrics_pirb.items()})

    if output_root is not None:
        (output_root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if metrics and settings.log_to_wandb:
        _log_to_wandb(metrics, step=step, label=label)

    return metrics


BenchmarkRunner = Callable[..., dict[str, Any]]
PirbTaskLoader = Callable[[str], Sequence[PirbTaskGroup | dict[str, Any]]]


def _parallel_pirb_devices(settings: BenchmarkSettings) -> list[str]:
    if not settings.run_pirb or settings.run_mteb:
        return []

    devices = visible_cuda_devices()
    worker_limit = settings.parallel_checkpoint_workers
    if worker_limit is not None:
        devices = devices[:worker_limit]
    return devices if len(devices) > 1 else []


def _coerce_pirb_task_groups(
    values: Sequence[PirbTaskGroup | dict[str, Any]],
) -> list[PirbTaskGroup]:
    groups: list[PirbTaskGroup] = []
    seen_cache_names: set[str] = set()
    seen_task_ids: set[str] = set()
    for value in values:
        if isinstance(value, PirbTaskGroup):
            group = value
        else:
            raw_task_ids = value.get("task_ids", ())
            if isinstance(raw_task_ids, str) or not isinstance(raw_task_ids, Sequence):
                raise ValueError("PIRB task group 'task_ids' must be a sequence of task names.")
            group = PirbTaskGroup(
                cache_name=str(value.get("cache_name", "")),
                task_ids=tuple(str(task_id) for task_id in raw_task_ids),
                size_bytes=max(0, int(value.get("size_bytes", 0))),
            )
        if not group.cache_name or not group.task_ids:
            raise ValueError("PIRB task groups must contain a cache name and at least one task.")
        if group.cache_name in seen_cache_names:
            raise ValueError(f"PIRB task manifest contains duplicate cache group '{group.cache_name}'.")
        if len(set(group.task_ids)) != len(group.task_ids):
            raise ValueError(f"PIRB cache group '{group.cache_name}' contains a duplicate task.")
        duplicate_task_ids = seen_task_ids.intersection(group.task_ids)
        if duplicate_task_ids:
            duplicates = ", ".join(sorted(duplicate_task_ids))
            raise ValueError(f"PIRB task manifest contains duplicate tasks: {duplicates}.")
        seen_cache_names.add(group.cache_name)
        seen_task_ids.update(group.task_ids)
        groups.append(group)
    return groups


def _partition_pirb_task_groups(
    groups: Sequence[PirbTaskGroup],
    chunk_count: int,
) -> list[PirbTaskChunk]:
    """Balance indivisible cache groups across a fixed number of chunks."""
    if not groups:
        return []
    chunk_count = max(1, min(chunk_count, len(groups)))
    bins: list[list[tuple[int, PirbTaskGroup]]] = [[] for _ in range(chunk_count)]
    bin_sizes = [0] * chunk_count
    weighted_groups = sorted(
        enumerate(groups),
        key=lambda item: (-max(1, item[1].size_bytes), item[0]),
    )
    for original_index, group in weighted_groups:
        bin_index = min(
            range(chunk_count),
            key=lambda index: (bin_sizes[index], len(bins[index]), index),
        )
        bins[bin_index].append((original_index, group))
        bin_sizes[bin_index] += max(1, group.size_bytes)

    chunks: list[PirbTaskChunk] = []
    for index, entries in enumerate(bins):
        ordered_groups = tuple(group for _, group in sorted(entries, key=lambda item: item[0]))
        chunks.append(
            PirbTaskChunk(
                index=index,
                groups=ordered_groups,
                size_bytes=sum(max(1, group.size_bytes) for group in ordered_groups),
            )
        )
    return chunks


def _pirb_chunk_count(
    settings: BenchmarkSettings,
    *,
    group_count: int,
    worker_count: int,
    simultaneous_target_count: int,
) -> int:
    desired_jobs = max(
        simultaneous_target_count,
        worker_count * settings.pirb_jobs_per_worker,
    )
    return min(group_count, max(1, math.ceil(desired_jobs / simultaneous_target_count)))


def _prepare_and_load_pirb_task_groups(scope: str) -> Sequence[dict[str, Any]]:
    from convert_utils import prepare_pirb_data

    return prepare_pirb_data(scope)


def _write_target_metrics(
    settings: BenchmarkSettings,
    target: BenchmarkTarget,
    metrics: dict[str, Any],
) -> None:
    if settings.output_dir is None:
        return
    output_root = settings.output_dir / _safe_label(target.label)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _merge_pirb_part_metrics(
    part_metrics: Iterable[dict[str, Any]],
    *,
    metric_prefix: str,
) -> dict[str, Any]:
    parts = list(part_metrics)
    if not parts:
        return {}

    datasets_key = f"{metric_prefix}pirb_datasets"
    ndcg_tasks_key = f"{metric_prefix}pirb_ndcg_tasks"
    average_prefix = f"{metric_prefix}pirb_average_ndcg@"
    merged: dict[str, Any] = {}
    average_keys: set[str] = set()
    total_datasets = 0
    total_ndcg = 0.0
    datasets_seen = 0
    ndcg_seen = 0

    for metrics in parts:
        for key, value in metrics.items():
            if key == datasets_key:
                total_datasets += int(value)
                datasets_seen += 1
            elif key == ndcg_tasks_key:
                total_ndcg += float(value)
                ndcg_seen += 1
            elif key.startswith(average_prefix):
                average_keys.add(key)
            elif key in merged and merged[key] != value:
                raise ValueError(f"Conflicting PIRB metric '{key}' while merging task groups.")
            else:
                merged[key] = value

    if len(parts) > 1 and (datasets_seen != len(parts) or ndcg_seen != len(parts)):
        raise ValueError("Each PIRB task-group result must contain 'datasets' and 'ndcg_tasks' totals.")
    if datasets_seen:
        merged[datasets_key] = total_datasets
    if ndcg_seen:
        merged[ndcg_tasks_key] = total_ndcg
    if datasets_seen and ndcg_seen:
        for key in sorted(average_keys):
            merged[key] = total_ndcg / total_datasets if total_datasets else 0.0
    elif len(parts) == 1:
        for key in average_keys:
            merged[key] = parts[0][key]
    return merged


def _run_parallel_checkpoint_targets(
    targets: Sequence[BenchmarkTarget],
    settings: BenchmarkSettings,
    devices: Sequence[str],
    runner: BenchmarkRunner,
) -> dict[str, dict[str, Any]]:
    selected_devices = list(devices[: len(targets)])
    print(
        "[INFO] Running checkpoint benchmarks in parallel: "
        f"targets={len(targets)} workers={len(selected_devices)} GPUs={','.join(selected_devices)}.",
        flush=True,
    )
    available_devices: Queue[str] = Queue()
    for device in selected_devices:
        available_devices.put(device)
    worker_settings = replace(settings, log_to_wandb=False)

    def run_target(target: BenchmarkTarget) -> dict[str, Any]:
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

    with ThreadPoolExecutor(max_workers=len(selected_devices), thread_name_prefix="pirb-checkpoint") as executor:
        futures = [executor.submit(run_target, target) for target in targets]
        ordered_metrics = [future.result() for future in futures]

    results = {target.label: metrics for target, metrics in zip(targets, ordered_metrics)}
    if settings.log_to_wandb:
        for target, metrics in zip(targets, ordered_metrics):
            if metrics:
                _log_to_wandb(metrics, step=target.step, label=target.label)
    return results


def _run_parallel_pirb_chunks(
    targets: Sequence[BenchmarkTarget],
    chunks: Sequence[PirbTaskChunk],
    settings: BenchmarkSettings,
    devices: Sequence[str],
    runner: BenchmarkRunner,
) -> dict[str, dict[str, Any]]:
    jobs = [(target, chunk) for target in targets for chunk in chunks]
    jobs.sort(key=lambda item: -item[1].size_bytes)
    selected_devices = list(devices[: len(jobs)])
    print(
        "[INFO] Running PIRB task groups dynamically: "
        f"targets={len(targets)} chunks_per_target={len(chunks)} jobs={len(jobs)} "
        f"workers={len(selected_devices)} GPUs={','.join(selected_devices)}.",
        flush=True,
    )
    available_devices: Queue[str] = Queue()
    for device in selected_devices:
        available_devices.put(device)
    worker_settings = replace(settings, log_to_wandb=False)

    def run_job(target: BenchmarkTarget, chunk: PirbTaskChunk) -> dict[str, Any]:
        device = available_devices.get()
        try:
            part_settings = replace(worker_settings, pirb_scope=chunk.scope)
            if settings.output_dir is not None:
                part_settings = replace(
                    part_settings,
                    output_dir=settings.output_dir / _safe_label(target.label) / "pirb-parts",
                )
            print(
                f"[INFO] Running {target.label} PIRB part {chunk.index:03d} "
                f"({sum(len(group.task_ids) for group in chunk.groups)} tasks) on GPU {device}.",
                flush=True,
            )
            return runner(
                str(target.path.resolve()),
                part_settings,
                metric_prefix=f"{target.label}/",
                step=target.step,
                label=f"part-{chunk.index:03d}",
                pirb_cuda_visible_device=device,
            )
        finally:
            available_devices.put(device)

    with ThreadPoolExecutor(max_workers=len(selected_devices), thread_name_prefix="pirb-task-group") as executor:
        submitted = [
            (target, chunk, executor.submit(run_job, target, chunk))
            for target, chunk in jobs
        ]
        metrics_by_target: dict[str, dict[int, dict[str, Any]]] = {
            target.label: {} for target in targets
        }
        for target, chunk, future in submitted:
            metrics_by_target[target.label][chunk.index] = future.result()

    results: dict[str, dict[str, Any]] = {}
    for target in targets:
        ordered_parts = [metrics_by_target[target.label][chunk.index] for chunk in chunks]
        metrics = _merge_pirb_part_metrics(
            ordered_parts,
            metric_prefix=f"{target.label}/",
        )
        datasets_key = f"{target.label}/pirb_datasets"
        average_prefix = f"{target.label}/pirb_average_ndcg@"
        for key, value in sorted(metrics.items()):
            if key.startswith(average_prefix):
                ndcg_k = key.removeprefix(average_prefix)
                print(
                    f"[checkpoint: {target.label}] Average NDCG@{ndcg_k} "
                    f"for {metrics[datasets_key]} tasks: {float(value):.2f}",
                    flush=True,
                )
        results[target.label] = metrics
        _write_target_metrics(settings, target, metrics)
        if metrics and settings.log_to_wandb:
            _log_to_wandb(metrics, step=target.step, label=target.label)
    return results


def run_benchmarks_for_targets(
    targets: Sequence[BenchmarkTarget],
    settings: BenchmarkSettings,
    *,
    runner: BenchmarkRunner = run_benchmarks_for_model,
    prepare_pirb: Callable[[], None] | None = None,
    load_pirb_task_groups: PirbTaskLoader | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate targets with dynamically scheduled checkpoint/task PIRB jobs."""
    selected_targets = list(targets)
    if not selected_targets:
        return {}

    devices = _parallel_pirb_devices(settings)
    checkpoint_parallel = settings.parallel_checkpoints and len(selected_targets) > 1
    task_parallel_requested = settings.parallel_pirb_tasks
    if not devices or not (checkpoint_parallel or task_parallel_requested):
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

    task_groups: list[PirbTaskGroup] = []
    pirb_prepared = False
    if task_parallel_requested:
        loader = load_pirb_task_groups or _prepare_and_load_pirb_task_groups
        task_groups = _coerce_pirb_task_groups(loader(settings.pirb_scope))
        pirb_prepared = True

    simultaneous_target_count = len(selected_targets) if checkpoint_parallel else 1
    chunk_count = _pirb_chunk_count(
        settings,
        group_count=len(task_groups),
        worker_count=len(devices),
        simultaneous_target_count=simultaneous_target_count,
    ) if task_groups else 0
    if chunk_count > 1:
        chunks = _partition_pirb_task_groups(task_groups, chunk_count)
        if checkpoint_parallel:
            return _run_parallel_pirb_chunks(selected_targets, chunks, settings, devices, runner)

        results: dict[str, dict[str, Any]] = {}
        for target in selected_targets:
            results.update(_run_parallel_pirb_chunks([target], chunks, settings, devices, runner))
        return results

    if checkpoint_parallel:
        if not pirb_prepared:
            if prepare_pirb is None:
                from convert_utils import prepare_pirb_data

                prepare_pirb = prepare_pirb_data
            prepare_pirb()
        return _run_parallel_checkpoint_targets(selected_targets, settings, devices, runner)

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
