"""Shared post-training benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK_NAME = "NanoBEIR"
DEFAULT_PIRB_SCOPE = "tiny"
DEFAULT_BENCHMARK_BATCH_SIZE = 64
DEFAULT_PIRB_MAX_SEQ_LENGTH = 512


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
    )


def _safe_label(value: str) -> str:
    allowed = []
    for char in value.strip().replace("\\", "/"):
        if char.isalnum() or char in {"_", "-"}:
            allowed.append(char)
        else:
            allowed.append("-")
    return "".join(allowed).strip("-_") or "benchmark"


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
        )
        metrics.update({f"{metric_prefix}{key}": value for key, value in metrics_pirb.items()})

    if output_root is not None:
        (output_root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if metrics and settings.log_to_wandb:
        _log_to_wandb(metrics, step=step)

    return metrics
