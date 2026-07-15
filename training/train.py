"""Main CLI for backend-selectable training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

from training.backends.registry import (
    BackendDependencyError,
    TrainingCliError,
    TrainingRequest,
    all_training_types,
    backend_names,
    load_backend_module,
    normalize_backend,
    normalize_training_type,
    validate_backend_training_type,
)
from training.checkpoints import AUTO_RESUME, LATEST_CHECKPOINT, output_dir_from_checkpoint, resolve_resume_checkpoint, resume_spec_from_sources
from training.config_grid import BACKEND_SECTION_KEYS, expand_config_grid
from training.distributed import argv_from_args, is_main_process, maybe_launch_distributed_training
from training.run_metadata import create_run_metadata, verify_resume_metadata


class ConfigError(TrainingCliError):
    """Raised when the training configuration cannot be loaded."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backend-selectable training launcher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported combinations:\n"
            "  flagembedding          embedder\n"
            "  sentence-transformers  embedder, matryoshka, splade, multimodal, adaptive-layer, matryoshka-2d\n"
            "  pylate                 colbert, late-interaction\n\n"
            "Examples:\n"
            "  python -m training.train --backend flagembedding --training-type embedder --config configs/grid.yaml\n"
            "  python -m training.train --backend sentence-transformers --training-type matryoshka --config configs/smoke_sentence_transformers_matryoshka.yaml\n"
            "  python -m training.train --backend sentence-transformers --training-type splade --config configs/smoke_sentence_transformers_splade.yaml\n"
            "  python -m training.train --backend pylate --training-type colbert --config configs/smoke_pylate_colbert.yaml"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=backend_names(),
        default=None,
        help="Training backend to run. CLI value overrides the config file.",
    )
    parser.add_argument(
        "--training-type",
        choices=all_training_types(),
        default=None,
        help="Training recipe for the selected backend. CLI value overrides the config file.",
    )
    parser.add_argument(
        "--config",
        default="configs/grid.yaml",
        help="Path to YAML training configuration. Missing files fall back to built-in FlagEmbedding defaults.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Override or add a YAML config value after loading --config. "
            "Use dot paths for nested values and numeric path segments for list indexes, "
            "for example --set train_data=dataset-a "
            "--set sentence_transformers.train_batch_size=4 "
            "--set hparams.0.learning_rate=2e-5. Values are parsed as YAML."
        ),
    )
    parser.add_argument(
        "--set-str",
        dest="config_string_overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Override or add a YAML config value as a literal string. "
            "Use this for values that YAML would coerce, such as --set-str sentence_transformers.save_strategy=no."
        ),
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Resolve config overrides and grids, print the run plan, and exit without training.",
    )
    parser.add_argument(
        "--print-config",
        dest="print_config",
        action="store_true",
        help="Print fully resolved expanded YAML config(s) and exit without training.",
    )
    parser.add_argument(
        "--no-save-resolved-config",
        dest="save_resolved_config",
        action="store_false",
        default=True,
        help="Do not write resolved_config.yaml and command.txt into each resolved output_dir.",
    )
    parser.add_argument(
        "--benchmark-name",
        default=None,
        help="Name of the MTEB benchmark to use when --run-mteb is enabled.",
    )
    parser.add_argument(
        "--benchmark-output-dir",
        dest="benchmark_output_dir",
        default=None,
        help="Directory for benchmark configs, raw outputs, and metrics JSON files.",
    )
    parser.add_argument(
        "--benchmark-batch-size",
        dest="benchmark_batch_size",
        type=int,
        default=None,
        help="Batch size used by MTEB encoding.",
    )
    parser.add_argument(
        "--benchmark-query-instruction",
        dest="benchmark_query_instruction",
        default=None,
        help="Query instruction used by post-training benchmarks.",
    )
    parser.add_argument(
        "--run-mteb",
        "--run_mteb",
        dest="run_mteb",
        action="store_true",
        help="Run MTEB evaluation after FlagEmbedding training. This may require sentence-transformers for conversion.",
    )
    parser.add_argument(
        "--run-pirb",
        "--run_pirb",
        dest="run_pirb",
        action="store_true",
        help="Run PIRB evaluation after FlagEmbedding training. This may require sentence-transformers for conversion.",
    )
    parser.add_argument(
        "--pirb-scope",
        "--pirb_scope",
        dest="pirb_scope",
        choices=("tiny", "small", "all"),
        default=None,
        help="PIRB benchmark scope used with --run-pirb.",
    )
    parser.add_argument(
        "--pirb-max-seq-length",
        "--pirb_max_seq_length",
        dest="pirb_max_seq_length",
        type=int,
        default=None,
        help="Maximum sequence length used by PIRB evaluation.",
    )
    parser.add_argument(
        "--remove-checkpoints",
        "--remove_checkpoints",
        dest="remove_checkpoints",
        action="store_true",
        help="Remove checkpoint-* directories after a successful FlagEmbedding run.",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated GPU ids to expose to the run, for example '0,1' or '2'. "
            "Use 'auto' to use all visible GPUs."
        ),
    )
    gpu_group.add_argument(
        "--num-gpus",
        dest="num_gpus",
        type=int,
        default=None,
        help="Use the first N visible GPUs. Prefer --gpus when selecting specific GPU ids.",
    )
    parser.add_argument(
        "--no-distributed",
        dest="no_distributed",
        action="store_true",
        help="Disable automatic torchrun launch and force single-process training.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume each run from its latest checkpoint under the resolved output directory.",
    )
    resume_group.add_argument(
        "--resume-if-available",
        dest="resume_if_available",
        action="store_true",
        help="Resume each run from its latest checkpoint when present; otherwise start a new run.",
    )
    resume_group.add_argument(
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        default=None,
        help="Resume from a specific checkpoint directory. Use --resume for expanded grids.",
    )
    return parser


def _load_yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ConfigError("YAML config support requires PyYAML. Install it before using --config.") from exc
    return yaml


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        print(f"[WARN] Config file {config_path} not found; using built-in defaults.", file=sys.stderr)
        return {}

    yaml = _load_yaml_module()
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise ConfigError(f"Could not read config file '{config_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config file '{config_path}' must contain a YAML mapping at the top level.")
    return data


def _split_override(raw_override: str, *, option_name: str) -> tuple[str, str]:
    if "=" not in raw_override:
        raise ConfigError(f"{option_name} expects PATH=VALUE, got '{raw_override}'.")
    path, value = raw_override.split("=", 1)
    path = path.strip()
    if not path:
        raise ConfigError(f"{option_name} override path cannot be empty.")
    return path, value


def _parse_override_path(path: str) -> list[str | int]:
    parts: list[str | int] = []
    for raw_part in path.split("."):
        part = raw_part.strip()
        if not part:
            raise ConfigError(f"Config override path '{path}' contains an empty segment.")
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    return parts


def _new_override_container(next_part: str | int) -> dict[str, Any] | list[Any]:
    return [] if isinstance(next_part, int) else {}


def _ensure_list_index(values: list[Any], index: int, path: str) -> None:
    if index < 0:
        raise ConfigError(f"Config override path '{path}' contains a negative list index.")
    while len(values) <= index:
        values.append(None)


def _apply_config_override(config: dict[str, Any], path: str, value: Any) -> None:
    parts = _parse_override_path(path)
    current: Any = config

    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        if isinstance(part, int):
            if not isinstance(current, list):
                raise ConfigError(f"Config override path '{path}' expected a list before index {part}.")
            _ensure_list_index(current, part, path)
            if current[part] is None:
                current[part] = _new_override_container(next_part)
            current = current[part]
        else:
            if not isinstance(current, dict):
                raise ConfigError(f"Config override path '{path}' expected a mapping before '{part}'.")
            if part not in current or current[part] is None:
                current[part] = _new_override_container(next_part)
            current = current[part]

        expected_type = list if isinstance(next_part, int) else dict
        if not isinstance(current, expected_type):
            expected_name = "list" if expected_type is list else "mapping"
            raise ConfigError(f"Config override path '{path}' expected a {expected_name} before '{next_part}'.")

    leaf = parts[-1]
    if isinstance(leaf, int):
        if not isinstance(current, list):
            raise ConfigError(f"Config override path '{path}' expected a list before index {leaf}.")
        _ensure_list_index(current, leaf, path)
        current[leaf] = value
    else:
        if not isinstance(current, dict):
            raise ConfigError(f"Config override path '{path}' expected a mapping before '{leaf}'.")
        current[leaf] = value


def _parse_yaml_override_value(raw_value: str, *, path: str) -> Any:
    yaml = _load_yaml_module()
    try:
        return yaml.safe_load(raw_value)
    except Exception as exc:
        raise ConfigError(f"Could not parse YAML value for override '{path}': {exc}") from exc


def apply_config_overrides(
    config: dict[str, Any],
    yaml_overrides: Sequence[str] | None = None,
    string_overrides: Sequence[str] | None = None,
) -> None:
    """Apply command-line config overrides in-place."""

    for raw_override in yaml_overrides or []:
        path, raw_value = _split_override(raw_override, option_name="--set")
        _apply_config_override(config, path, _parse_yaml_override_value(raw_value, path=path))

    for raw_override in string_overrides or []:
        path, value = _split_override(raw_override, option_name="--set-str")
        _apply_config_override(config, path, value)


def resolve_backend_and_training_type(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[str, str]:
    backend = args.backend or config.get("backend") or "flagembedding"
    training_type = args.training_type or config.get("training_type") or config.get("training-type") or "embedder"
    return normalize_backend(str(backend)), normalize_training_type(str(training_type))


def _dict_section(config: dict[str, Any], key: str | None) -> dict[str, Any]:
    if key is None:
        return {}
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _resume_backend_config(config: dict[str, Any], backend: str) -> dict[str, Any]:
    backend_section_key = BACKEND_SECTION_KEYS.get(backend)
    common = {
        key: config[key]
        for key in ("resume", "resume_from_checkpoint", "epoch_checkpoint_dir")
        if key in config
    }
    return common | _dict_section(config, "backend_config") | _dict_section(config, backend_section_key)


def _set_backend_values(config: dict[str, Any], backend: str, values: dict[str, Any]) -> None:
    config.update(values)
    backend_section_key = BACKEND_SECTION_KEYS.get(backend)
    if backend_section_key is None:
        return
    section = config.get(backend_section_key)
    if section is None:
        section = {}
        config[backend_section_key] = section
    if isinstance(section, dict):
        section.update(values)


def _apply_resume_to_configs(configs: list[dict[str, Any]], backend: str, resume_spec: str | None) -> None:
    if resume_spec is None:
        return

    is_explicit_checkpoint = resume_spec not in {LATEST_CHECKPOINT, AUTO_RESUME}
    if is_explicit_checkpoint and len(configs) > 1:
        raise ConfigError("--resume-from-checkpoint can only be used with a single expanded run. Use --resume for grids.")

    for run_config in configs:
        values = {"resume": True, "resume_from_checkpoint": resume_spec}
        if is_explicit_checkpoint:
            epoch_dir = str(_resume_backend_config(run_config, backend).get("epoch_checkpoint_dir") or "epoch-checkpoints")
            output_dir = output_dir_from_checkpoint(Path(resume_spec), epoch_checkpoint_dir=epoch_dir)
            values["output_dir"] = str(output_dir)
        _set_backend_values(run_config, backend, values)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _should_save_resolved_config(config: dict[str, Any], args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "save_resolved_config", True)):
        return False
    if "save_resolved_config" in config:
        return _as_bool(config["save_resolved_config"])
    return True


def _resolved_output_dir(config: dict[str, Any], backend: str, training_type: str) -> Path | None:
    backend_section_key = BACKEND_SECTION_KEYS.get(backend)
    for source in (
        _dict_section(config, backend_section_key) if backend_section_key else {},
        _dict_section(config, "backend_config"),
        config,
    ):
        if source.get("output_dir") is not None:
            return Path(str(source["output_dir"]))
    return None


def _prepare_run_metadata(
    run_config: dict[str, Any],
    args: argparse.Namespace,
    *,
    backend: str,
    training_type: str,
) -> None:
    if not is_main_process():
        return
    output_dir = _resolved_output_dir(run_config, backend, training_type)
    if output_dir is None:
        return

    backend_config = _resume_backend_config(run_config, backend)
    checkpoint = resolve_resume_checkpoint(output_dir, run_config, backend_config, args)
    if checkpoint is not None:
        verify_resume_metadata(output_dir, run_config)
        return

    if _should_save_resolved_config(run_config, args):
        create_run_metadata(output_dir, run_config, args)


def _print_run_plan(configs: list[dict[str, Any]], *, backend: str, training_type: str) -> None:
    print(f"[DRY-RUN] {len(configs)} run(s) would execute.")
    for index, run_config in enumerate(configs, start=1):
        output_dir = _resolved_output_dir(run_config, backend, training_type)
        run_name = run_config.get("run_name", "<none>")
        train_data = run_config.get("train_data", "<unset>")
        print(
            f"[DRY-RUN] {index}/{len(configs)} "
            f"backend={backend} training_type={training_type} "
            f"run_name={run_name} output_dir={output_dir or '<unresolved>'} train_data={train_data}",
        )


def _print_resolved_configs(configs: list[dict[str, Any]]) -> None:
    yaml = _load_yaml_module()
    print(yaml.safe_dump_all(configs, sort_keys=False, allow_unicode=True), end="")


def run_training(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    apply_config_overrides(
        config,
        getattr(args, "config_overrides", None),
        getattr(args, "config_string_overrides", None),
    )
    backend, training_type = resolve_backend_and_training_type(args, config)
    spec = validate_backend_training_type(backend, training_type)
    dry_run = bool(getattr(args, "dry_run", False) or getattr(args, "print_config", False))
    launch_result = maybe_launch_distributed_training(
        backend=backend,
        config=config,
        cli_args=args,
        argv=argv_from_args(args),
    ) if not dry_run else None
    if launch_result is not None:
        return launch_result

    resume_spec = resume_spec_from_sources(config, _resume_backend_config(config, backend), args)
    try:
        configs = expand_config_grid(
            config,
            backend=backend,
            training_type=training_type,
            resume=resume_spec is not None,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    _apply_resume_to_configs(configs, backend, resume_spec)

    if getattr(args, "dry_run", False):
        _print_run_plan(configs, backend=backend, training_type=training_type)
    if getattr(args, "print_config", False):
        _print_resolved_configs(configs)
    if dry_run:
        return 0

    backend_module = load_backend_module(spec)
    for run_config in configs:
        _prepare_run_metadata(run_config, args, backend=backend, training_type=training_type)
        request = TrainingRequest(
            backend=backend,
            training_type=training_type,
            config=run_config,
            config_path=args.config,
            cli_args=args,
        )
        result = backend_module.run_training(request)
        if result is not None and int(result) != 0:
            return int(result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args._raw_argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        return run_training(args)
    except BackendDependencyError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except TrainingCliError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
