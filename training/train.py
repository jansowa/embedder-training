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
        "--benchmark-name",
        default="NanoBEIR",
        help="Name of the MTEB benchmark to use when --run-mteb is enabled.",
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
        default="tiny",
        help="PIRB benchmark scope used with --run-pirb.",
    )
    parser.add_argument(
        "--remove-checkpoints",
        "--remove_checkpoints",
        dest="remove_checkpoints",
        action="store_true",
        help="Remove checkpoint-* directories after a successful FlagEmbedding run.",
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


def resolve_backend_and_training_type(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[str, str]:
    backend = args.backend or config.get("backend") or "flagembedding"
    training_type = args.training_type or config.get("training_type") or config.get("training-type") or "embedder"
    return normalize_backend(str(backend)), normalize_training_type(str(training_type))


def run_training(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    backend, training_type = resolve_backend_and_training_type(args, config)
    spec = validate_backend_training_type(backend, training_type)
    backend_module = load_backend_module(spec)
    request = TrainingRequest(
        backend=backend,
        training_type=training_type,
        config=config,
        config_path=args.config,
        cli_args=args,
    )
    result = backend_module.run_training(request)
    return 0 if result is None else int(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
