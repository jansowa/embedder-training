"""Immutable run metadata and compatibility checks for checkpoint resumes."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any

from training.backends.registry import TrainingCliError


MANIFEST_FILENAME = "training_manifest.json"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
COMMAND_FILENAME = "command.txt"
_SCHEMA_VERSION = 1
_RUNTIME_CONFIG_KEYS = frozenset({"resume", "resume_from_checkpoint", "resume_if_available", "run_timestamp"})


class RunMetadataError(TrainingCliError):
    """Raised when a checkpoint is incompatible with the requested run."""


def _yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RunMetadataError("YAML config support requires PyYAML to write or verify run metadata.") from exc
    return yaml


def _without_runtime_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_runtime_values(item)
            for key, item in value.items()
            if str(key) not in _RUNTIME_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [_without_runtime_values(item) for item in value]
    if isinstance(value, tuple):
        return [_without_runtime_values(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable fingerprint for settings that affect a training run."""
    normalized = _without_runtime_values(deepcopy(config))
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_text(cli_args: Any) -> str:
    raw_argv = list(getattr(cli_args, "_raw_argv", None) or [])
    return shlex.join(["python", "-m", "training.train", *raw_argv]) + "\n"


def _artifact_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / MANIFEST_FILENAME,
        output_dir / RESOLVED_CONFIG_FILENAME,
        output_dir / COMMAND_FILENAME,
    )


def create_run_metadata(output_dir: Path, config: dict[str, Any], cli_args: Any) -> None:
    """Persist an immutable baseline for a newly started training run."""
    manifest_path, config_path, command_path = _artifact_paths(output_dir)
    existing = [path.name for path in (manifest_path, config_path, command_path) if path.exists()]
    if existing:
        existing_text = ", ".join(existing)
        raise RunMetadataError(
            f"Output directory '{output_dir}' already contains run metadata ({existing_text}). "
            "Refusing to overwrite it; choose a new output_dir or resume the existing run."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    yaml = _yaml_module()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    command_path.write_text(command_text(cli_args), encoding="utf-8")
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint(config),
        "resolved_config_sha256": _file_sha256(config_path),
        "command_sha256": _file_sha256(command_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunMetadataError(f"Run metadata '{path}' cannot be read.") from exc
    if not isinstance(data, dict) or data.get("schema_version") != _SCHEMA_VERSION:
        raise RunMetadataError(f"Run metadata '{path}' has an unsupported format.")
    return data


def _saved_config_fingerprint(config_path: Path) -> str:
    yaml = _yaml_module()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunMetadataError(f"Saved resolved config '{config_path}' cannot be read.") from exc
    if not isinstance(config, dict):
        raise RunMetadataError(f"Saved resolved config '{config_path}' must contain a YAML mapping.")
    return config_fingerprint(config)


def verify_resume_metadata(output_dir: Path, config: dict[str, Any]) -> None:
    """Ensure a requested resume targets the exact same resolved training setup."""
    manifest_path, config_path, command_path = _artifact_paths(output_dir)
    expected_fingerprint = config_fingerprint(config)

    if manifest_path.exists():
        manifest = _read_manifest(manifest_path)
        if not config_path.exists() or not command_path.exists():
            raise RunMetadataError(
                f"Run metadata under '{output_dir}' is incomplete; both {RESOLVED_CONFIG_FILENAME} and "
                f"{COMMAND_FILENAME} are required to resume safely."
            )
        if manifest.get("resolved_config_sha256") != _file_sha256(config_path):
            raise RunMetadataError(f"Saved resolved config under '{output_dir}' was modified after the run started.")
        if manifest.get("command_sha256") != _file_sha256(command_path):
            raise RunMetadataError(f"Saved command under '{output_dir}' was modified after the run started.")
        saved_fingerprint = str(manifest.get("config_fingerprint", ""))
    else:
        # Checkpoints created before manifests existed can still be resumed when
        # their pre-existing config and command artifacts agree with this run.
        if not config_path.exists() or not command_path.exists():
            raise RunMetadataError(
                f"Cannot safely resume from '{output_dir}': {MANIFEST_FILENAME} is missing and legacy "
                f"{RESOLVED_CONFIG_FILENAME}/{COMMAND_FILENAME} artifacts are unavailable."
            )
        saved_fingerprint = _saved_config_fingerprint(config_path)
        print(f"[WARN] Resuming legacy run without {MANIFEST_FILENAME}; metadata integrity cannot be verified.", flush=True)

    if saved_fingerprint != expected_fingerprint:
        raise RunMetadataError(
            f"Checkpoint under '{output_dir}' does not match the current resolved training configuration. "
            "Use the original configuration or a different output_dir/checkpoint."
        )

