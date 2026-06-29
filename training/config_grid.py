"""Shared configuration-grid expansion for training backends."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


BACKEND_SECTION_KEYS = {
    "flagembedding": "flagembedding",
    "sentence-transformers": "sentence_transformers",
    "pylate": "pylate",
}

GRID_KEYS = {"architectures", "hparams"}

HPARAM_SLUG_LABELS = {
    "learning_rate": "lr",
    "num_train_epochs": "ep",
    "train_data": "data",
    "max_steps": "steps",
    "loss": "loss",
    "document_regularizer_weight": "docreg",
    "query_regularizer_weight": "qreg",
    "score_normalization": "score-norm",
}

HPARAM_SLUG_ORDER = (
    "learning_rate",
    "num_train_epochs",
    "max_steps",
    "loss",
    "document_regularizer_weight",
    "query_regularizer_weight",
    "score_normalization",
    "train_data",
)


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _dict_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _as_list(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise ValueError(f"Grid field '{field_name}' must be a list.")


def _as_hparams_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Grid field 'hparams' must be a list of mappings.")
    if not value:
        return []
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("Grid field 'hparams' must contain only mappings.")
    return [dict(item) for item in value]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _first_config_value(config: dict[str, Any], backend: str, key: str) -> Any:
    backend_section_key = BACKEND_SECTION_KEYS.get(backend)
    for section in (
        _dict_section(config, backend_section_key) if backend_section_key else {},
        _dict_section(config, "backend_config"),
        config,
    ):
        if key in section:
            return section[key]
    return None


def _safe_slug(value: Any) -> str:
    slug = str(value).strip().replace("/", "_").replace(".", "_")
    allowed = []
    for char in slug:
        if char.isalnum() or char in {"_", "-"}:
            allowed.append(char)
        else:
            allowed.append("-")
    cleaned = "".join(allowed).strip("-_")
    return cleaned or "run"


def _hparam_value_slug(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        raw = json.dumps(value, ensure_ascii=True, sort_keys=True)
    else:
        raw = str(value)
    slug = _safe_slug(raw)
    if len(slug) <= 80:
        return slug
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:69].rstrip('-_')}-{digest}"


def _hparam_slug_keys(hparams: dict[str, Any]) -> list[str]:
    ordered = [key for key in HPARAM_SLUG_ORDER if key in hparams]
    ordered.extend(sorted(key for key in hparams if key not in HPARAM_SLUG_ORDER))
    return ordered


def _hparam_slug(hparams: dict[str, Any]) -> str:
    parts = []
    for key in _hparam_slug_keys(hparams):
        label = HPARAM_SLUG_LABELS.get(key, _safe_slug(key))
        parts.append(f"{label}-{_hparam_value_slug(hparams[key])}")
    if parts:
        return "-".join(parts)
    return "default"


def _run_slug(architecture: Any, hparams: dict[str, Any]) -> str:
    return f"{_safe_slug(architecture)}-{_hparam_slug(hparams)}"


def _set_backend_override(config: dict[str, Any], backend: str, values: dict[str, Any]) -> None:
    backend_section_key = BACKEND_SECTION_KEYS.get(backend)
    if not backend_section_key:
        return
    section = config.get(backend_section_key)
    if section is None:
        section = {}
        config[backend_section_key] = section
    if isinstance(section, dict):
        section.update(values)


def _strip_grid_keys(config: dict[str, Any]) -> None:
    for key in GRID_KEYS:
        config.pop(key, None)
    for section_key in ("backend_config", *BACKEND_SECTION_KEYS.values()):
        section = config.get(section_key)
        if isinstance(section, dict):
            for key in GRID_KEYS:
                section.pop(key, None)


def expand_config_grid(config: dict[str, Any], *, backend: str, training_type: str) -> list[dict[str, Any]]:
    """Expand root/backend `architectures` x `hparams` into per-run configs.

    The expanded configs use shared top-level keys and the selected backend
    section so backend-specific precedence still lets grid values win.
    """

    backend_section_key = BACKEND_SECTION_KEYS.get(backend)
    backend_section = _dict_section(config, backend_section_key) if backend_section_key else {}

    architectures = _as_list(
        config.get("architectures", backend_section.get("architectures")),
        field_name="architectures",
    )
    hparams = _as_hparams_list(config.get("hparams", backend_section.get("hparams")))

    if not architectures and not hparams:
        return [config]

    if not architectures:
        configured_model = _first_config_value(config, backend, "model_name_or_path")
        architectures = [configured_model] if configured_model is not None else [None]
    if not hparams:
        hparams = [{}]

    variants: list[dict[str, Any]] = []
    total_variants = len(architectures) * len(hparams)
    root_output_dir = _first_config_value(config, backend, "output_dir")
    runs_dir = _first_config_value(config, backend, "runs_dir") or "runs"
    timestamp_output_dir = _as_bool(_first_config_value(config, backend, "timestamp_output_dir"))
    run_timestamp = _timestamp_slug() if timestamp_output_dir else None

    for architecture in architectures:
        for hparam in hparams:
            variant = deepcopy(config)
            _strip_grid_keys(variant)

            overrides = dict(hparam)
            if architecture is not None:
                overrides["model_name_or_path"] = architecture
            variant.update(overrides)
            variant["grid_architecture"] = architecture
            variant["grid_hparams"] = dict(hparam)
            variant["run_name"] = _run_slug(architecture, hparam)
            if run_timestamp is not None:
                variant["run_timestamp"] = run_timestamp
                variant["run_name"] = f"{variant['run_name']}-{run_timestamp}"
            variant["backend"] = backend
            variant["training_type"] = training_type
            _set_backend_override(variant, backend, overrides)

            if "output_dir" in hparam and timestamp_output_dir:
                variant["output_dir"] = str(Path(str(hparam["output_dir"])) / variant["run_name"])
                _set_backend_override(variant, backend, {"output_dir": variant["output_dir"]})
            elif "output_dir" not in hparam:
                if root_output_dir is not None and (total_variants > 1 or timestamp_output_dir):
                    variant["output_dir"] = str(Path(str(root_output_dir)) / variant["run_name"])
                    _set_backend_override(variant, backend, {"output_dir": variant["output_dir"]})
                elif root_output_dir is None:
                    variant["output_dir"] = str(Path(str(runs_dir)) / backend / training_type / variant["run_name"])
                    _set_backend_override(variant, backend, {"output_dir": variant["output_dir"]})

            variants.append(variant)

    return variants
