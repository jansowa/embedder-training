"""Registry and validation for training backends."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Any


class TrainingCliError(RuntimeError):
    """Base class for user-facing CLI errors."""


class UnsupportedBackendError(TrainingCliError):
    """Raised when a backend name is unknown."""


class UnsupportedTrainingCombination(TrainingCliError):
    """Raised when a backend does not support a training type."""


class BackendDependencyError(TrainingCliError):
    """Raised when the selected backend dependency is not installed."""


@dataclass(frozen=True)
class BackendSpec:
    name: str
    module: str
    supported_training_types: tuple[str, ...]
    install_hint: str


@dataclass(frozen=True)
class TrainingRequest:
    backend: str
    training_type: str
    config: dict[str, Any]
    config_path: str
    cli_args: Any


_BACKEND_ALIASES = {
    "flag": "flagembedding",
    "flagembedding": "flagembedding",
    "sentence_transformers": "sentence-transformers",
    "sentence-transformers": "sentence-transformers",
    "st": "sentence-transformers",
    "pylate": "pylate",
}

BACKENDS: dict[str, BackendSpec] = {
    "flagembedding": BackendSpec(
        name="flagembedding",
        module="training.backends.flagembedding_backend",
        supported_training_types=("embedder",),
        install_hint="pip install -r requirements/requirements-flagembedding.txt",
    ),
    "sentence-transformers": BackendSpec(
        name="sentence-transformers",
        module="training.backends.sentence_transformers_backend",
        supported_training_types=(
            "embedder",
            "matryoshka",
            "splade",
            "multimodal",
            "adaptive-layer",
            "matryoshka-2d",
        ),
        install_hint="pip install -r requirements/requirements-sentence-transformers.txt",
    ),
    "pylate": BackendSpec(
        name="pylate",
        module="training.backends.pylate_backend",
        supported_training_types=("colbert", "late-interaction"),
        install_hint="pip install -r requirements/requirements-pylate.txt",
    ),
}


def backend_names() -> tuple[str, ...]:
    return tuple(BACKENDS)


def all_training_types() -> tuple[str, ...]:
    values = {training_type for spec in BACKENDS.values() for training_type in spec.supported_training_types}
    return tuple(sorted(values))


def normalize_backend(name: str) -> str:
    key = name.strip().lower()
    try:
        return _BACKEND_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(backend_names())
        raise UnsupportedBackendError(f"Backend '{name}' is not supported. Supported backends: {supported}") from exc


def normalize_training_type(training_type: str) -> str:
    return training_type.strip().lower()


def get_backend_spec(backend: str) -> BackendSpec:
    backend = normalize_backend(backend)
    try:
        return BACKENDS[backend]
    except KeyError as exc:
        supported = ", ".join(backend_names())
        raise UnsupportedBackendError(f"Backend '{backend}' is not supported. Supported backends: {supported}") from exc


def validate_backend_training_type(backend: str, training_type: str) -> BackendSpec:
    spec = get_backend_spec(backend)
    normalized_type = normalize_training_type(training_type)
    if normalized_type not in spec.supported_training_types:
        supported = ", ".join(spec.supported_training_types)
        raise UnsupportedTrainingCombination(
            f"Training type '{training_type}' is not supported by backend '{spec.name}'. "
            f"Supported types: {supported}"
        )
    return spec


def load_backend_module(spec: BackendSpec) -> ModuleType:
    return importlib.import_module(spec.module)

