"""Checkpoint discovery and resume helpers for training backends."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any

from training.backends.registry import TrainingCliError


LATEST_CHECKPOINT = "latest"
DEFAULT_EPOCH_CHECKPOINT_DIR = "epoch-checkpoints"

_STEP_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_EPOCH_CHECKPOINT_RE = re.compile(r"^epoch-(\d+)-step-(\d+)$")


class CheckpointError(TrainingCliError):
    """Raised when a requested checkpoint cannot be resolved."""


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_resume_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return LATEST_CHECKPOINT if value else None
    raw_value = str(value).strip()
    if not raw_value:
        return None
    if raw_value.lower() in {"1", "true", "yes", "y", "on", LATEST_CHECKPOINT}:
        return LATEST_CHECKPOINT
    return raw_value


def resume_spec_from_sources(
    config: dict[str, Any],
    backend_config: dict[str, Any],
    cli_args: Any,
) -> str | None:
    cli_checkpoint = getattr(cli_args, "resume_from_checkpoint", None)
    if cli_checkpoint:
        return str(cli_checkpoint)
    if as_bool(getattr(cli_args, "resume", False)):
        return LATEST_CHECKPOINT

    for source in (backend_config, config):
        resume_from_checkpoint = normalize_resume_value(source.get("resume_from_checkpoint"))
        if resume_from_checkpoint is not None:
            return resume_from_checkpoint
        if as_bool(source.get("resume", False)):
            return LATEST_CHECKPOINT
    return None


def epoch_checkpoint_dir_name(config: dict[str, Any], backend_config: dict[str, Any]) -> str:
    value = backend_config.get("epoch_checkpoint_dir", config.get("epoch_checkpoint_dir"))
    if value is None or str(value).strip() == "":
        return DEFAULT_EPOCH_CHECKPOINT_DIR
    return str(value)


def checkpoint_step(path: Path) -> int:
    step_match = _STEP_CHECKPOINT_RE.match(path.name)
    if step_match:
        return int(step_match.group(1))

    epoch_match = _EPOCH_CHECKPOINT_RE.match(path.name)
    if epoch_match:
        return int(epoch_match.group(2))

    state_path = path / "trainer_state.json"
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return -1
        try:
            return int(data.get("global_step", -1))
        except (TypeError, ValueError):
            return -1
    return -1


def _checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return checkpoint_step(path), mtime, path.name


def iter_checkpoint_dirs(output_dir: Path, *, epoch_checkpoint_dir: str = DEFAULT_EPOCH_CHECKPOINT_DIR) -> list[Path]:
    candidates: list[Path] = []
    if output_dir.exists():
        candidates.extend(path for path in output_dir.glob("checkpoint-*") if path.is_dir())
        epoch_root = output_dir / epoch_checkpoint_dir
        if epoch_root.exists():
            candidates.extend(path for path in epoch_root.iterdir() if path.is_dir())
    return candidates


def find_latest_checkpoint(
    output_dir: Path,
    *,
    epoch_checkpoint_dir: str = DEFAULT_EPOCH_CHECKPOINT_DIR,
) -> Path | None:
    candidates = iter_checkpoint_dirs(output_dir, epoch_checkpoint_dir=epoch_checkpoint_dir)
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_sort_key)


def ensure_checkpoint_path(path: Path) -> Path:
    if not path.exists():
        raise CheckpointError(f"Requested checkpoint '{path}' does not exist.")
    if not path.is_dir():
        raise CheckpointError(f"Requested checkpoint '{path}' is not a directory.")
    return path


def output_dir_from_checkpoint(path: Path, *, epoch_checkpoint_dir: str = DEFAULT_EPOCH_CHECKPOINT_DIR) -> Path:
    checkpoint = ensure_checkpoint_path(path)
    if checkpoint.parent.name == epoch_checkpoint_dir:
        return checkpoint.parent.parent
    return checkpoint.parent


def resolve_resume_checkpoint(
    output_dir: Path,
    config: dict[str, Any],
    backend_config: dict[str, Any],
    cli_args: Any,
) -> str | None:
    resume_spec = resume_spec_from_sources(config, backend_config, cli_args)
    if resume_spec is None:
        return None

    epoch_dir = epoch_checkpoint_dir_name(config, backend_config)
    if resume_spec == LATEST_CHECKPOINT:
        checkpoint = find_latest_checkpoint(output_dir, epoch_checkpoint_dir=epoch_dir)
        if checkpoint is None:
            raise CheckpointError(
                f"Resume requested, but no checkpoint-* or {epoch_dir} checkpoint was found under '{output_dir}'."
            )
        return str(checkpoint)

    return str(ensure_checkpoint_path(Path(resume_spec)))


class EpochCheckpointCallback:
    """Copy end-of-epoch checkpoints outside Trainer's save_total_limit rotation."""

    def __init__(self, output_dir: Path | str, *, epoch_checkpoint_dir: str = DEFAULT_EPOCH_CHECKPOINT_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.epoch_checkpoint_dir = epoch_checkpoint_dir
        self._pending_epoch_checkpoint: tuple[int, int] | None = None
        self._fallback_epoch = 0

    def __getattr__(self, name: str) -> Any:
        if not name.startswith("on_"):
            raise AttributeError(name)

        def _noop(args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            return control

        return _noop

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0:
            return control

        epoch = getattr(state, "epoch", None)
        if epoch is None:
            self._fallback_epoch += 1
            epoch_index = self._fallback_epoch
        else:
            try:
                epoch_index = max(1, int(round(float(epoch))))
            except (TypeError, ValueError):
                self._fallback_epoch += 1
                epoch_index = self._fallback_epoch

        self._pending_epoch_checkpoint = (epoch_index, step)
        if hasattr(control, "should_save"):
            control.should_save = True
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        pending = self._pending_epoch_checkpoint
        self._pending_epoch_checkpoint = None
        if pending is None:
            return control

        epoch_index, step = pending
        source = self.output_dir / f"checkpoint-{step}"
        if not source.is_dir():
            return control

        target_root = self.output_dir / self.epoch_checkpoint_dir
        target_root.mkdir(parents=True, exist_ok=True)
        target = target_root / f"epoch-{epoch_index:04d}-step-{step}"
        if target.exists():
            return control

        tmp_target = target_root / f".{target.name}.tmp"
        if tmp_target.exists():
            shutil.rmtree(tmp_target)
        shutil.copytree(source, tmp_target)
        tmp_target.rename(target)
        print(f"[INFO] Preserved epoch checkpoint: {target}", flush=True)
        return control


def build_epoch_checkpoint_callback(
    output_dir: Path,
    config: dict[str, Any],
    backend_config: dict[str, Any],
) -> EpochCheckpointCallback | None:
    if not as_bool(backend_config.get("keep_epoch_checkpoints", config.get("keep_epoch_checkpoints", False))):
        return None
    return EpochCheckpointCallback(
        output_dir,
        epoch_checkpoint_dir=epoch_checkpoint_dir_name(config, backend_config),
    )


def train_with_resume(trainer: Any, resume_from_checkpoint: str | None) -> Any:
    if resume_from_checkpoint:
        print(f"[INFO] Resuming training from checkpoint: {resume_from_checkpoint}", flush=True)
        return trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer.train()
