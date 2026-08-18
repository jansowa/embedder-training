"""DDP-safe wrapper for the SentenceTransformers backend.

This module keeps the existing SentenceTransformers implementation intact while
ensuring that callbacks which mutate TrainerControl are installed and executed
on every distributed rank. File copying remains restricted to rank 0 by the
callbacks' on_save handlers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.backends import sentence_transformers_backend as _backend
from training.backends.registry import TrainingRequest
from training.checkpoints import (
    EpochCheckpointCallback,
    build_epoch_checkpoint_callback,
    build_step_checkpoint_callback,
)


class DistributedEpochCheckpointCallback(EpochCheckpointCallback):
    """Request an epoch checkpoint consistently on every distributed rank."""

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


def _build_distributed_epoch_checkpoint_callback(
    output_dir: Path,
    config: dict[str, Any],
    backend_config: dict[str, Any],
) -> DistributedEpochCheckpointCallback | None:
    callback = build_epoch_checkpoint_callback(output_dir, config, backend_config)
    if callback is None:
        return None
    return DistributedEpochCheckpointCallback(
        callback.output_dir,
        epoch_checkpoint_dir=callback.epoch_checkpoint_dir,
    )


def _add_checkpoint_callbacks(
    trainer: Any,
    output_dir: Path,
    config: dict[str, Any],
    backend_config: dict[str, Any],
) -> None:
    """Install control-mutating callbacks on all ranks.

    The callbacks themselves retain their rank-0 guards in ``on_save``, so only
    the main process copies checkpoint files. Registering them everywhere keeps
    ``TrainerControl.should_save`` identical across ranks and prevents ranks
    from entering different NCCL collectives.
    """

    if not hasattr(trainer, "add_callback"):
        return

    for callback in (
        _build_distributed_epoch_checkpoint_callback(output_dir, config, backend_config),
        build_step_checkpoint_callback(output_dir, config, backend_config),
    ):
        if callback is not None:
            trainer.add_callback(callback)


def run_training(request: TrainingRequest) -> int:
    # The original run_* functions resolve _add_checkpoint_callbacks through
    # their defining module's globals, so replacing it here affects all three
    # SentenceTransformers recipes without duplicating their training logic.
    _backend._add_checkpoint_callbacks = _add_checkpoint_callbacks
    return _backend.run_training(request)
