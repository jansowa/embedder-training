"""W&B run identity management shared by trainer-based backends."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator

from training.distributed import is_main_process
from training.run_metadata import get_or_create_wandb_run_id


def reports_to_wandb(report_to: Any) -> bool:
    if report_to is None:
        return False
    if isinstance(report_to, str):
        values = [report_to]
    else:
        try:
            values = list(report_to)
        except TypeError:
            values = [report_to]
    return any(str(value).lower() in {"all", "wandb"} for value in values)


@contextmanager
def wandb_run_environment(
    output_dir: Path,
    *,
    report_to: Any,
    resume_from_checkpoint: str | None,
    configured_run_id: str | None = None,
) -> Iterator[None]:
    """Expose a stable W&B run identity while a trainer initializes callbacks."""
    if not is_main_process() or not reports_to_wandb(report_to):
        yield
        return

    run_id = get_or_create_wandb_run_id(output_dir, requested_id=configured_run_id)
    if run_id is None:
        print(
            "[WARN] W&B run continuity is unavailable because this run has no training manifest.",
            flush=True,
        )
        yield
        return

    previous = {key: os.environ.get(key) for key in ("WANDB_RUN_ID", "WANDB_RESUME")}
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "must" if resume_from_checkpoint is not None else "never"
    action = "Resuming" if resume_from_checkpoint is not None else "Starting"
    print(f"[INFO] {action} W&B run: {run_id}", flush=True)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
