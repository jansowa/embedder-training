"""Run a bounded diagnostic training via the normal CLI, with a dead-gradient guard.

Only this entry point installs the callback; regular training is unchanged.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path


def update_zero_streak(previous: int, value: float | None) -> int:
    return previous + 1 if value is not None and value == 0 else 0


def save_control_state(state, output_dir: str) -> None:
    state.save_to_json(str(Path(output_dir) / "control_train_state.json"))


def main() -> int:
    import sentence_transformers.sparse_encoder as sparse_encoder
    from transformers import TrainerCallback
    from training.train import main as train_main

    started = time.monotonic()

    class ControlMonitor(TrainerCallback):
        def __init__(self):
            self.zero_streak = 0

        def on_train_begin(self, args, state, control, **kwargs):
            print("[CONTROL] dead-gradient guard installed (3 logged zero norms)", flush=True)

        def on_train_end(self, args, state, control, **kwargs):
            save_control_state(state, args.output_dir)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            grad = logs.get("grad_norm")
            self.zero_streak = update_zero_streak(self.zero_streak, grad)
            reason = None
            if self.zero_streak >= 3:
                reason = "three consecutive logged zero gradient norms"
            if any(not math.isfinite(float(logs[k])) for k in ("loss", "grad_norm") if k in logs):
                reason = "nonfinite training loss or gradient norm"
            row = {"step": state.global_step, "elapsed_seconds": time.monotonic() - started,
                   "metrics": logs, "stop_reason": reason}
            try:
                row["gpu"] = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                row["gpu"] = None
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            with (output / "collapse_control.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            if reason:
                control.should_training_stop = True
                (output / "collapse_control_stop.json").write_text(json.dumps(row, indent=2) + "\n")
                print(f"[CONTROL STOP] step {state.global_step}: {reason}", flush=True)
            return control

    original = sparse_encoder.SparseEncoderTrainer

    class DiagnosticTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(ControlMonitor())

    sparse_encoder.SparseEncoderTrainer = DiagnosticTrainer
    return train_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
