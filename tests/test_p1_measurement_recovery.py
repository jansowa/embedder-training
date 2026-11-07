import json
from pathlib import Path
import runpy

import pytest


MODULE = runpy.run_path(str(Path(__file__).parents[1] / "experiments/polish-splade-2026-10/queues/recover_p1_measurements.py"))


def finished_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "resolved_config.yaml").write_text("training_type: splade\n")
    (run / "final").mkdir()
    (run / "final/model.safetensors").write_bytes(b"final")
    for epoch in (1, 2, 3):
        path = run / f"epoch-checkpoints/epoch-{epoch:04d}-step-{epoch*10}"
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(b"weights")
        (path / "trainer_state.json").write_text(json.dumps({"epoch": epoch if epoch<3 else 2.9997,
            "global_step": epoch*10,"max_steps":30,"log_history":[{"loss":1.1,"grad_norm":2.0}]}))
    return run


def test_recovery_requires_all_native_epochs_and_final_model(tmp_path):
    run = finished_run(tmp_path)
    actual, checkpoints = MODULE["validated_finished_run"](tmp_path)
    assert actual == run and [r[0] for r in checkpoints] == [1,2,3]
    (run / "final/model.safetensors").unlink()
    with pytest.raises(ValueError, match="Final"):
        MODULE["validated_finished_run"](tmp_path)


def test_recovery_refuses_an_incomplete_or_nonfinite_run(tmp_path):
    run = finished_run(tmp_path)
    state_path = run / "epoch-checkpoints/epoch-0003-step-30/trainer_state.json"
    state = json.loads(state_path.read_text())
    state["global_step"] = 29
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="have not finished"):
        MODULE["validated_finished_run"](tmp_path)
    state["global_step"] = 30
    state["log_history"][0]["grad_norm"] = float("nan")
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Non-finite"):
        MODULE["validated_finished_run"](tmp_path)
