import json
from pathlib import Path
import runpy
import struct
from types import SimpleNamespace
import zipfile

import pytest

FOLDER = Path(__file__).parents[1] / "experiments/polish-splade-2026-10/queues"
MODULE = runpy.run_path(str(FOLDER / "queue_5070_p1_continue_ab.py"))


def checkpoint(path, step, start):
    path.mkdir(parents=True)
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1))
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 1250, "epoch": step / 1250}))
    (path / "continuation-start.json").write_text(json.dumps(start))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        with zipfile.ZipFile(path / name, "w") as archive:
            archive.writestr("tensor", b"state")


def test_selects_intact_arm_checkpoint_and_rejects_different_start(tmp_path):
    start = {"checkpoint": "common", "weights_sha256": "original"}
    checkpoint(tmp_path / "checkpoint-100", 100, start)
    checkpoint(tmp_path / "checkpoint-200", 200, start)
    (tmp_path / "checkpoint-200/optimizer.pt").write_bytes(b"partial")
    selected, state = MODULE["arm_checkpoint"](tmp_path, start)
    assert selected.name == "checkpoint-100" and state["global_step"] == 100
    with pytest.raises(ValueError, match="different starting model"):
        MODULE["arm_checkpoint"](tmp_path, dict(start, weights_sha256="changed"))


def test_resume_policy_updates_restored_save_interval():
    args = SimpleNamespace(save_strategy="epoch", save_steps=500, save_total_limit=1)
    state = SimpleNamespace(save_steps=500, global_step=100)
    MODULE["apply_arm_checkpoint_policy"](args, state)
    assert args.save_strategy == "steps" and args.save_steps == state.save_steps == 100
    assert args.save_total_limit == 2 and state.global_step == 100


def test_discarded_branch_metrics_are_archived_not_mixed_with_resume(tmp_path):
    path = tmp_path / "p1-ab-telemetry.jsonl"
    original = '{"step": 100}\n{"step": 125}\n{"step": '
    path.write_text(original)
    MODULE["archive_interrupted_telemetry"](tmp_path, 100)
    assert path.read_text() == '{"step": 100}\n'
    archive = list(tmp_path.glob("p1-ab-telemetry-before-resume-*.jsonl"))
    assert len(archive) == 1 and archive[0].read_text() == original


def test_run_arm_preserves_original_run_path_when_resuming(monkeypatch, tmp_path):
    fn = MODULE["run_arm"]
    namespace = fn.__globals__
    monkeypatch.setitem(namespace, "ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "runs").mkdir()
    start = {"checkpoint": "common", "weights_sha256": "original"}
    start_path = tmp_path / "logs/start.json"
    start_path.write_text(json.dumps(start))
    config = tmp_path / "logs/config.yaml"
    config.write_text("frozen config")
    spec = {"snapshot": str(config), "sha256": MODULE["digest"](config), "run_name": "arm-kl"}
    target = MODULE["prepare_run"]("kl", "arm-kl")
    run = target / "training"
    checkpoint(run / "checkpoint-100", 100, start)
    (run / "resolved_config.yaml").write_text("frozen run")
    with pytest.raises(RuntimeError, match="implicit continuation restart"):
        MODULE["prepare_run"]("kl", "arm-kl")
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        (tmp_path / "logs/mlm-p1-continue-ab-kl-result.json").write_text(json.dumps({"tag": "kl"}))
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(namespace["subprocess"], "Popen", execute)
    result = fn(spec, "kl", start_path, {}, resume=True)
    assert result["tag"] == "kl"
    command = commands[0]
    assert command[command.index("--resume-checkpoint") + 1] == str(Path("runs/arm-kl/training/checkpoint-100"))


def test_completed_smoke_is_reused_without_launching_training(monkeypatch, tmp_path):
    fn = MODULE["run_arm"]
    namespace = fn.__globals__
    monkeypatch.setitem(namespace, "ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "runs").mkdir()
    config = tmp_path / "logs/config.yaml"
    config.write_text("frozen smoke")
    spec = {"snapshot": str(config), "sha256": MODULE["digest"](config), "run_name": "arm-smoke"}
    final = tmp_path / "final"
    final.mkdir()
    (final / "model.safetensors").write_bytes(b"saved weights")
    result = {"tag": "smoke", "steps": 20, "config_sha256": spec["sha256"], "final": str(final)}
    (tmp_path / "logs/mlm-p1-continue-ab-smoke-result.json").write_text(json.dumps(result))
    monkeypatch.setattr(namespace["subprocess"], "Popen", lambda *a, **k: pytest.fail("Must reuse completed smoke"))
    assert fn(spec, "smoke", tmp_path / "unused", {}, resume=True) == result
