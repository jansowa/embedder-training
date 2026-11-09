from pathlib import Path
import json
import runpy
import struct
from types import SimpleNamespace
import zipfile

import pytest

FOLDER = Path(__file__).resolve().parents[1] / "experiments/polish-splade-2026-10/queues"
MODULE = runpy.run_path(str(FOLDER / "queue_5070_p1_service.py"))
RESUME = runpy.run_path(str(FOLDER / "queue_5070_p1_resume.py"))


def checkpoint(path, step):
    path.mkdir(parents=True)
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1))
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 32595, "epoch": step / 10866}))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        with zipfile.ZipFile(path / name, "w") as archive:
            archive.writestr("tensor", b"fixture")


def test_recovery_selects_recent_intact_checkpoint_and_skips_partial_saves(tmp_path):
    checkpoint(tmp_path / "checkpoint-22500", 22500)
    partial = tmp_path / "checkpoint-23000"
    checkpoint(partial, 23000)
    (partial / "optimizer.pt").write_bytes(b"truncated torch save")
    checkpoint(tmp_path / "checkpoint-22000", 22000)
    selected, state = MODULE["latest_intact"](tmp_path)
    assert selected.name == "checkpoint-22500" and state["global_step"] == 22500
    (tmp_path / "checkpoint-22500/model.safetensors").write_bytes(b"truncated")
    selected, state = MODULE["latest_intact"](tmp_path)
    assert selected.name == "checkpoint-22000"


def test_resume_accepts_an_interior_epoch_checkpoint_and_rejects_wrong_state():
    approval = {"resume_step": 22500, "resume_epoch": 2.0706}
    state = SimpleNamespace(global_step=22500, max_steps=32595, epoch=2.0706)
    RESUME["assert_restored_state"](state, approval)
    state.global_step = 21732
    with pytest.raises(ValueError, match="approved checkpoint"):
        RESUME["assert_restored_state"](state, approval)


def test_repeated_no_progress_failures_eventually_stop_retrying(monkeypatch, tmp_path):
    fn = MODULE["run_kl"]
    namespace = fn.__globals__
    service_state = tmp_path / "service.json"
    state = tmp_path / "state.json"
    for key, value in (("SERVICE_STATE", service_state), ("STATE", state), ("ROOT", tmp_path)):
        monkeypatch.setitem(namespace, key, value)
    run = tmp_path / "run"
    monkeypatch.setitem(namespace, "locate_run", lambda: run)
    monkeypatch.setitem(namespace, "latest_intact", lambda run: (run / "checkpoint-22500", {"global_step": 22500}))
    monkeypatch.setitem(namespace, "recovery_approval", lambda *args: tmp_path / "approval.json")
    monkeypatch.setitem(namespace, "environment", lambda: {})
    monkeypatch.setattr(namespace["subprocess"], "run", lambda *args, **kwargs: SimpleNamespace(returncode=-9))
    assert fn() == 1
    assert fn() == 1
    assert fn() == 2
    assert json.loads(state.read_text())["status"] == "failed"
    assert json.loads(service_state.read_text())["consecutive_failures"] == 3


def test_service_has_control_group_supervision_and_bounded_restart():
    text = (FOLDER.parent / "services/splade-p1.service.in").read_text()
    for setting in ("KillMode=control-group", "Restart=on-failure", "RestartPreventExitStatus=2", "StartLimitBurst=3"):
        assert setting in text


def test_three_unclean_service_stops_do_not_launch_another_gpu_attempt(monkeypatch, tmp_path):
    fn = MODULE["run_kl"]
    namespace = fn.__globals__
    service_state = tmp_path / "service.json"
    service_state.write_text(json.dumps({"status": "running", "consecutive_failures": 2}))
    monkeypatch.setitem(namespace, "SERVICE_STATE", service_state)
    monkeypatch.setitem(namespace, "STATE", tmp_path / "state.json")
    monkeypatch.setitem(namespace, "locate_run", lambda: pytest.fail("Must not start another recovery attempt"))
    assert fn() == 2
    assert json.loads(service_state.read_text())["status"] == "failed"
