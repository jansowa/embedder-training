from pathlib import Path
import runpy

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "experiments/polish-splade-2026-10/queues/queue_5070_p1_ab.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_weight_three_is_conditional_on_measured_component_sizes():
    needed = MODULE["weight_three_needed"]
    assert needed({"kl": 2, "mnrl": 0.1})
    assert not needed({"kl": 0.2, "mnrl": 0.1})
    assert not needed({"kl": 2, "mnrl": 2})
    assert not needed({})
    assert not needed({"kl": float("nan"), "mnrl": 0.1})


def test_component_logging_keeps_the_original_loss_and_computes_it_once():
    class Loss:
        def detach(self):
            return self
        def item(self):
            return 2.5
    loss = Loss()
    calls = []
    def calculate(*args):
        calls.append(args)
        return loss
    stats = MODULE["Components"]()
    wrapped = stats.capture("kl", calculate)
    assert wrapped("embeddings", "labels") is loss
    assert calls == [("embeddings", "labels")]
    assert stats.flush() == {"kl": 2.5}
    assert stats.flush() == {}
    assert stats.last == {"kl": 2.5} and stats.means() == {"kl": 2.5}


def test_ab_waits_for_primary_completion_and_does_not_use_g1_as_a_gate(monkeypatch, tmp_path):
    ready = MODULE["primary_ready"]
    monkeypatch.setitem(ready.__globals__, "RECOVERY_STATE", tmp_path / "missing.json")
    assert not ready({"status": "training", "g1": "tiny10_below_59.5"})
    assert ready({"status": "completed", "g1": "density_outside_band"})
    with pytest.raises(RuntimeError, match="review"):
        ready({"status": "failed"})


def test_ab_waits_for_measurement_recovery_but_not_a_failed_recovery(monkeypatch, tmp_path):
    import json
    path = tmp_path / "recovery.json"
    ready = MODULE["primary_ready"]
    monkeypatch.setitem(ready.__globals__, "RECOVERY_STATE", path)
    path.write_text(json.dumps({"status": "evaluating"}))
    assert not ready({"status": "failed"})
    path.write_text(json.dumps({"status": "failed"}))
    with pytest.raises(RuntimeError):
        ready({"status": "failed"})


def test_ab_worker_requires_approval_before_backend_import(monkeypatch):
    monkeypatch.delenv("P1_AB_APPROVED", raising=False)
    with pytest.raises(RuntimeError, match="approval"):
        MODULE["worker"](Path("missing.yaml"), "smoke")


def test_control_and_mnrl_arms_have_identical_data_and_training_settings():
    folder = SCRIPT.parents[1] / "configs/next/5070"
    control = yaml.safe_load((folder / "mlm_p1_ab_kl_control.yaml").read_text())
    combined = yaml.safe_load((folder / "mlm_p1_ab_kl_mnrl.yaml").read_text())
    assert control["hparams"][0]["loss"] == "sparse_distill_kl_div"
    assert combined["hparams"][0]["loss"] == "sparse_kl_plus_mnrl"
    assert combined["sentence_transformers"]["mnrl_weight"] == 1
    for cfg in (control, combined):
        assert cfg["hparams"][0]["num_train_epochs"] == 1
        cfg.pop("runs_dir")
        cfg["hparams"][0].pop("loss")
        cfg["sentence_transformers"].pop("mnrl_weight", None)
    assert control == combined
