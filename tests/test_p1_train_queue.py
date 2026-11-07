import json
from pathlib import Path
import runpy

import pytest


SCRIPT = Path(__file__).parents[1] / "experiments/polish-splade-2026-10/queues/queue_5070_p1_train.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_g1_boundaries_are_fixed_before_training():
    gate = MODULE["g1_result"]
    assert gate(59.5, 330) == "pass"
    assert gate(59.5, 460) == "pass"
    assert gate(59.49, 400) == "tiny10_below_59.5"
    assert gate(65, 329.9) == "density_outside_band"
    assert gate(65, 460.1) == "density_outside_band"
    assert gate(float("nan"), 400) == "invalid"


def test_teacher_validation_rejects_other_pool_and_double_scaling():
    pool = {"rows": 2, "examples": 2, "sha256": "approved"}
    side = {"rows": 2, "examples": 2, "input_pool_sha256": "approved", "dense_scale": 123, "training_score_scale": 1}
    MODULE["validate_teacher"](side, pool)
    with pytest.raises(ValueError, match="approved KL pool"):
        MODULE["validate_teacher"](dict(side, input_pool_sha256="old-pool"), pool)
    with pytest.raises(ValueError, match="second time"):
        MODULE["validate_teacher"](dict(side, training_score_scale=123), pool)
    with pytest.raises(ValueError, match="Invalid dense"):
        MODULE["validate_teacher"](dict(side, dense_scale=float("nan")), pool)


def test_failed_scoring_never_triggers_training(monkeypatch, tmp_path):
    wait = MODULE["wait_for_scoring"]
    monkeypatch.setitem(wait.__globals__, "SCORED", tmp_path / "missing.jsonl")
    monkeypatch.setitem(wait.__globals__, "scoring_alive", lambda: False)
    with pytest.raises(RuntimeError, match="training not started"):
        wait({"rows": 2, "examples": 2, "sha256": "approved"}, lambda *a, **k: None)


def test_complete_sidecar_still_requires_matching_scored_bytes(monkeypatch, tmp_path):
    wait = MODULE["wait_for_scoring"]
    output = tmp_path / "scores.jsonl"
    output.write_text("valid scored data\n")
    side = {"rows": 2, "examples": 2, "input_pool_sha256": "approved", "dense_scale": 12, "training_score_scale": 1,
            "sha256": MODULE["digest"](output)}
    Path(str(output) + ".teacher.json").write_text(json.dumps(side))
    monkeypatch.setitem(wait.__globals__, "SCORED", output)
    assert wait({"rows": 2, "examples": 2, "sha256": "approved"}, lambda *a, **k: None) == side
    output.write_text("truncated")
    with pytest.raises(ValueError, match="checksum"):
        wait({"rows": 2, "examples": 2, "sha256": "approved"}, lambda *a, **k: None)


def test_result_commit_whitelist_cannot_include_arbitrary_paths():
    paths = MODULE["result_paths"]({"epochs": [{"epoch": 1, "label": "../../unrelated"}]})
    assert all(".." not in path.parts for path in paths)
    assert any(str(path).endswith("quantized/mlm-p1-ep1/tiny10.json") for path in paths)
    with pytest.raises(ValueError, match="epoch"):
        MODULE["result_paths"]({"epochs": [{"epoch": 8}]})


def test_density_controller_uses_the_same_band_as_the_g1_gate():
    import yaml
    config = yaml.safe_load(MODULE["CONFIG"].read_text())
    control = config["sentence_transformers"]["regularizer_control"]["density"]
    assert control["reference_dims"] * control["band"][0] == pytest.approx(330)
    assert control["reference_dims"] * control["band"][1] == pytest.approx(460)


def test_weak_or_dense_epoch_one_is_informational_and_all_three_epochs_are_recorded():
    result = {"status": "training", "epochs": []}
    for epoch, score, density in [(1, 55.0, 600), (2, 58.0, 650), (3, 60.0, 620)]:
        MODULE["record_epoch"](result, {"epoch": epoch, "tiny10": score, "density": {"doc_active_mean": density}})
    assert result["status"] == "training"
    assert len(result["epochs"]) == 3
    assert "informational" in result["g1_policy"]
    with pytest.raises(ValueError, match="consecutive"):
        MODULE["record_epoch"](result, {"epoch": 5})


def test_observer_survives_the_backend_registered_ddp_dispatcher(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend as core
    from training.backends import sentence_transformers_ddp_backend as ddp
    from training.backends.registry import BACKENDS, load_backend_module
    assert load_backend_module(BACKENDS["sentence-transformers"]) is ddp
    callbacks = []
    class Trainer:
        def add_callback(self, callback):
            callbacks.append(callback)
    monkeypatch.setattr(core, "_add_checkpoint_callbacks", core._add_checkpoint_callbacks)
    monkeypatch.setattr(ddp, "_add_checkpoint_callbacks", ddp._add_checkpoint_callbacks)
    def run(request):
        core._add_checkpoint_callbacks(Trainer(), tmp_path, {}, {})
        return 0
    monkeypatch.setattr(core, "run_training", run)
    observer = object()
    MODULE["attach_checkpoint_observer"](lambda: observer)
    assert ddp.run_training(None) == 0
    assert callbacks == [observer]


def test_preserved_checkpoint_identifies_last_epoch_despite_fractional_rounding(tmp_path):
    path = tmp_path / "epoch-checkpoints/epoch-0003-step-32595"
    path.mkdir(parents=True)
    assert MODULE["preserved_epoch"](tmp_path, 32595) == (3, path)
    assert MODULE["preserved_epoch"](tmp_path, 32000) is None
