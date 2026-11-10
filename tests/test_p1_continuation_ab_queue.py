from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/polish-splade-2026-10/queues/queue_5070_p1_continue_ab.py"
MODULE = runpy.run_path(str(SCRIPT))
CONFIGS = SCRIPT.parents[1] / "configs/next/5070"


def template(tag):
    suffix = {"kl": "kl", "kl-mnrl": "kl_mnrl", "smoke": "smoke"}[tag]
    return yaml.safe_load((CONFIGS / f"mlm_p1_continue_ab_{suffix}.yaml").read_text())


def test_checkpoint_selection_waits_for_all_epochs_and_ignores_heldout_and_density():
    rows = [{"epoch": i, "tiny10": score, "heldout8": 100 - score, "density": {"doc_active_mean": dims}}
            for i, score, dims in [(1, 59, 350), (2, 61, 700), (3, 60, 400)]]
    selected = MODULE["choose_start"]({"status": "completed", "epochs": rows})
    assert selected["epoch"] == 2
    selected["tiny10"] = 0
    assert rows[1]["tiny10"] == 61
    for data in ({"status": "training", "epochs": rows}, {"status": "completed", "epochs": rows[:2]}):
        with pytest.raises(ValueError, match="Three"):
            MODULE["choose_start"](data)
    rows[2]["tiny10"] = float("nan")
    with pytest.raises(ValueError, match="metric"):
        MODULE["choose_start"]({"status": "completed", "epochs": rows})


def test_regularizers_are_taken_from_the_selected_step_not_future_controls(tmp_path):
    source = tmp_path / "run/epoch-checkpoints/epoch-0002-step-20"
    source.mkdir(parents=True)
    (source / "trainer_state.json").write_text(json.dumps({"global_step": 20}))
    history = source.parent.parent / "regularizer_control.json"
    history.write_text(json.dumps({"events": [
        {"step": 10, "event": "control", "document_weight": .003, "query_weight": .00015},
        {"step": 20, "event": "control", "document_weight": .0024, "query_weight": .00012},
        {"step": 30, "event": "control", "document_weight": .001, "query_weight": .00005}]}))
    weights = MODULE["selected_regularizers"](source)
    assert weights["document"] == .0024
    assert weights["query"] == .00012
    assert weights["source_step"] == 20


def test_binding_produces_matched_continuations_without_reinitialization():
    weights = {"document": .00192, "query": .000096}
    source = Path("/selected-checkpoint")
    configs = {tag: MODULE["bind_config"](template(tag), source, weights) for tag in ("kl", "kl-mnrl", "smoke")}
    MODULE["assert_matched"](configs["kl"], configs["kl-mnrl"])
    for cfg in configs.values():
        assert cfg["architectures"] == [str(source)]
        assert cfg["hparams"][0]["learning_rate"] == 5e-6
        assert cfg["hparams"][0]["document_regularizer_weight"] == weights["document"]
        backend = cfg["sentence_transformers"]
        assert backend["query_regularizer_weight"] == weights["query"]
        assert "splade_decoder_bias_shift" not in backend and "regularizer_control" not in backend
        assert backend["splade_regularizer_scheduler"] is False
        assert backend["save_strategy"] == "steps" and backend["save_steps"] == 500 and backend["save_total_limit"] == 2
        assert backend["train_batch_size"] == 8 and backend["gradient_accumulation_steps"] == 4
    assert configs["smoke"]["sentence_transformers"]["max_steps"] == 20
    modified = deepcopy(configs["kl-mnrl"])
    modified["sentence_transformers"]["seed"] = 43
    with pytest.raises(ValueError, match="differ"):
        MODULE["assert_matched"](configs["kl"], modified)
    modified = deepcopy(configs["kl-mnrl"])
    modified["sentence_transformers"]["mnrl_weight"] = 3
    with pytest.raises(ValueError, match="weight-one"):
        MODULE["assert_matched"](configs["kl"], modified)


def test_binding_rejects_mlm_starts_and_decoder_recalibration():
    weights = {"document": .003, "query": .00015}
    for change in ("architectures", "bias", "controller"):
        cfg = template("kl")
        if change == "architectures":
            cfg["architectures"] = ["sdadas/polish-distilroberta"]
        elif change == "bias":
            cfg["sentence_transformers"]["splade_decoder_bias_shift"] = "auto"
        else:
            cfg["sentence_transformers"]["regularizer_control"] = {"warmup_steps": 1000}
        with pytest.raises(ValueError):
            MODULE["bind_config"](cfg, Path("/checkpoint"), weights)


def test_worker_requires_approval_before_heavy_imports(monkeypatch):
    monkeypatch.delenv("P1_AB_APPROVED", raising=False)
    with pytest.raises(RuntimeError, match="approval"):
        MODULE["worker"](Path("missing.yaml"), "smoke", Path("missing.json"))


def test_comparison_reports_quality_and_each_index_cost():
    costs = {"doc_active_mean": 400, "query_active_mean": 100, "postings_per_query": 500}
    base = {"tiny10": 60}
    arms = [{"tiny10": 60.2, "density": costs},
            {"tiny10": 60.8, "density": {k: v * 1.1 for k, v in costs.items()}}]
    result = MODULE["comparison"](base, arms)
    assert result["mnrl_minus_kl_tiny10"] == pytest.approx(.6)
    assert all(v == pytest.approx(1.1) for v in result["mnrl_over_kl_cost_ratios"].values())


def test_supervisor_binds_best_start_then_smoke_and_only_two_arms(monkeypatch, tmp_path):
    fn = MODULE["supervisor"]
    namespace = fn.__globals__
    exp = tmp_path / "experiment"
    for path in (tmp_path / "logs", tmp_path / "datasets/arms", exp / "results/data"):
        path.mkdir(parents=True)
    for name, value in (("ROOT", tmp_path), ("EXP", exp), ("STATE", tmp_path / "logs/ab.json"),
                        ("RESULT", exp / "results/data/ab.json")):
        monkeypatch.setitem(namespace, name, value)
    monkeypatch.setenv("P1_AB_APPROVED", "yes")
    data = tmp_path / "datasets/arms/data.jsonl"
    data.write_text("scored subset fixture")
    monkeypatch.setitem(namespace, "DATA", data)
    sha = MODULE["digest"]
    pool = {"rows": 3, "examples": 3, "sha256": "pool-sha"}
    side = {"rows": 3, "examples": 3, "input_pool_sha256": "pool-sha", "dense_scale": 97,
            "training_score_scale": 1., "ab_rows": 40000, "ab_sha256": sha(data)}
    (data.parent / "mlm-p1-kl-v2.jsonl.manifest.json").write_text(json.dumps(pool))
    (data.parent / "mlm-p1-kl-v2-tens400.jsonl.teacher.json").write_text(json.dumps(side))
    start = tmp_path / "selected"
    start.mkdir()
    (start / "model.safetensors").write_bytes(b"serialized weights fixture")
    (start / "trainer_state.json").write_text(json.dumps({"global_step": 20}))
    (start / "regularizer_control.json").write_text(json.dumps({"events": [
        {"event": "control", "step": 20, "document_weight": .0024, "query_weight": .00012}]}))
    costs = {"doc_active_mean": 400, "query_active_mean": 100, "postings_per_query": 500}
    primary = {"status": "completed", "epochs": [
        {"epoch": i, "tiny10": score, "label": f"ep{i}", "checkpoint": str(start),
         "source_checkpoint": str(start), "density": costs} for i, score in [(1, 59), (2, 62), (3, 61)]]}
    (tmp_path / "logs/mlm-p1-train-queue.json").write_text(json.dumps({"status": "completed"}))
    (exp / "results/data/mlm-p1-v2-training.json").write_text(json.dumps(primary))
    specs = {}
    for tag in ("smoke", "kl", "kl-mnrl"):
        path = tmp_path / "logs" / f"template-{tag}.yaml"
        path.write_text(yaml.safe_dump(template(tag)))
        specs[tag] = {"snapshot": str(path), "sha256": sha(path)}
    approval = tmp_path / "logs/approval.json"
    approval.write_text(json.dumps({"pool_sha256": "pool-sha", "data_sha256": sha(data), "templates": specs}))
    calls = []
    def run_arm(spec, tag, start_path, env, **options):
        cfg = yaml.safe_load(Path(spec["snapshot"]).read_text())
        assert cfg["architectures"] == [str(start)]
        assert cfg["hparams"][0]["document_regularizer_weight"] == .0024
        assert cfg["sentence_transformers"]["query_regularizer_weight"] == .00012
        calls.append(tag)
        return {"tag": tag, "final": str(tmp_path / tag / "final")}
    monkeypatch.setitem(namespace, "run_arm", run_arm)
    def measured(checkpoint, label, env, **options):
        return {"checkpoint": str(checkpoint), "label": label, "density": costs,
                "tiny10": 62.2 if label.endswith("-kl") else 63.1}
    monkeypatch.setitem(namespace, "measure_model", measured)
    monkeypatch.setitem(namespace, "publish_paths", lambda *args: {"pushed": False})
    monkeypatch.setattr(namespace["subprocess"], "check_output", lambda *args, **kwargs: "")
    heldout_calls = []
    def evaluate(args, **kwargs):
        assert args[5] == "heldout8"
        heldout_calls.append(args[3])
        output = exp / f"results-regimes/quantized/{args[3]}/heldout8.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"n_tasks": 8, "mean_ndcg@10": 42., "regime": "quantized",
                                      "max_seq_length": 512, "model": args[4]}))
    monkeypatch.setattr(namespace["subprocess"], "run", evaluate)
    fn(approval)
    assert calls == ["smoke", "kl", "kl-mnrl"]
    assert heldout_calls == ["mlm-p1-continue-ab-kl-mnrl"]
    result = json.loads((exp / "results/data/ab.json").read_text())
    assert result["start_provenance"]["epoch"] == 2
    assert result["selected_label"] == "mlm-p1-continue-ab-kl-mnrl"
    assert json.loads((tmp_path / "logs/ab.json").read_text())["status"] == "completed"
    # A restart must reuse the frozen binding instead of rejecting it or
    # changing the selected start/configuration.
    bound_path = tmp_path / "logs/mlm-p1-continue-ab-bound.json"
    original_binding = bound_path.read_bytes()
    calls.clear()
    fn(approval, resume=True)
    assert calls == ["smoke", "kl", "kl-mnrl"]
    assert bound_path.read_bytes() == original_binding
    frozen = tmp_path / "logs/mlm-p1-continue-ab-kl-bound.yaml"
    frozen.write_text(frozen.read_text() + "# changed after approval\n")
    calls.clear()
    with pytest.raises(ValueError, match="Frozen continuation configuration"):
        fn(approval, resume=True)
    assert calls == []
