"""Lambda warm-up, density control and L0 thresholds of the SPLADE regularizer (CPU, no torch, no ST)."""

from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from training.splade_regularizer_control import (
    DensityControlConfig,
    RegularizerControlCallback,
    RegularizerControlConfig,
    RegularizerControlError,
    build_regularizer_control_callback,
    install_regularizer_control_callback,
    next_document_weight,
    parse_regularizer_control,
    quadratic_warmup_weight,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = REPO_ROOT / "experiments/polish-splade-2026-10/configs/next/5070/mlm_probe_a.yaml"
BAND = (0.75, 1.05)
BOUNDS = (5e-4, 3e-2)


# -- warm-up and rule ------------------------------------------------------------------------------


def test_quadratic_warmup_weight():
    assert quadratic_warmup_weight(0, 1000, 0.003) == 0
    assert quadratic_warmup_weight(500, 1000, 0.003) == pytest.approx(0.00075)
    assert quadratic_warmup_weight(1000, 1000, 0.003) == 0.003
    assert quadratic_warmup_weight(5000, 1000, 0.003) == 0.003
    assert quadratic_warmup_weight(7, 0, 0.003) == 0.003
    assert quadratic_warmup_weight(-5, 1000, 0.003) == 0


def test_next_document_weight_moves_outside_the_band_only():
    assert next_document_weight(0.003, 500, 400, BAND, 1.25, BOUNDS) == pytest.approx(0.00375)
    assert next_document_weight(0.003, 250, 400, BAND, 1.25, BOUNDS) == pytest.approx(0.0024)
    assert next_document_weight(0.003, 400, 400, BAND, 1.25, BOUNDS) == 0.003
    # The band edges themselves are inside the band.
    assert next_document_weight(0.003, 300, 400, BAND, 1.25, BOUNDS) == 0.003
    assert next_document_weight(0.003, 420, 400, BAND, 1.25, BOUNDS) == 0.003


def test_next_document_weight_clamps():
    assert next_document_weight(0.025, 500, 400, BAND, 1.25, BOUNDS) == pytest.approx(0.03)
    assert next_document_weight(0.0006, 250, 400, BAND, 1.25, BOUNDS) == pytest.approx(0.0005)


def test_next_document_weight_rejects_a_zero_reference():
    with pytest.raises(RegularizerControlError):
        next_document_weight(0.003, 500, 0, BAND, 1.25, BOUNDS)


# -- parser ----------------------------------------------------------------------------------------


def _probe_backend_config() -> dict:
    from training.backends.sentence_transformers_backend import _backend_config
    from training.config_grid import expand_config_grid

    config = yaml.safe_load(PROBE_CONFIG.read_text(encoding="utf-8"))
    runs = expand_config_grid(config, backend="sentence-transformers", training_type="splade")
    assert len(runs) == 1
    return _backend_config(runs[0])


def test_parser_ignores_absent_or_disabled_control():
    assert parse_regularizer_control({}) is None
    assert parse_regularizer_control({"regularizer_control": None}) is None
    assert parse_regularizer_control({"regularizer_control": False}) is None


def test_parser_reads_the_probe_config_from_the_repo():
    backend_config = _probe_backend_config()
    config = parse_regularizer_control(backend_config)
    assert config == RegularizerControlConfig(
        warmup_steps=1000,
        density=DensityControlConfig(
            reference_model="models/base-splade-wrapped",
            reference_dims=None,
            band=(0.75, 1.05),
            interval_steps=250,
            factor=1.25,
            min_document_weight=0.0005,
            max_document_weight=0.03,
            query_ratio=0.05,
            sample_size=256,
            batch_size=32,
            quantization_factor=100,
            max_seq_length=None,
        ),
    )


def test_probe_config_builds_the_same_loss_kwargs_as_before():
    from training.backends.sentence_transformers_backend import _splade_loss_kwargs

    assert _splade_loss_kwargs(_probe_backend_config()) == {
        "document_regularizer_weight": 0.003,
        "query_regularizer_weight": 0.00015,
    }


def test_parser_fills_defaults():
    config = parse_regularizer_control({"regularizer_control": {"density": {"reference_dims": 400}}})
    assert config.warmup_steps == 0
    assert config.density == DensityControlConfig(
        reference_model=None,
        reference_dims=400.0,
        band=(0.75, 1.05),
        interval_steps=250,
        factor=1.25,
        min_document_weight=5e-4,
        max_document_weight=3e-2,
        query_ratio=0.05,
        sample_size=256,
        batch_size=32,
        quantization_factor=100,
        max_seq_length=None,
    )
    warmup_only = parse_regularizer_control({"regularizer_control": {"warmup_steps": 20}})
    assert warmup_only == RegularizerControlConfig(warmup_steps=20, density=None)
    no_query = parse_regularizer_control(
        {"regularizer_control": {"density": {"reference_dims": "400", "query_ratio": None, "min_document_weight": "5e-4"}}}
    )
    assert no_query.density.query_ratio is None
    assert no_query.density.min_document_weight == 5e-4


@pytest.mark.parametrize(
    "raw",
    [
        {"density": {"reference_dims": 400, "band": [1.05, 0.75]}},
        {"density": {"reference_dims": 400, "band": [0, 1.05]}},
        {"density": {"reference_dims": 400, "band": [0.75]}},
        {"density": {"reference_dims": 400, "factor": 1.0}},
        {"density": {"reference_dims": 400, "min_document_weight": 0.01, "max_document_weight": 0.001}},
        {"density": {"reference_dims": 400, "min_document_weight": -0.001}},
        {"density": {"reference_model": "models/base-splade-wrapped", "reference_dims": 400}},
        {"density": {"band": [0.75, 1.05]}},
        {"density": {"reference_dims": 0}},
        {"density": {"reference_dims": 400, "interval_steps": 0}},
        {"density": {"reference_dims": 400, "sample_size": -1}},
        {"density": {"reference_dims": 400, "query_ratio": 0}},
        {"density": {"reference_dims": 400, "intervalsteps": 250}},
        {"density": "yes"},
        {"warmup_step": 1000},
        {"warmup_steps": -1},
        {"warmup_steps": 1.5},
        {"warmup_steps": True},
    ],
)
def test_parser_rejects_bad_values(raw):
    with pytest.raises(RegularizerControlError):
        parse_regularizer_control({"regularizer_control": raw})


def test_parser_rejects_a_non_mapping():
    with pytest.raises(RegularizerControlError):
        parse_regularizer_control({"regularizer_control": [1000]})


# -- callback --------------------------------------------------------------------------------------


def _config(warmup_steps=1000, interval_steps=250, query_ratio=0.05, reference_model="models/base-splade-wrapped"):
    return RegularizerControlConfig(
        warmup_steps=warmup_steps,
        density=DensityControlConfig(
            reference_model=reference_model,
            reference_dims=None if reference_model else 400.0,
            band=BAND,
            interval_steps=interval_steps,
            factor=1.25,
            min_document_weight=BOUNDS[0],
            max_document_weight=BOUNDS[1],
            query_ratio=query_ratio,
            sample_size=256,
            batch_size=32,
            quantization_factor=100,
            max_seq_length=None,
        ),
    )


def _state(step, log_history=None):
    return SimpleNamespace(global_step=step, log_history=[] if log_history is None else log_history)


def _callback(densities, config=None, loss=None, reference=400.0):
    loss = loss or SimpleNamespace(document_regularizer_weight=0.003, query_regularizer_weight=0.00015)
    density_values = iter(densities)
    reference_calls = []

    def reference_fn():
        reference_calls.append(True)
        return reference

    callback = RegularizerControlCallback(
        loss,
        object(),
        [],
        config or _config(),
        target_document_weight=0.003,
        target_query_weight=0.00015,
        density_fn=lambda: next(density_values),
        reference_fn=reference_fn,
        main_process=False,
    )
    return callback, loss, reference_calls


def test_callback_warms_up_and_controls_density():
    callback, loss, reference_calls = _callback([500, 500, 290])

    assert callback.on_train_begin(None, _state(0), "control") == "control"
    assert reference_calls == [True]
    assert callback.reference_dims == 400.0
    assert loss.document_regularizer_weight == 0
    assert loss.query_regularizer_weight == 0

    callback.on_step_begin(None, _state(500), "control")
    assert loss.document_regularizer_weight == pytest.approx(0.00075)
    assert loss.query_regularizer_weight == pytest.approx(0.0000375)

    callback.on_step_begin(None, _state(1000), "control")
    assert loss.document_regularizer_weight == 0.003
    assert loss.query_regularizer_weight == 0.00015

    callback.on_step_end(None, _state(1250), "control")
    assert callback.document_weight == pytest.approx(0.00375)
    assert callback.query_weight == pytest.approx(0.0001875)
    assert loss.document_regularizer_weight == pytest.approx(0.00375)
    assert loss.query_regularizer_weight == pytest.approx(0.0001875)
    callback.on_step_begin(None, _state(1250), "control")
    assert loss.document_regularizer_weight == pytest.approx(0.00375)

    callback.on_step_end(None, _state(1300), "control")  # not a multiple of interval_steps
    assert callback.document_weight == pytest.approx(0.00375)
    assert callback.last_control_step == 1250

    callback.on_step_end(None, _state(1500), "control")
    assert callback.document_weight == pytest.approx(0.0046875)

    callback.on_step_end(None, _state(1500), "control")  # the same step is controlled once
    assert callback.document_weight == pytest.approx(0.0046875)

    callback.on_step_end(None, _state(1750), "control")  # 290 < 0.75 * 400
    assert callback.document_weight == pytest.approx(0.00375)
    assert callback.query_weight == pytest.approx(0.0001875)
    assert callback.last_density == 290

    history_entry = {"loss": 1.0, "step": 1750}
    state = _state(1750, [history_entry])
    logs = {"loss": 1.0}
    callback.on_log(None, state, "control", logs=logs)
    expected = {
        "regularizer/document_weight": pytest.approx(0.00375),
        "regularizer/query_weight": pytest.approx(0.0001875),
        "regularizer/document_dims_mean": 290,
        "regularizer/document_dims_reference": 400.0,
        "regularizer/document_dims_ratio": pytest.approx(0.725),
    }
    assert logs == {"loss": 1.0, **expected}
    assert history_entry == {"loss": 1.0, "step": 1750, **expected}


def test_callback_does_not_control_during_warmup():
    callback, loss, _ = _callback([])
    callback.on_train_begin(None, _state(0), "control")
    callback.on_step_end(None, _state(750), "control")
    assert callback.last_control_step is None
    assert callback.document_weight == 0.003


def test_callback_logs_warmup_weights_without_density():
    loss = SimpleNamespace(document_regularizer_weight=0.003, query_regularizer_weight=0.00015)
    callback = RegularizerControlCallback(
        loss,
        object(),
        [],
        RegularizerControlConfig(warmup_steps=20, density=None),
        target_document_weight=0.003,
        target_query_weight=0.00015,
        main_process=False,
    )
    callback.on_train_begin(None, _state(0), "control")
    assert loss.document_regularizer_weight == 0
    callback.on_step_end(None, _state(250), "control")
    logs = {}
    callback.on_log(None, _state(10), "control", logs=logs)
    assert logs == {
        "regularizer/document_weight": pytest.approx(0.00075),
        "regularizer/query_weight": pytest.approx(0.0000375),
    }
    callback.on_step_begin(None, _state(20), "control")
    assert loss.document_regularizer_weight == 0.003
    assert callback.on_epoch_end(None, _state(20), "control") == "control"


def test_callback_keeps_a_disabled_query_regularizer_disabled():
    loss = SimpleNamespace(document_regularizer_weight=0.003, query_regularizer_weight=None)
    callback = RegularizerControlCallback(
        loss,
        object(),
        [],
        _config(warmup_steps=0, interval_steps=10),
        target_document_weight=0.003,
        target_query_weight=None,
        density_fn=lambda: 500,
        reference_fn=lambda: 400,
        main_process=False,
    )
    callback.on_train_begin(None, _state(0), "control")
    callback.on_step_end(None, _state(10), "control")
    assert loss.document_regularizer_weight == pytest.approx(0.00375)
    assert loss.query_regularizer_weight is None


def test_callback_uses_configured_reference_dims():
    callback, _, reference_calls = _callback([500], config=_config(reference_model=None))
    callback.on_train_begin(None, _state(0), "control")
    assert reference_calls == []
    assert callback.reference_dims == 400.0


def test_callback_writes_history_and_restores_it_on_resume(tmp_path):
    args = SimpleNamespace(output_dir=str(tmp_path))
    callback, _, _ = _callback([500, 500, 500])
    callback.main_process = True
    callback.on_train_begin(args, _state(0), "control")
    for step in (1000, 1250, 1500):
        callback.on_step_end(args, _state(step), "control")
    callback.on_train_end(args, _state(1500), "control")
    payload = json.loads((tmp_path / "regularizer_control.json").read_text(encoding="utf-8"))
    assert [event["event"] for event in payload["events"]] == [
        "train_begin",
        "warmup_end",
        "control",
        "control",
        "control",
        "train_end",
    ]
    assert payload["document_dims_reference"] == 400.0
    assert payload["events"][4]["document_weight"] == pytest.approx(0.003 * 1.25**3)

    # The run is resumed from checkpoint-1250: the control at 1500 belongs to the lost segment.
    resumed, loss, _ = _callback([])
    resumed.main_process = True
    resumed.on_train_begin(args, _state(1250), "control")
    assert resumed.document_weight == pytest.approx(0.003 * 1.25**2)
    assert resumed.query_weight == pytest.approx(0.05 * 0.003 * 1.25**2)
    assert resumed.last_control_step == 1250
    assert loss.document_regularizer_weight == pytest.approx(0.003 * 1.25**2)
    payload = json.loads((tmp_path / "regularizer_control.json").read_text(encoding="utf-8"))
    assert [event["step"] for event in payload["events"]] == [0, 1000, 1000, 1250, 1250]


def test_callback_rejects_a_loss_without_weights():
    with pytest.raises(RegularizerControlError, match="SimpleNamespace"):
        RegularizerControlCallback(
            SimpleNamespace(),
            object(),
            ["doc"],
            _config(),
            target_document_weight=0.003,
            target_query_weight=0.00015,
            main_process=False,
        )


def test_callback_needs_documents_for_density_control():
    with pytest.raises(RegularizerControlError):
        RegularizerControlCallback(
            SimpleNamespace(document_regularizer_weight=0.003, query_regularizer_weight=0.00015),
            object(),
            [],
            _config(),
            target_document_weight=0.003,
            target_query_weight=0.00015,
            main_process=False,
        )


def test_build_callback_samples_documents_and_reads_targets():
    rows = [
        {"anchor": f"q{idx}", "positive": f"p{idx}", "negative_1": f"n{idx}", "negative_2": ""}
        for idx in range(10)
    ]
    loss = SimpleNamespace(document_regularizer_weight=0.003, query_regularizer_weight=0.00015)
    backend_config = {
        "document_regularizer_weight": 0.003,
        "query_regularizer_weight": None,
        "regularizer_control": {"warmup_steps": 5, "density": {"reference_dims": 300, "sample_size": 4}},
    }
    callback = build_regularizer_control_callback(loss, object(), rows, backend_config)
    assert callback.document_texts == ["p0", "p6", "n3", "n9"]
    assert callback.target_document_weight == 0.003
    assert callback.target_query_weight is None
    assert build_regularizer_control_callback(loss, object(), rows, {}) is None


def test_install_replaces_the_builtin_weight_scheduler():
    class SpladeRegularizerWeightSchedulerCallback:
        pass

    class DefaultFlowCallback:
        pass

    class WandbCallback:
        pass

    callback, _, _ = _callback([])
    callbacks = [DefaultFlowCallback(), SpladeRegularizerWeightSchedulerCallback(), WandbCallback()]
    trainer = SimpleNamespace(callback_handler=SimpleNamespace(callbacks=callbacks))
    assert install_regularizer_control_callback(trainer, callback) == "replaced SpladeRegularizerWeightSchedulerCallback"
    assert callbacks[1] is callback
    assert len(callbacks) == 3

    callbacks = [DefaultFlowCallback(), WandbCallback()]
    trainer = SimpleNamespace(callback_handler=SimpleNamespace(callbacks=callbacks))
    assert install_regularizer_control_callback(trainer, callback) == "inserted"
    assert callbacks[1] is callback

    added = []
    trainer = SimpleNamespace(add_callback=added.append)
    assert install_regularizer_control_callback(trainer, callback) == "added"
    assert added == [callback]


# -- loss kwargs -----------------------------------------------------------------------------------


def test_splade_loss_kwargs_pass_thresholds_only_when_set():
    from training.backends.sentence_transformers_backend import _splade_loss_kwargs

    kwargs = _splade_loss_kwargs({"document_regularizer_weight": 0.003, "document_regularizer_threshold": 350})
    assert kwargs == {
        "document_regularizer_weight": 0.003,
        "query_regularizer_weight": 5e-5,
        "document_regularizer_threshold": 350,
    }
    assert "query_regularizer_threshold" not in kwargs
    assert _splade_loss_kwargs({"query_regularizer_threshold": 0})["query_regularizer_threshold"] == 0
    assert _splade_loss_kwargs({"document_regularizer_threshold": None, "query_regularizer_threshold": None}) == {
        "document_regularizer_weight": 3e-5,
        "query_regularizer_weight": 5e-5,
    }
    assert _splade_loss_kwargs({}) == {"document_regularizer_weight": 3e-5, "query_regularizer_weight": 5e-5}


@pytest.mark.parametrize("value", [-1, "x", 1.5, True])
def test_splade_loss_kwargs_reject_bad_thresholds(value):
    from training.backends.sentence_transformers_backend import SentenceTransformersConfigError, _splade_loss_kwargs

    with pytest.raises(SentenceTransformersConfigError):
        _splade_loss_kwargs({"document_regularizer_threshold": value})


def test_new_keys_work_from_the_top_level_config():
    from training.backends.sentence_transformers_backend import _backend_config

    backend_config = _backend_config(
        {"document_regularizer_threshold": 120, "query_regularizer_threshold": 30, "regularizer_control": {"warmup_steps": 5}}
    )
    assert backend_config["document_regularizer_threshold"] == 120
    assert backend_config["query_regularizer_threshold"] == 30
    assert parse_regularizer_control(backend_config).warmup_steps == 5


def test_module_does_not_import_torch_or_sentence_transformers(monkeypatch):
    heavy = ("torch", "sentence_transformers", "transformers")
    for module_name in list(sys.modules):
        if module_name == "training.splade_regularizer_control":
            monkeypatch.delitem(sys.modules, module_name)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in heavy:
            raise AssertionError(f"Unexpected heavy import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module("training.splade_regularizer_control")


# -- backend wiring --------------------------------------------------------------------------------


def _wiring_stack(calls: dict):
    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeMLMTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            pass

    class FakeSpladePooling:
        def __init__(self, pooling_strategy):
            pass

    class FakeSparseEncoder:
        def __init__(self, modules):
            pass

        def save_pretrained(self, output_path):
            pass

    class FakeArgs:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]

    class FakeSparseMarginMSELoss:
        def __init__(self, model):
            pass

    class FakeSparseRankingLoss:
        def __init__(self, model, scale, gather_across_devices):
            pass

    class FakeSpladeLoss:
        def __init__(self, model, loss, **kwargs):
            calls["loss_kwargs"] = kwargs
            self.document_regularizer_weight = kwargs["document_regularizer_weight"]
            self.query_regularizer_weight = kwargs["query_regularizer_weight"]
            calls["loss"] = self

    class SpladeRegularizerWeightSchedulerCallback:
        pass

    class DefaultFlowCallback:
        pass

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            self.args = args
            self.callback_handler = SimpleNamespace(
                callbacks=[DefaultFlowCallback(), SpladeRegularizerWeightSchedulerCallback()]
            )
            calls["trainer"] = self

        def add_callback(self, callback):
            self.callback_handler.callbacks.append(callback)

        def train(self, **kwargs):
            state = SimpleNamespace(global_step=0, log_history=[])
            weights = []
            controller = [cb for cb in self.callback_handler.callbacks if hasattr(cb, "weights_at")]
            for callback in controller:
                callback.on_train_begin(self.args, state, "control")
            for _ in range(3):
                for callback in controller:
                    callback.on_step_begin(self.args, state, "control")
                weights.append(calls["loss"].document_regularizer_weight)
                state.global_step += 1
            calls["weights"] = weights

    return (
        FakeDataset,
        FakeSparseEncoder,
        FakeTrainer,
        FakeArgs,
        FakeSparseMarginMSELoss,
        FakeSparseRankingLoss,
        FakeSpladeLoss,
        FakeMLMTransformer,
        FakeSpladePooling,
    )


def _patch_backend(monkeypatch, backend, calls: dict) -> None:
    monkeypatch.setattr(backend, "_load_sparse_sentence_transformers", lambda: _wiring_stack(calls))
    # The real builder imports transformers (and torch); these tests stay on fakes.
    monkeypatch.setattr(
        backend, "_build_splade_model", lambda *args, **kwargs: SimpleNamespace(save_pretrained=lambda path: None)
    )


def _wiring_request(tmp_path, **sentence_transformers):
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1"]}) + "\n",
        encoding="utf-8",
    )
    return TrainingRequest(
        backend="sentence-transformers",
        training_type="splade",
        config={
            "train_data": str(data_dir),
            "output_dir": str(tmp_path / "out"),
            "sentence_transformers": {"model_name_or_path": "tiny-mlm", **sentence_transformers},
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )


def test_splade_training_installs_the_controller_and_passes_thresholds(monkeypatch, tmp_path, capsys):
    from training.backends import sentence_transformers_backend

    calls: dict = {}
    _patch_backend(monkeypatch, sentence_transformers_backend, calls)
    request = _wiring_request(
        tmp_path,
        document_regularizer_weight=0.004,
        query_regularizer_weight=0.0002,
        document_regularizer_threshold=350,
        regularizer_control={"warmup_steps": 2},
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["loss_kwargs"] == {
        "document_regularizer_weight": 0.004,
        "query_regularizer_weight": 0.0002,
        "document_regularizer_threshold": 350,
    }
    callbacks = calls["trainer"].callback_handler.callbacks
    assert [type(cb).__name__ for cb in callbacks] == ["DefaultFlowCallback", "RegularizerControlCallback"]
    assert calls["weights"] == [0, pytest.approx(0.001), 0.004]
    assert "[INFO] Regularizer control (replaced SpladeRegularizerWeightSchedulerCallback): warmup_steps=2" in capsys.readouterr().out
    history = json.loads((tmp_path / "out" / "regularizer_control.json").read_text(encoding="utf-8"))
    assert history["target_document_weight"] == 0.004


def test_splade_training_without_control_keeps_the_builtin_scheduler(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend

    calls: dict = {}
    _patch_backend(monkeypatch, sentence_transformers_backend, calls)
    request = _wiring_request(tmp_path, document_regularizer_weight=0.004, regularizer_control=None)

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["loss_kwargs"] == {"document_regularizer_weight": 0.004, "query_regularizer_weight": 5e-5}
    callbacks = calls["trainer"].callback_handler.callbacks
    assert [type(cb).__name__ for cb in callbacks] == ["DefaultFlowCallback", "SpladeRegularizerWeightSchedulerCallback"]
    assert not (tmp_path / "out" / "regularizer_control.json").exists()


def test_splade_training_rejects_a_bad_control_config_before_loading_data(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend

    _patch_backend(monkeypatch, sentence_transformers_backend, {})
    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_training_rows",
        lambda *args, **kwargs: pytest.fail("data must not be loaded for an invalid config"),
    )
    request = _wiring_request(tmp_path, regularizer_control={"warmup_step": 1000})
    with pytest.raises(sentence_transformers_backend.SentenceTransformersConfigError, match="warmup_step"):
        sentence_transformers_backend.run_training(request)


def test_callback_records_the_end_of_the_warmup(tmp_path, capsys):
    callback, _, _ = _callback([500])
    callback.main_process = True
    args = SimpleNamespace(output_dir=str(tmp_path))
    callback.on_train_begin(args, _state(0), "control")
    callback.on_step_end(args, _state(1000), "control")
    out = capsys.readouterr().out
    assert out.index("warm-up finished at step 1000, document_weight=0.003") < out.index("0.003 -> 0.00375 (up)")
    events = json.loads((tmp_path / "regularizer_control.json").read_text(encoding="utf-8"))["events"]
    assert [event["event"] for event in events] == ["train_begin", "warmup_end", "control"]


def test_activation_stats_reach_the_log_history():
    from training.backends.sentence_transformers_backend import SpladeActivationStatsCallback

    callback = SpladeActivationStatsCallback(
        model=object(),
        query_texts=[],
        document_texts=["d"],
        batch_size=1,
        interval_steps=10,
        quantization_factor=100,
        log_on_train_begin=False,
        log_on_train_end=False,
        prefix="train/splade_activation_stats",
    )
    callback._compute_metrics = lambda: {"train/splade_activation_stats/document_active_dims_mean": 321.0}
    callback._log_to_wandb = lambda metrics, step: None
    entry = {"loss": 1.0, "step": 10}
    logs = {"loss": 1.0}
    callback.on_log(None, _state(10, [entry]), "control", logs=logs)
    assert entry["train/splade_activation_stats/document_active_dims_mean"] == 321.0
    assert logs["train/splade_activation_stats/document_active_dims_mean"] == 321.0
    stale = {"loss": 1.0, "step": 9}
    callback.on_log(None, _state(20, [stale]), "control", logs={})
    assert stale == {"loss": 1.0, "step": 9}
