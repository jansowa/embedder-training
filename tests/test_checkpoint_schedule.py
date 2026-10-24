import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.checkpoints import (  # noqa: E402
    CheckpointError,
    FractionCheckpointCallback,
    build_fraction_checkpoint_callback,
    fraction_checkpoint_name,
)


class _TrainingArgumentsStub:
    """Stand-in for SentenceTransformerTrainingArguments in config-flow tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def _write_complete_checkpoint(path, *, step):
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    (path / "model.safetensors").write_bytes(f"weights-{step}".encode())


def _training_args(config):
    from training.backends.sentence_transformers_backend import _backend_config, _training_args

    return _training_args(Path("out"), _backend_config(config), _TrainingArgumentsStub)


def test_warmup_ratio_from_the_backend_section_reaches_training_arguments():
    args = _training_args({"sentence_transformers": {"warmup_ratio": 0.1}})
    assert args.kwargs["warmup_ratio"] == pytest.approx(0.1)


def test_warmup_ratio_from_the_top_level_reaches_training_arguments():
    args = _training_args({"warmup_ratio": 0.25})
    assert args.kwargs["warmup_ratio"] == pytest.approx(0.25)


def test_backend_section_warmup_ratio_wins_over_the_top_level():
    args = _training_args({"warmup_ratio": 0.25, "sentence_transformers": {"warmup_ratio": 0.1}})
    assert args.kwargs["warmup_ratio"] == pytest.approx(0.1)


def test_warmup_ratio_defaults_to_zero():
    assert _training_args({}).kwargs["warmup_ratio"] == pytest.approx(0.0)


def test_lr_scheduler_type_defaults_to_linear():
    assert _training_args({}).kwargs["lr_scheduler_type"] == "linear"


def test_lr_scheduler_type_can_be_configured():
    args = _training_args({"sentence_transformers": {"lr_scheduler_type": "cosine"}})
    assert args.kwargs["lr_scheduler_type"] == "cosine"


def test_fraction_checkpoint_name_is_zero_padded():
    assert fraction_checkpoint_name(0.8) == "frac-0.80"
    assert fraction_checkpoint_name(0.5) == "frac-0.50"
    assert fraction_checkpoint_name(1.0) == "frac-1.00"


def test_fraction_checkpoint_callback_is_optional(tmp_path):
    assert build_fraction_checkpoint_callback(tmp_path, {}, {}) is None


def test_fraction_checkpoint_callback_reads_the_backend_section(tmp_path):
    callback = build_fraction_checkpoint_callback(
        tmp_path, {"save_at_fractions": [0.5]}, {"save_at_fractions": [0.8]}
    )
    assert callback is not None
    assert callback.fractions == (0.8,)


@pytest.mark.parametrize("fractions", [[0], [-0.1], [1.5], ["most"], [True]])
def test_invalid_fractions_are_rejected(tmp_path, fractions):
    with pytest.raises(CheckpointError, match="save_at_fractions"):
        build_fraction_checkpoint_callback(tmp_path, {}, {"save_at_fractions": fractions})


def test_fraction_checkpoint_is_preserved_at_the_target_step(tmp_path):
    output_dir = tmp_path / "out"
    callback = FractionCheckpointCallback(output_dir, [0.8])
    control = SimpleNamespace(should_save=False)

    callback.on_train_begin(None, SimpleNamespace(max_steps=100, global_step=0), control)
    callback.on_step_end(None, SimpleNamespace(global_step=79), control)
    assert control.should_save is False

    callback.on_step_end(None, SimpleNamespace(global_step=80), control)
    assert control.should_save is True

    _write_complete_checkpoint(output_dir / "checkpoint-80", step=80)
    callback.on_save(None, SimpleNamespace(global_step=80), control)

    preserved = output_dir / "frac-0.80"
    assert json.loads((preserved / "trainer_state.json").read_text(encoding="utf-8"))["global_step"] == 80
    assert not (output_dir / ".frac-0.80.tmp").exists()


def test_fraction_target_step_is_rounded_within_the_run(tmp_path):
    callback = FractionCheckpointCallback(tmp_path, [0.8])
    callback.on_train_begin(None, SimpleNamespace(max_steps=7), SimpleNamespace(should_save=False))
    # 0.8 * 7 = 5.6, which rounds to the sixth of seven optimizer steps.
    assert sorted(callback._targets) == [6]


def test_full_fraction_targets_the_last_step(tmp_path):
    callback = FractionCheckpointCallback(tmp_path, [1.0])
    callback.on_train_begin(None, SimpleNamespace(max_steps=5), SimpleNamespace(should_save=False))
    assert sorted(callback._targets) == [5]


def test_fractions_sharing_a_step_each_get_a_directory(tmp_path):
    output_dir = tmp_path / "out"
    callback = FractionCheckpointCallback(output_dir, [0.8, 0.81])
    control = SimpleNamespace(should_save=False)

    callback.on_train_begin(None, SimpleNamespace(max_steps=10), control)
    callback.on_step_end(None, SimpleNamespace(global_step=8), control)
    _write_complete_checkpoint(output_dir / "checkpoint-8", step=8)
    callback.on_save(None, SimpleNamespace(global_step=8), control)

    assert (output_dir / "frac-0.80" / "model.safetensors").exists()
    assert (output_dir / "frac-0.81" / "model.safetensors").exists()


def test_other_steps_do_not_preserve_a_fraction_checkpoint(tmp_path):
    output_dir = tmp_path / "out"
    callback = FractionCheckpointCallback(output_dir, [0.8])
    control = SimpleNamespace(should_save=False)

    callback.on_train_begin(None, SimpleNamespace(max_steps=100), control)
    _write_complete_checkpoint(output_dir / "checkpoint-50", step=50)
    callback.on_step_end(None, SimpleNamespace(global_step=50), control)
    callback.on_save(None, SimpleNamespace(global_step=50), control)

    assert control.should_save is False
    assert not (output_dir / "frac-0.80").exists()


def test_existing_fraction_checkpoint_is_not_overwritten(tmp_path):
    output_dir = tmp_path / "out"
    _write_complete_checkpoint(output_dir / "frac-0.80", step=80)
    (output_dir / "frac-0.80" / "model.safetensors").write_bytes(b"original")
    _write_complete_checkpoint(output_dir / "checkpoint-80", step=80)

    callback = FractionCheckpointCallback(output_dir, [0.8])
    control = SimpleNamespace(should_save=False)
    callback.on_train_begin(None, SimpleNamespace(max_steps=100), control)
    callback.on_step_end(None, SimpleNamespace(global_step=80), control)
    callback.on_save(None, SimpleNamespace(global_step=80), control)

    assert (output_dir / "frac-0.80" / "model.safetensors").read_bytes() == b"original"


def test_unknown_total_steps_disables_fraction_checkpoints(tmp_path, capsys):
    callback = FractionCheckpointCallback(tmp_path, [0.8])
    control = SimpleNamespace(should_save=False)

    callback.on_train_begin(None, SimpleNamespace(max_steps=0), control)
    callback.on_step_end(None, SimpleNamespace(global_step=1), control)

    assert callback._targets == {}
    assert control.should_save is False
    assert "save_at_fractions" in capsys.readouterr().out


def test_sentence_transformers_registers_the_fraction_callback(tmp_path):
    from training.backends.sentence_transformers_backend import _add_checkpoint_callbacks, _backend_config

    class _Trainer:
        def __init__(self):
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

    config = {"sentence_transformers": {"save_at_fractions": [0.8]}}
    trainer = _Trainer()
    _add_checkpoint_callbacks(trainer, tmp_path, config, _backend_config(config))

    assert [type(callback).__name__ for callback in trainer.callbacks] == ["FractionCheckpointCallback"]


def test_distributed_backend_registers_the_fraction_callback(tmp_path):
    from training.backends.sentence_transformers_backend import _backend_config
    from training.backends.sentence_transformers_ddp_backend import _add_checkpoint_callbacks

    class _Trainer:
        def __init__(self):
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

    config = {"sentence_transformers": {"save_at_fractions": [0.8]}}
    trainer = _Trainer()
    _add_checkpoint_callbacks(trainer, tmp_path, config, _backend_config(config))

    assert [type(callback).__name__ for callback in trainer.callbacks] == ["FractionCheckpointCallback"]
