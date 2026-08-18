from pathlib import Path
from types import SimpleNamespace


def test_epoch_checkpoint_request_is_identical_on_non_main_rank(tmp_path, monkeypatch):
    import training.checkpoints as checkpoints
    from training.backends.sentence_transformers_ddp_backend import DistributedEpochCheckpointCallback

    monkeypatch.setattr(checkpoints, "is_main_process", lambda: False)
    callback = DistributedEpochCheckpointCallback(tmp_path)
    state = SimpleNamespace(global_step=2, epoch=1.0)
    control = SimpleNamespace(should_save=False)

    returned = callback.on_epoch_end(None, state, control)

    assert returned is control
    assert control.should_save is True
    assert callback._pending_epoch_checkpoint == (1, 2)


def test_checkpoint_callbacks_are_registered_on_every_rank(tmp_path, monkeypatch):
    import training.backends.sentence_transformers_ddp_backend as backend
    import training.checkpoints as checkpoints

    monkeypatch.setattr(checkpoints, "is_main_process", lambda: False)

    class FakeTrainer:
        def __init__(self):
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

    trainer = FakeTrainer()
    config = {
        "keep_epoch_checkpoints": True,
        "keep_step_checkpoints": [5],
    }

    backend._add_checkpoint_callbacks(trainer, Path(tmp_path), config, {})

    assert len(trainer.callbacks) == 2
    assert isinstance(trainer.callbacks[0], backend.DistributedEpochCheckpointCallback)
    assert isinstance(trainer.callbacks[1], checkpoints.StepCheckpointCallback)


def test_sentence_transformers_registry_uses_ddp_safe_backend():
    from training.backends.registry import BACKENDS

    assert BACKENDS["sentence-transformers"].module == "training.backends.sentence_transformers_ddp_backend"
