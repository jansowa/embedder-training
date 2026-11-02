from types import SimpleNamespace

import pytest

from training.backends.sentence_transformers_backend import (
    _backend_config,
    _configure_splade_regularizer_scheduler,
)


class SpladeRegularizerWeightSchedulerCallback:
    pass


class _Trainer:
    def __init__(self, loss):
        self.loss = loss
        self.other = object()
        self.callback_handler = SimpleNamespace(
            callbacks=[self.other, SpladeRegularizerWeightSchedulerCallback()]
        )

    def remove_callback(self, callback_type):
        self.callback_handler.callbacks = [
            callback for callback in self.callback_handler.callbacks
            if not isinstance(callback, callback_type)
        ]


@pytest.mark.parametrize('loss_kind', ['SpladeLoss', 'CachedSpladeLoss'])
def test_fixed_weights_remove_only_automatic_scheduler(loss_kind):
    loss = type(loss_kind, (), {})()
    loss.document_regularizer_weight = 0.003
    loss.query_regularizer_weight = 0.00015
    trainer = _Trainer(loss)
    _configure_splade_regularizer_scheduler(trainer, {'splade_regularizer_scheduler': False})
    assert trainer.callback_handler.callbacks == [trainer.other]
    assert loss.document_regularizer_weight == 0.003
    assert loss.query_regularizer_weight == 0.00015


def test_default_keeps_automatic_scheduler():
    trainer = _Trainer(SimpleNamespace())
    original = list(trainer.callback_handler.callbacks)
    _configure_splade_regularizer_scheduler(trainer, {})
    assert trainer.callback_handler.callbacks == original


def test_switch_can_be_set_at_top_level():
    assert _backend_config({'splade_regularizer_scheduler': False})['splade_regularizer_scheduler'] is False


def test_activation_metrics_reach_the_already_copied_trainer_history():
    from training.backends.sentence_transformers_backend import SpladeActivationStatsCallback

    callback = SpladeActivationStatsCallback(
        model=object(), query_texts=[], document_texts=[], batch_size=1,
        interval_steps=250, quantization_factor=100, log_on_train_begin=False,
        log_on_train_end=False, prefix='train/splade_activation_stats',
    )
    metrics = {'train/splade_activation_stats/document_active_dims_mean': 375.0}
    callback._log = lambda step: metrics
    state = SimpleNamespace(global_step=250, log_history=[{'step': 250, 'loss': 0.8}])
    logs = {'loss': 0.8}
    callback.on_log(None, state, object(), logs=logs)
    assert state.log_history[-1]['train/splade_activation_stats/document_active_dims_mean'] == 375.0
    assert logs['train/splade_activation_stats/document_active_dims_mean'] == 375.0
