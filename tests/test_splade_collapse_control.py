import json

from scripts.run_splade_collapse_control import save_control_state, update_zero_streak


def test_zero_gradient_stop_streak_requires_consecutive_measurements():
    assert update_zero_streak(2, 0.0) == 3
    assert update_zero_streak(2, 0.001) == 0
    assert update_zero_streak(2, None) == 0


def test_save_real_transformers_trainer_state(tmp_path):
    from transformers.trainer_callback import TrainerState

    state = TrainerState(global_step=500, log_history=[{"step": 500, "loss": 2.1}])
    save_control_state(state, str(tmp_path))
    saved = json.loads((tmp_path / "control_train_state.json").read_text())
    assert saved["global_step"] == 500
    assert saved["log_history"] == state.log_history
