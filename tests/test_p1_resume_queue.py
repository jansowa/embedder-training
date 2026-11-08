import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/polish-splade-2026-10/queues/queue_5070_p1_resume.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_resume_policy_overrides_epoch_policy_and_restored_save_interval():
    args = SimpleNamespace(save_strategy="epoch", save_steps=123, save_total_limit=1)
    state = SimpleNamespace(save_steps=123, global_step=21732)
    MODULE["apply_checkpoint_policy"](args, state)
    assert args.save_strategy == "steps"
    assert args.save_steps == state.save_steps == 500
    assert args.save_total_limit == 2
    assert state.global_step == 21732


def test_resumable_checkpoint_requires_optimizer_scheduler_and_rng(tmp_path):
    state = {"global_step": 21732, "max_steps": 32595, "epoch": 2}
    (tmp_path / "trainer_state.json").write_text(json.dumps(state))
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (tmp_path / name).write_bytes(b"fixture")
    assert MODULE["validate_checkpoint"](tmp_path, 21732) == state
    with pytest.raises(ValueError, match="step"):
        MODULE["validate_checkpoint"](tmp_path, 32595)
    (tmp_path / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        MODULE["validate_checkpoint"](tmp_path)


def test_tiny_cache_checks_checkpoint_regime_and_all_ten_tasks(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "tiny10.json"
    score = {"model": str(checkpoint), "regime": "quantized", "max_seq_length": 512,
             "n_tasks": 10, "mean_ndcg@10": 60.0}
    output.write_text(json.dumps(score))
    assert MODULE["tiny_result"](output, checkpoint) == score
    for key, bad in (("model", "another-checkpoint"), ("regime", "truncated"), ("max_seq_length", 256), ("n_tasks", 9), ("mean_ndcg@10", float("nan"))):
        output.write_text(json.dumps(dict(score, **{key: bad})))
        with pytest.raises(ValueError):
            MODULE["tiny_result"](output, checkpoint)


def test_real_cpu_trainer_resumes_and_retains_two_step_saves_plus_epochs(tmp_path):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from training.checkpoints import EpochCheckpointCallback

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, input_ids, labels=None):
            return {"loss": ((input_ids.float() * self.weight) ** 2).mean()}

    class StopAfterTwo(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.epoch >= 2:
                control.should_training_stop = True
            return control

    data = [{"input_ids": [1.0]} for _ in range(10)]
    args = TrainingArguments(output_dir=str(tmp_path), num_train_epochs=3, per_device_train_batch_size=1,
                             use_cpu=True, save_strategy="epoch", save_total_limit=1, report_to=[], disable_tqdm=True)
    trainer = Trainer(model=Model(), args=args, train_dataset=data,
                      callbacks=[EpochCheckpointCallback(tmp_path), StopAfterTwo()])
    trainer.train()
    checkpoint = tmp_path / "epoch-checkpoints/epoch-0002-step-20"
    assert (checkpoint / "optimizer.pt").is_file()

    class ResumePolicy(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            assert state.global_step == 20
            MODULE["apply_checkpoint_policy"](args, state)
            # Small CPU analogue of the 500-step production interval.
            args.save_steps = state.save_steps = 4
            return control

    MODULE["apply_checkpoint_policy"](args)
    args.save_steps = 4
    resumed = Trainer(model=Model(), args=args, train_dataset=data,
                      callbacks=[EpochCheckpointCallback(tmp_path), ResumePolicy()])
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 30
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == ["checkpoint-28", "checkpoint-30"]
    assert {p.name for p in (tmp_path / "epoch-checkpoints").iterdir()} == {
        "epoch-0001-step-10", "epoch-0002-step-20", "epoch-0003-step-30"}
