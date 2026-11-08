"""Evaluate saved P1 epochs, then resume with two rotating step checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from queue_5070_p1_train import ROOT, EXP, PYTHON, STATE, RESULT, SCORED, attach_checkpoint_observer, digest, now, save_json

CONFIG = ROOT / "logs/mlm-p1-v2-train-approved.yaml"
EVALUATION = ROOT / "logs/mlm-p1-pre-resume-tiny10.json"
APPROVAL = ROOT / "logs/mlm-p1-resume-approval.json"
RESUME_STATE = ROOT / "logs/mlm-p1-resume-queue.json"


def loadable_checkpoint(source: Path, epoch: int) -> Path:
    sys.path.insert(0, str(ROOT))
    from training.backends.splade_checkpoint_compat import compatible_export
    return compatible_export(source, ROOT / f"cache/mlm-p1-loadable-exports/epoch-{epoch}")


def locate_run() -> Path:
    paths = list((ROOT / "runs/arm-mlm-p1").glob("**/resolved_config.yaml"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one original P1 run")
    # Keep the original relative spelling for immutable metadata verification.
    return paths[0].parent.relative_to(ROOT)


def validate_checkpoint(path: Path, expected_step: int | None = None) -> dict:
    required = ("model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    if any(not (path / name).is_file() or not (path / name).stat().st_size for name in required):
        raise ValueError(f"Incomplete resumable checkpoint: {path}")
    state = json.loads((path / "trainer_state.json").read_text())
    if state.get("max_steps") != 32595 or (expected_step is not None and state.get("global_step") != expected_step):
        raise ValueError("Unexpected checkpoint step or training duration")
    return state


def status(value: str, **extra) -> None:
    record = {"status": value, "pid": os.getpid(), "updated_at": now(), "approved": True, **extra}
    save_json(RESUME_STATE, record)
    save_json(STATE, record)


def environment() -> dict:
    return dict(os.environ, P1_TRAINING_APPROVED="yes", WANDB_MODE="disabled", ORPHAN_GUARD_EXEMPT="1",
                PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", REGIMES_ROOT=str(EXP / "results-regimes"),
                RESULTS_ROOT=str(EXP / "results"))


def tiny_result(path: Path, checkpoint: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    from training.backends.splade_checkpoint_compat import same_model_source
    score = json.loads(path.read_text())
    if score.get("n_tasks") != 10 or not isinstance(score.get("mean_ndcg@10"), (int, float)) or not math.isfinite(score["mean_ndcg@10"]):
        raise ValueError("Incomplete tiny10 result")
    if not same_model_source(Path(score["model"]), checkpoint) or score.get("max_seq_length") != 512 or score.get("regime") != "quantized":
        raise ValueError("tiny10 result refers to a different checkpoint")
    return score


def evaluate_saved_epochs(run: Path, env: dict) -> None:
    results = {"status": "evaluating", "epochs": [], "started_at": now()}
    for epoch, step in ((1, 10866), (2, 21732)):
        checkpoint = run / "epoch-checkpoints" / f"epoch-{epoch:04d}-step-{step}"
        validate_checkpoint(checkpoint, step)
        source_checkpoint = checkpoint
        checkpoint = loadable_checkpoint(source_checkpoint, epoch)
        label = f"mlm-p1-ep{epoch}"
        output = EXP / f"results-regimes/quantized/{label}/tiny10.json"
        status("evaluating_saved_epochs", epoch=epoch, checkpoint=str(checkpoint))
        if not output.exists():
            subprocess.run(["bash", "scripts/pirb_regime_eval.sh", "quantized", label, str(checkpoint), "tiny10", "16", "512"],
                           cwd=ROOT, env=env, check=True)
        score = tiny_result(output, checkpoint)
        results["epochs"].append({"epoch": epoch, "checkpoint": str(checkpoint), "label": label,
                                  "source_checkpoint": str(source_checkpoint),
                                  "tiny10": score["mean_ndcg@10"], "measured_at": now()})
        save_json(EVALUATION, results)
        print(f"[P1] Epoch {epoch}: tiny10={score['mean_ndcg@10']:.4f}", flush=True)
    results.update(status="completed", finished_at=now())
    save_json(EVALUATION, results)


def apply_checkpoint_policy(args, state=None) -> None:
    from transformers.trainer_utils import IntervalStrategy
    args.save_strategy = IntervalStrategy.STEPS
    args.save_steps = 500
    args.save_total_limit = 2
    if state is not None:
        # TrainerState is restored from the old epoch-only checkpoint.
        state.save_steps = 500


def verify_approval() -> dict:
    approval = json.loads(APPROVAL.read_text())
    if approval["config_sha256"] != digest(CONFIG):
        raise ValueError("Original approved training configuration changed")
    if approval["checkpoint_policy"] != {"save_strategy": "steps", "save_steps": 500, "save_total_limit": 2, "keep_epoch_checkpoints": True}:
        raise ValueError("Unexpected recovery checkpoint policy")
    checkpoint = Path(approval["checkpoint"])
    validate_checkpoint(checkpoint, 21732)
    for name, expected in approval["checkpoint_sha256"].items():
        if digest(checkpoint / name) != expected:
            raise ValueError(f"Source checkpoint changed: {name}")
    return approval


def resume_worker() -> int:
    sys.path.insert(0, str(ROOT))
    from transformers import TrainerCallback
    from training.backends import sentence_transformers_backend as backend
    from training.backends.splade_checkpoint_compat import add_decoder_bias_alias, reload_sparse_model_on_cpu
    from sentence_transformers import SparseEncoderTrainer
    from training.train import main as train_main

    approval = verify_approval()
    run = Path(approval["run"])
    history = run / "regularizer_control.json"
    backup = ROOT / "logs/mlm-p1-controller-before-resume.json"
    if history.is_file() and not backup.exists():
        with backup.open("x") as handle:
            handle.write(history.read_text())
    base_args = backend._training_args

    def training_args(*args, **kwargs):
        result = base_args(*args, **kwargs)
        apply_checkpoint_policy(result)
        return result

    backend._training_args = training_args
    started = time.monotonic()
    def verified_reload(trainer, checkpoint_path):
        verification = reload_sparse_model_on_cpu(trainer, Path(checkpoint_path))
        save_json(run / "resume-weight-verification.json", dict(verification, checkpoint=checkpoint_path, time=now()))
        print(f"[P1] Verified every serialized MLM tensor after reload: {verification}", flush=True)

    SparseEncoderTrainer._load_from_checkpoint = verified_reload

    class ResumeMonitor(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step != 21732 or state.max_steps != 32595 or abs(state.epoch - 2) > 0.001:
                raise ValueError("Trainer did not restore the approved epoch-two state")
            apply_checkpoint_policy(args, state)
            save_json(run / "resume-checkpoint-policy.json", dict(approval, restored_at=now(), restored_step=state.global_step))
            print("[P1] Restored step 21732; save every 500 steps, retain two rotating checkpoints plus epoch copies", flush=True)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if any(not math.isfinite(float(logs[k])) for k in ("loss", "base_loss", "grad_norm") if k in logs):
                raise RuntimeError("Non-finite training metrics")
            if state.global_step <= 21832 and logs.get("base_loss", 0) > 100:
                raise RuntimeError("Unexpected KL spike immediately after checkpoint reload")
            row = {"step": state.global_step, "epoch": state.epoch, "time": now(), "metrics": logs,
                   "elapsed_seconds": round(time.monotonic() - started, 2)}
            with (run / "p1-resume-telemetry.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            save_json(ROOT / "logs/mlm-p1-resume-progress.json", row)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            # Prove the full save path promptly, then use the normal 500-step cadence.
            if state.global_step == 21733:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            add_decoder_bias_alias(checkpoint)
            validate_checkpoint(checkpoint, state.global_step)
            for target in (checkpoint, *list((run / "epoch-checkpoints").glob(f"epoch-*-step-{state.global_step}"))):
                add_decoder_bias_alias(target)
                for name in ("regularizer_control.json", "resume-checkpoint-policy.json"):
                    if (run / name).is_file():
                        shutil.copy2(run / name, target / name)
            recent = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if len(recent) > 2:
                raise RuntimeError("Rolling checkpoint rotation did not retain the requested two saves")
            for saved in recent:
                validate_checkpoint(saved)
            save_json(ROOT / "logs/mlm-p1-checkpoint-status.json", {
                "updated_at": now(), "step": state.global_step,
                "checkpoints": [str(p) for p in recent], "max_rolling_checkpoints": 2,
                "epoch_checkpoints": [str(p) for p in sorted((run / "epoch-checkpoints").glob("epoch-*-step-*"))]})
            print(f"[P1] Verified resumable checkpoint at step {state.global_step}; rotating saves: {len(recent)}", flush=True)
            return control

    attach_checkpoint_observer(ResumeMonitor)
    rc = train_main(["--config", str(CONFIG), "--resume-from-checkpoint", approval["checkpoint"]])
    if rc:
        return rc
    if not (run / "final/model.safetensors").is_file():
        raise RuntimeError("Final model was not saved")
    add_decoder_bias_alias(run / "final")
    paths = list((run / "epoch-checkpoints").glob("epoch-0003-step-32595"))
    if len(paths) != 1:
        raise RuntimeError("Third epoch checkpoint was not preserved")
    validate_checkpoint(paths[0], 32595)
    return 0


def finish_measurements(run: Path, env: dict) -> None:
    from queue_5070_p1_train import measure_model, record_epoch, render_report, publish_results
    evaluation = json.loads(EVALUATION.read_text())
    result = {"status": "evaluating", "epochs": [], "resumed": True,
              "checkpoint_policy": json.loads(APPROVAL.read_text())["checkpoint_policy"],
              "teacher": json.loads(Path(str(SCORED) + ".teacher.json").read_text()),
              "config_sha256": digest(CONFIG), "measurements_after_training": True,
              "pre_resume_tiny10": evaluation["epochs"]}
    for epoch, step in ((1, 10866), (2, 21732), (3, 32595)):
        checkpoint = run / "epoch-checkpoints" / f"epoch-{epoch:04d}-step-{step}"
        exported = loadable_checkpoint(checkpoint, epoch)
        record = measure_model(exported, f"mlm-p1-ep{epoch}", env, reuse_tiny=True)
        record["source_checkpoint"] = str(checkpoint)
        record["epoch"] = epoch
        record_epoch(result, record)
        save_json(RESULT, result)
        render_report(result)
    best = max(result["epochs"], key=lambda r: r["tiny10"])
    subprocess.run(["bash", "scripts/pirb_regime_eval.sh", "quantized", best["label"], best["checkpoint"], "heldout8", "16", "512"],
                   cwd=ROOT, env=env, check=True)
    heldout = json.loads((EXP / f"results-regimes/quantized/{best['label']}/heldout8.json").read_text())
    if heldout.get("n_tasks") != 8:
        raise ValueError("Incomplete heldout8")
    best["heldout8"] = heldout["mean_ndcg@10"]
    result.update(status="completed", finished_at=now(), best_checkpoint=best["checkpoint"],
                  best_within_density_band=330 <= best["density"]["doc_active_mean"] <= 460)
    save_json(RESULT, result)
    render_report(result)
    try:
        publication = publish_results(result)
    except Exception as exc:
        publication = {"pushed": False, "error": str(exc)}
    status("completed", best_checkpoint=best["checkpoint"], publication=publication)


def continue_training(run: Path, env: dict) -> None:
    approval = verify_approval()
    if Path(approval["run"]) != run:
        raise ValueError("Run differs from the approved resume")
    evaluation = json.loads(EVALUATION.read_text())
    if evaluation.get("status") != "completed" or [e["epoch"] for e in evaluation["epochs"]] != [1, 2]:
        raise ValueError("Both tiny10 evaluations must finish before training resumes")
    for row in evaluation["epochs"]:
        tiny_result(EXP / f"results-regimes/quantized/{row['label']}/tiny10.json", Path(row["checkpoint"]))
    if shutil.disk_usage(run).free < 20_000_000_000 or shutil.disk_usage("/mnt/c").free < 10_000_000_000:
        raise RuntimeError("Insufficient disk reserve")
    status("training_resumed", checkpoint=approval["checkpoint"], checkpoint_policy=approval["checkpoint_policy"])
    with (ROOT / "logs/mlm-p1-v2-resume-cpu-load-train.log").open("x") as log:
        process = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--worker"], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        def terminate(sig, frame):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            raise SystemExit(128 + sig)
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        while process.poll() is None:
            if shutil.disk_usage(run).free < 10_000_000_000 or shutil.disk_usage("/mnt/c").free < 10_000_000_000:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
                raise RuntimeError("Disk reserve below 10 GB")
            time.sleep(15)
        if process.returncode:
            raise RuntimeError(f"Resumed training failed: exit {process.returncode}; see resume log")
    status("evaluating_epochs")
    finish_measurements(run, env)


def main() -> None:
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume-after-evaluation", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if os.environ.get("P1_TRAINING_APPROVED") != "yes":
        raise RuntimeError("P1 approval required")
    os.chdir(ROOT)
    if args.worker:
        raise SystemExit(resume_worker())
    with (ROOT / "logs/mlm-p1-resume-queue.lock").open("a") as own:
        # An already-running --evaluate-only process owns this until both tiny10 runs finish.
        fcntl.flock(own, fcntl.LOCK_EX)
        try:
            with (ROOT / "logs/gpu_5070.lock").open("a") as gpu:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
                    raise RuntimeError("GPU already has a compute process")
                run = locate_run()
                if not args.resume_after_evaluation:
                    evaluate_saved_epochs(run, environment())
                if args.evaluate_only:
                    status("saved_epochs_evaluated")
                else:
                    continue_training(run, environment())
        except Exception as exc:
            status("failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
