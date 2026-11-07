"""Recover epoch evaluations for the already-running KL job without restarting it."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from queue_5070_p1_train import ROOT, EXP, STATE, RESULT, SCORED, digest, measure_model, now, publish_results, record_epoch, render_report, save_json

RECOVERY = ROOT / "logs/mlm-p1-measurement-recovery.json"


def validated_finished_run(run_root: Path):
    configs = list(run_root.glob("**/resolved_config.yaml"))
    if len(configs) != 1:
        raise ValueError("Cannot identify the single approved KL run")
    run = configs[0].parent
    if not (run / "final/model.safetensors").is_file():
        raise ValueError("Final trained model is missing")
    checkpoints = []
    for epoch in (1, 2, 3):
        paths = list((run / "epoch-checkpoints").glob(f"epoch-{epoch:04d}-step-*"))
        if len(paths) != 1:
            raise ValueError(f"Missing or ambiguous epoch {epoch}")
        path = paths[0]
        state = json.loads((path / "trainer_state.json").read_text())
        if not (path / "model.safetensors").is_file() or round(float(state.get("epoch", 0))) != epoch:
            raise ValueError(f"Invalid epoch {epoch} checkpoint")
        checkpoints.append((epoch, path, state))
    final = checkpoints[-1][2]
    if float(final.get("epoch", 0)) < 2.99 or final.get("max_steps", 0) <= 0 or final["global_step"] != final["max_steps"]:
        raise ValueError("All three training epochs have not finished")
    for row in final.get("log_history", []):
        for key in ("loss", "grad_norm"):
            if key in row and not math.isfinite(float(row[key])):
                raise ValueError("Non-finite metrics in the completed run")
    return run, checkpoints


def main() -> None:
    import fcntl
    if os.environ.get("P1_TRAINING_APPROVED") != "yes":
        raise RuntimeError("Training/evaluation approval required")
    os.chdir(ROOT)
    def status(value, **extra):
        save_json(RECOVERY, {"status": value, "updated_at": now(), "pid": os.getpid(),
                             "training_restart": False, **extra})
    with (ROOT / "logs/mlm-p1-measurement-recovery.lock").open("a") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                state = json.loads(STATE.read_text())
                if state["status"] == "completed":
                    status("not_needed")
                    return
                if state["status"] == "failed":
                    break
                status("waiting_training", kl_status=state["status"])
                time.sleep(15)
            log = ROOT / "logs/mlm-p1-v2-train.log"
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 32768))
                tail = handle.read().decode(errors="replace")
            if "Training returned without all three epoch checkpoints" not in tail:
                raise RuntimeError("KL failed for a reason other than the identified callback dispatch issue")
            run, checkpoints = validated_finished_run((ROOT / "cache/mlm-p1-v2-runs").resolve())
            status("evaluating", run=str(run))
            save_json(STATE, dict(state, status="evaluating_epochs", pid=os.getpid(), updated_at=now(), error=None))
            with (ROOT / "logs/gpu_5070.lock").open("a") as gpu:
                fcntl.flock(gpu, fcntl.LOCK_EX)
                while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
                    time.sleep(15)
                env = dict(os.environ, WANDB_MODE="disabled", ORPHAN_GUARD_EXEMPT="1", REGIMES_ROOT=str(EXP / "results-regimes"),
                           RESULTS_ROOT=str(EXP / "results"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
                approval = json.loads((ROOT / "logs/mlm-p1-v2-training-approval.json").read_text())
                result = {"status": "evaluating", "epochs": [], "teacher": json.loads(Path(str(SCORED) + ".teacher.json").read_text()),
                          "config_sha256": digest(Path(approval["config"])), "measurements_after_training": True,
                          "training_restart": False,
                          "recovery_reason": "DDP dispatcher replaced the experiment callback; all three native epoch checkpoints and the final model validated"}
                for epoch, path, _ in checkpoints:
                    record = measure_model(path, f"mlm-p1-ep{epoch}", env)
                    record["epoch"] = epoch
                    record_epoch(result, record)
                    save_json(RESULT, result)
                    render_report(result)
                best = max(result["epochs"], key=lambda row: row["tiny10"])
                subprocess.run(["bash", "scripts/pirb_regime_eval.sh", "quantized", best["label"], best["checkpoint"], "heldout8", "16", "512"],
                               cwd=ROOT, env=env, check=True)
                heldout = json.loads((EXP / f"results-regimes/quantized/{best['label']}/heldout8.json").read_text())
                if heldout.get("n_tasks") != 8:
                    raise ValueError("Incomplete recovered heldout8")
                best["heldout8"] = heldout["mean_ndcg@10"]
                result.update(status="completed", finished_at=now(), best_checkpoint=best["checkpoint"],
                              best_within_density_band=330 <= best["density"]["doc_active_mean"] <= 460)
                save_json(RESULT, result)
                render_report(result)
                try:
                    publication = publish_results(result)
                except Exception as exc:
                    publication = {"pushed": False, "error": str(exc)}
                save_json(STATE, dict(state, status="completed", error=None, pid=os.getpid(), updated_at=now(),
                                     measurement_recovered=True, best_checkpoint=best["checkpoint"], publication=publication))
                status("completed")
        except Exception as exc:
            status("failed", error=str(exc))
            state = json.loads(STATE.read_text())
            if state["status"] == "evaluating_epochs":
                save_json(STATE, dict(state, status="failed", updated_at=now(), error=str(exc)))
            raise


if __name__ == "__main__":
    main()
