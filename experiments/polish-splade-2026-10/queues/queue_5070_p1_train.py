"""Wait for the approved P1 scoring block, then train KL with epoch evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/polish-splade-2026-10"
CONFIG = EXP / "configs/next/5070/mlm_p1_kl_3ep.yaml"
POOL = ROOT / "datasets/arms/mlm-p1-kl-v2.jsonl"
SCORED = ROOT / "datasets/arms/mlm-p1-kl-v2-tens400.jsonl"
STATE = ROOT / "logs/mlm-p1-train-queue.json"
RESULT = EXP / "results/data/mlm-p1-v2-training.json"
PYTHON = ROOT / ".venv/bin/python"


def now() -> str:
    return datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(timespec="seconds")


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def g1_result(score: float, density: float) -> str:
    if not math.isfinite(score) or not math.isfinite(density):
        return "invalid"
    if not 330 <= density <= 460:
        return "density_outside_band"
    return "pass" if score >= 59.5 else "tiny10_below_59.5"


def record_epoch(result: dict, record: dict) -> None:
    if record["epoch"] != len(result["epochs"]) + 1:
        raise ValueError("Epoch measurements are not consecutive")
    result["epochs"].append(record)
    if record["epoch"] == 1:
        result["g1"] = g1_result(record["tiny10"], record["density"]["doc_active_mean"])
        result["g1_policy"] = "informational; continue all three epochs"


def validate_teacher(side: dict, pool: dict) -> None:
    if (side.get("rows"), side.get("examples"), side.get("input_pool_sha256")) != (pool["rows"], pool["examples"], pool["sha256"]):
        raise ValueError("Ensemble sidecar does not match the approved KL pool")
    scale = side.get("dense_scale")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid dense S in ensemble sidecar")
    if side.get("training_score_scale") != 1.0:
        raise ValueError("Ensemble scores must not be scaled a second time")


def scoring_alive() -> bool:
    try:
        pid = int((ROOT / "logs/mlm-p1-v2-scoring.pid").read_text())
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        return b"score_mlm_p1.py" in command
    except (OSError, ValueError):
        return False


def wait_for_scoring(pool: dict, status) -> dict:
    side_path = Path(str(SCORED) + ".teacher.json")
    while True:
        try:
            side = json.loads(side_path.read_text())
        except (OSError, ValueError):
            side = None
        if side is not None:
            validate_teacher(side, pool)
            status("verifying_scored_data", dense_scale=side["dense_scale"])
            if not SCORED.is_file() or digest(SCORED) != side["sha256"]:
                raise ValueError("Scored KL file checksum does not match its sidecar")
            return side
        if not scoring_alive():
            raise RuntimeError("Scoring exited before publishing a complete ensemble; training not started")
        status("waiting_scoring")
        time.sleep(30)


def prepare_run_root() -> Path:
    target = (ROOT / "cache/mlm-p1-v2-runs").resolve()
    parent = (ROOT / "cache").resolve()
    if not target.is_relative_to(parent) or target == parent:
        raise ValueError("Unexpected training output target")
    link = ROOT / "runs/arm-mlm-p1"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target:
            raise RuntimeError(f"Refusing to replace existing run path: {link}")
    else:
        target.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    target.mkdir(parents=True, exist_ok=True)
    if list(target.glob("**/resolved_config.yaml")):
        raise RuntimeError("A P1 training run already exists; refusing an implicit restart")
    if shutil.disk_usage(target).free < 20_000_000_000 or shutil.disk_usage("/mnt/c").free < 10_000_000_000:
        raise RuntimeError("Insufficient free space before training")
    return target


def render_report(result: dict) -> None:
    side = result.get("teacher", {})
    lines = ["# P1 v2: KL od MLM", "", f"Stan: {result.get('status')}; aktualizacja: {now()}", "",
             "347 687 zapytań, jeden pozytyw i 10 negatywów; start sdadas/polish-distilroberta, batch 8×4, LR 2e-5 stały po rozgrzewce, 3 epoki.", "",
             f"S PolDense: {side.get('dense_scale')}; zastosowane przed uśrednieniem nauczycieli. score_scale treningu: 1.0.", "",
             "| Model | tiny10 | heldout8 | wym./dok | wym./zapytanie | postings/zapytanie |",
             "|---|---:|---:|---:|---:|---:|", "| baza | 61.60 | 41.03 | — | — | — |",
             "| soup-v12 | 63.96 | 42.92 | — | — | — |", "| sondy MLM | 55.5–56.7 | — | — | — | — |"]
    for record in result.get("epochs", []):
        p = record["density"]
        heldout = record.get("heldout8", "—")
        lines.append(f"| epoka {record['epoch']} | {record['tiny10']:.2f} | {heldout} | {p['doc_active_mean']:.2f} | {p['query_active_mean']:.2f} | {p['postings_per_query']:.2f} |")
    lines += ["", f"G1 (informacyjnie): {result.get('g1', 'pending')} (tiny10 ≥59.5, gęstość dokumentów 330–460; nie zatrzymuje epok).", "",
              "Pomiary quantized przy 512 tokenach, pomiędzy epokami; dodatkowy profil min_weight=0.05 w plikach JSON.", "",
              "```json", json.dumps({k: v for k, v in result.items() if k != "epochs"}, indent=2), "```", ""]
    (EXP / "results/summary_5070_mlm_p1_v2.md").write_text("\n".join(lines), encoding="utf-8")


def result_paths(result: dict) -> list[Path]:
    paths = [RESULT, EXP / "results/summary_5070_mlm_p1_v2.md"]
    for record in result.get("epochs", []):
        number = int(record["epoch"])
        if number not in (1, 2, 3):
            raise ValueError("Unexpected epoch number in results")
        label = f"mlm-p1-ep{number}"
        paths.extend([EXP / f"results/sparsity/{label}-index-cost.json",
                      EXP / f"results-regimes/quantized/{label}/tiny10.json"])
        heldout = EXP / f"results-regimes/quantized/{label}/heldout8.json"
        if heldout.is_file():
            paths.append(heldout)
    return paths


def publish_paths(selected: list[Path], message: str, log_path: Path) -> dict:
    paths = [str(path.relative_to(ROOT)) for path in selected]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    with log_path.open("x") as log:
        subprocess.run(["git", "add", "--", *paths], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        changed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=ROOT, env=env).returncode
        if changed:
            subprocess.run(["git", "commit", "--only", "-m", message, "--", *paths],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        try:
            push = subprocess.run(["git", "-c", "credential.interactive=false", "-c", "push.recurseSubmodules=no", "push", "origin", "HEAD:master"],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            return {"commit": commit, "pushed": push.returncode == 0, "push_exit": push.returncode}
        except subprocess.TimeoutExpired:
            return {"commit": commit, "pushed": False, "push_error": "timeout"}


def publish_results(result: dict) -> dict:
    return publish_paths(result_paths(result), "Add the MLM P1 KL training results", ROOT / "logs/mlm-p1-v2-publish.log")


def measure_model(checkpoint: Path, label: str, env: dict) -> dict:
    started = time.monotonic()
    profile = EXP / "results/sparsity" / f"{label}-index-cost.json"
    profile.parent.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / f"{label}-index-cost.log"
    with log_path.open("x") as log:
        subprocess.run([str(PYTHON), "scripts/splade_index_cost.py", "--model", f"{label}={checkpoint}",
                        "--device", "cuda", "--min-weight", "0.05", "--output-json", str(profile)],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(["bash", "scripts/pirb_regime_eval.sh", "quantized", label, str(checkpoint), "tiny10", "16", "512"], cwd=ROOT, env=env, check=True)
    score = json.loads((EXP / f"results-regimes/quantized/{label}/tiny10.json").read_text())
    if score.get("n_tasks") != 10 or not isinstance(score.get("mean_ndcg@10"), (int, float)) or not math.isfinite(score["mean_ndcg@10"]):
        raise ValueError("Incomplete tiny10 result")
    p = json.loads(profile.read_text())["models"][label]
    if any(not math.isfinite(p[key]) for key in ("doc_active_mean", "query_active_mean", "postings_per_query")):
        raise ValueError("Invalid density profile")
    return {"checkpoint": str(checkpoint), "label": label,
            "tiny10": score["mean_ndcg@10"], "density": p, "measured_at": now(),
            "evaluation_seconds": round(time.monotonic() - started, 2)}


def worker(config: Path) -> int:
    if os.environ.get("P1_TRAINING_APPROVED") != "yes":
        raise RuntimeError("Training approval is required")
    # Heavy backend imports happen only in the approved training worker.
    sys.path.insert(0, str(ROOT))
    import torch
    from transformers import TrainerCallback
    from training.backends import sentence_transformers_backend as backend
    from training.train import main as train_main

    result = {"status": "training", "started_at": now(), "epochs": [],
              "teacher": json.loads(Path(str(SCORED) + ".teacher.json").read_text()),
              "config_sha256": digest(config),
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    started = time.monotonic()
    env = dict(os.environ, WANDB_MODE="disabled", ORPHAN_GUARD_EXEMPT="1")
    original = backend._add_checkpoint_callbacks

    class EpochMeasurements(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if any(not math.isfinite(float(logs[k])) for k in ("loss", "grad_norm") if k in logs):
                raise RuntimeError("Non-finite training loss or gradient norm")
            row = {"step": state.global_step, "epoch": state.epoch, "time": now(), "metrics": logs,
                   "elapsed_seconds": round(time.monotonic() - started, 2)}
            if torch.cuda.is_available():
                row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
            with (Path(args.output_dir) / "p1-telemetry.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            return control

        def on_save(self, args, state, control, **kwargs):
            epoch = float(state.epoch or 0)
            number = round(epoch)
            if number < 1 or abs(epoch - number) > 0.0001 or any(r["epoch"] == number for r in result["epochs"]):
                return control
            checkpoint = Path(args.output_dir) / "epoch-checkpoints" / f"epoch-{number:04d}-step-{state.global_step}"
            if not checkpoint.is_dir():
                raise RuntimeError(f"Preserved epoch checkpoint missing: {checkpoint}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[P1] Epoch {number} saved; evaluating before the next epoch", flush=True)
            record = measure_model(checkpoint, f"mlm-p1-ep{number}", env)
            record["epoch"] = number
            record_epoch(result, record)
            save_json(RESULT, result)
            render_report(result)
            return control

    def install(trainer, output_dir, cfg, backend_config):
        original(trainer, output_dir, cfg, backend_config)
        trainer.add_callback(EpochMeasurements())

    backend._add_checkpoint_callbacks = install
    rc = train_main(["--config", str(config)])
    if rc:
        return rc
    if len(result["epochs"]) != 3:
        raise RuntimeError("Training returned without all three epoch checkpoints")
    result["status"] = "completed"
    result["training_finished_at"] = now()
    save_json(RESULT, result)
    render_report(result)
    return 0


def supervisor(config: Path, approved_config_sha: str) -> None:
    import fcntl

    if os.environ.get("P1_TRAINING_APPROVED") != "yes":
        raise RuntimeError("Training approval is required")
    with (ROOT / "logs/mlm-p1-train-queue.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(value, **extra):
            record = {"status": value, "updated_at": now(), "pid": os.getpid(), "config": str(config),
                      "approved": True, "scope": "phase 1 KL, all three epochs; G1 informational; A/B handled by its own queue", **extra}
            save_json(STATE, record)
        try:
            pool = json.loads(Path(str(POOL) + ".manifest.json").read_text())
            side = wait_for_scoring(pool, status)
            if digest(config) != approved_config_sha:
                raise RuntimeError("Approved configuration changed while queued")
            with (ROOT / "logs/gpu_5070.lock").open("a") as gpu_lock:
                while True:
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        status("waiting_gpu_lock")
                        time.sleep(30)
                while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
                    status("waiting_gpu_idle")
                    time.sleep(30)
                run_root = prepare_run_root()
                status("training", dense_scale=side["dense_scale"], run_root=str(run_root))
                env = dict(os.environ, WANDB_MODE="disabled", PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1",
                           TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           REGIMES_ROOT=str(EXP / "results-regimes"), RESULTS_ROOT=str(EXP / "results"))
                with (ROOT / "logs/mlm-p1-v2-train.log").open("x") as log:
                    process = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--worker", "--config", str(config)],
                                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    def terminate(sig, frame):
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                        raise SystemExit(128 + sig)
                    signal.signal(signal.SIGTERM, terminate)
                    signal.signal(signal.SIGINT, terminate)
                    while process.poll() is None:
                        if shutil.disk_usage("/mnt/c").free < 10_000_000_000 or shutil.disk_usage(run_root).free < 10_000_000_000:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=60)
                            raise RuntimeError("Training stopped: disk reserve below 10 GB")
                        time.sleep(15)
                    if process.returncode:
                        raise RuntimeError(f"Training/evaluation failed with exit {process.returncode}; see logs/mlm-p1-v2-train.log")
                result = json.loads(RESULT.read_text())
                if result["epochs"]:
                    best = max(result["epochs"], key=lambda r: r["tiny10"])
                    result["best_within_density_band"] = 330 <= best["density"]["doc_active_mean"] <= 460
                    subprocess.run(["bash", "scripts/pirb_regime_eval.sh", "quantized", best["label"], best["checkpoint"], "heldout8", "16", "512"], cwd=ROOT, env=env, check=True)
                    heldout = json.loads((EXP / f"results-regimes/quantized/{best['label']}/heldout8.json").read_text())
                    if heldout.get("n_tasks") != 8:
                        raise ValueError("Incomplete heldout8 result")
                    best["heldout8"] = heldout["mean_ndcg@10"]
                    result["best_checkpoint"] = best["checkpoint"]
                result["finished_at"] = now()
                save_json(RESULT, result)
                render_report(result)
                try:
                    publication = publish_results(result)
                except Exception as exc:
                    publication = {"pushed": False, "error": str(exc)}
                status(result["status"], best_checkpoint=result.get("best_checkpoint"), publication=publication)
        except Exception as exc:
            status("failed", error=str(exc))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--approved-config-sha", default="")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        raise SystemExit(worker(args.config))
    supervisor(args.config, args.approved_config_sha)


if __name__ == "__main__":
    main()
