"""Run the approved matched A/B after P1 KL: smoke, KL, KL+MNRL, optional weight 3."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from queue_5070_p1_train import ROOT, EXP, PYTHON, attach_checkpoint_observer, digest, measure_model, now, publish_paths, save_json, validate_teacher

STATE = ROOT / "logs/mlm-p1-ab-queue.json"
RESULT = EXP / "results/data/mlm-p1-v2-ab.json"
DATA = ROOT / "datasets/arms/mlm-p1-mnrl-v2-ab40k-tens400.jsonl"
RECOVERY_STATE = ROOT / "logs/mlm-p1-measurement-recovery.json"


def weight_three_needed(components: dict) -> bool:
    return 1 <= components.get("kl", -1) <= 3 and 0.05 <= components.get("mnrl", -1) <= 0.2


def primary_ready(state: dict) -> bool:
    if state["status"] == "completed":
        return True
    if state["status"] in {"failed", "stopped_gate"}:
        try:
            recovery = json.loads(RECOVERY_STATE.read_text())
        except (OSError, ValueError):
            recovery = {}
        if recovery.get("status") in {"waiting_training", "evaluating"}:
            return False
        raise RuntimeError(f"Primary KL job needs review: {state['status']}")
    return False


class Components:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.window_sums = defaultdict(float)
        self.window_counts = defaultdict(int)
        self.last = {}

    def capture(self, name, method):
        def wrapped(*args, **kwargs):
            loss = method(*args, **kwargs)
            value = float(loss.detach().item())
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {name} loss")
            self.sums[name] += value
            self.counts[name] += 1
            self.window_sums[name] += value
            self.window_counts[name] += 1
            return loss
        return wrapped

    def flush(self) -> dict:
        values = {k: self.window_sums[k] / n for k, n in self.window_counts.items() if n}
        if values:
            self.last = values
        self.window_sums.clear()
        self.window_counts.clear()
        return values

    def means(self) -> dict:
        return {k: self.sums[k] / n for k, n in self.counts.items() if n}


def worker(config: Path, tag: str) -> int:
    if os.environ.get("P1_AB_APPROVED") != "yes":
        raise RuntimeError("A/B approval required")
    sys.path.insert(0, str(ROOT))
    import torch
    from transformers import TrainerCallback
    from training.backends import sentence_transformers_backend as backend
    from training.train import main as train_main

    components = Components()
    loader = backend._load_sparse_kl_plus_mnrl_loss
    def instrumented_loader():
        base = loader()
        class Instrumented(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.kl.compute_loss_from_embeddings = components.capture("kl", self.kl.compute_loss_from_embeddings)
                self.mnrl.compute_loss_from_embeddings = components.capture("mnrl", self.mnrl.compute_loss_from_embeddings)
        return Instrumented
    backend._load_sparse_kl_plus_mnrl_loss = instrumented_loader
    started = time.monotonic()
    result = {"tag": tag, "config": str(config), "config_sha256": digest(config), "started_at": now()}
    class Monitor(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if any(not math.isfinite(float(logs[k])) for k in ("loss", "grad_norm") if k in logs):
                raise RuntimeError("Non-finite loss/gradient in A/B")
            values = components.flush()
            row = {"step": state.global_step, "epoch": state.epoch, "metrics": logs, "components": values,
                   "elapsed_seconds": time.monotonic() - started, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}
            with (Path(args.output_dir) / "p1-ab-telemetry.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            return control

        def on_train_end(self, args, state, control, **kwargs):
            result.update(steps=state.global_step, epoch=state.epoch, output_dir=args.output_dir,
                          final=str(Path(args.output_dir) / "final"), seconds=time.monotonic()-started,
                          components_mean=components.means(), components_last_log=components.last)
            return control
    attach_checkpoint_observer(Monitor)
    rc = train_main(["--config", str(config)])
    if rc:
        return rc
    if tag == "smoke":
        if result.get("steps") != 20:
            raise RuntimeError("A/B smoke did not complete 20 steps")
    elif result.get("epoch", 0) < 0.999:
        raise RuntimeError("A/B arm did not complete one epoch")
    if not (Path(result["final"]) / "model.safetensors").is_file():
        raise RuntimeError("A/B final model missing")
    result["finished_at"] = now()
    save_json(ROOT / f"logs/mlm-p1-ab-{tag}-result.json", result)
    return 0


def prepare_run(tag: str, run_name: str) -> Path:
    parent = (ROOT / "cache/mlm-p1-v2-ab-runs").resolve()
    target = (parent / tag).resolve()
    if not target.is_relative_to(parent) or target == parent:
        raise ValueError("Invalid A/B run target")
    link = ROOT / "runs" / run_name
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target:
            raise RuntimeError(f"Unexpected existing A/B run path: {link}")
    target.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        link.symlink_to(target, target_is_directory=True)
    if list(target.glob("**/resolved_config.yaml")):
        raise RuntimeError("Refusing an implicit A/B restart")
    return target


def run_arm(spec: dict, tag: str, env: dict) -> dict:
    config = Path(spec["snapshot"])
    if digest(config) != spec["sha256"]:
        raise RuntimeError("Approved A/B configuration changed")
    target = prepare_run(tag, spec["run_name"])
    with (ROOT / f"logs/mlm-p1-ab-{tag}.log").open("x") as log:
        process = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--worker", "--tag", tag, "--config", str(config)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        def terminate(sig, frame):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            raise SystemExit(128 + sig)
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        while process.poll() is None:
            if shutil_free("/mnt/c") < 10_000_000_000 or shutil_free(target) < 10_000_000_000:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
                raise RuntimeError("A/B stopped: disk reserve below 10 GB")
            time.sleep(15)
        if process.returncode:
            raise RuntimeError(f"A/B {tag} failed; see logs/mlm-p1-ab-{tag}.log")
    return json.loads((ROOT / f"logs/mlm-p1-ab-{tag}-result.json").read_text())


def shutil_free(path) -> int:
    import shutil
    return shutil.disk_usage(path).free


def write_report(result: dict) -> None:
    lines = ["# P1 v2: A/B KL vs KL+MNRL", "", "Ten sam podzbiór MNRL 40 000 zapytań, jeden pozytyw, 10 negatywów; start MLM, jedna epoka, batch 8×4, seed 42.", "",
             "| Ramię | tiny10 | heldout8 | wym./dok | wym./zapytanie | postings/zapytanie |", "|---|---:|---:|---:|---:|---:|",
             "| baza | 61.60 | 41.03 | — | — | — |", "| soup-v12 | 63.96 | 42.92 | — | — | — |"]
    for record in result.get("arms", []):
        p = record["density"]
        lines.append(f"| {record['tag']} | {record['tiny10']:.2f} | {record.get('heldout8', '—')} | {p['doc_active_mean']:.2f} | {p['query_active_mean']:.2f} | {p['postings_per_query']:.2f} |")
    lines += ["", "Dodatkowe mnrl_weight=3 tylko gdy ostatnie średnie składniki spełniają KL∈[1,3] i MNRL∈[0.05,0.2].", "",
              "```json", json.dumps(result, indent=2), "```", ""]
    (EXP / "results/summary_5070_mlm_p1_v2_ab.md").write_text("\n".join(lines), encoding="utf-8")


def supervisor(approval_path: Path) -> None:
    import fcntl
    if os.environ.get("P1_AB_APPROVED") != "yes":
        raise RuntimeError("A/B approval required")
    approval = json.loads(approval_path.read_text())
    def status(value, **extra):
        save_json(STATE, {"status": value, "updated_at": now(), "pid": os.getpid(), "approved": True,
                          "scope": "matched KL/MNRL A/B after phase 1; no phase 2", **extra})
    with (ROOT / "logs/mlm-p1-ab-queue.lock").open("a") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                kl = json.loads((ROOT / "logs/mlm-p1-train-queue.json").read_text())
                if primary_ready(kl):
                    break
                status("waiting_kl", kl_status=kl["status"])
                time.sleep(30)
            pool = json.loads((ROOT / "datasets/arms/mlm-p1-kl-v2.jsonl.manifest.json").read_text())
            side = json.loads((ROOT / "datasets/arms/mlm-p1-kl-v2-tens400.jsonl.teacher.json").read_text())
            validate_teacher(side, pool)
            if pool["sha256"] != approval["pool_sha256"] or digest(DATA) != side["ab_sha256"] or side["ab_rows"] != 40000:
                raise RuntimeError("A/B teacher file differs from the approved subset")
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
                env = dict(os.environ, WANDB_MODE="disabled", ORPHAN_GUARD_EXEMPT="1", PYTHONPATH=str(ROOT),
                           PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           REGIMES_ROOT=str(EXP / "results-regimes"), RESULTS_ROOT=str(EXP / "results"))
                result = {"status": "running", "started_at": now(), "arms": [], "dense_scale": side["dense_scale"]}
                status("smoke")
                result["smoke"] = run_arm(approval["configs"]["smoke"], "smoke", env)
                for tag in ("kl", "kl-mnrl", "kl-mnrl-weight3"):
                    if tag.endswith("weight3") and not weight_three_needed(result["arms"][-1].get("components_last_log", {})):
                        result["weight3"] = "condition not met; not run"
                        break
                    status("training_ab", arm=tag)
                    arm = run_arm(approval["configs"][tag], tag, env)
                    arm.update(measure_model(Path(arm["final"]), f"mlm-p1-ab-{tag}", env))
                    result["arms"].append(arm)
                    save_json(RESULT, result)
                    write_report(result)
                best = max(result["arms"], key=lambda r: r["tiny10"])
                subprocess.run(["bash", "scripts/pirb_regime_eval.sh", "quantized", best["label"], best["final"], "heldout8", "16", "512"], cwd=ROOT, env=env, check=True)
                heldout = json.loads((EXP / f"results-regimes/quantized/{best['label']}/heldout8.json").read_text())
                if heldout.get("n_tasks") != 8:
                    raise ValueError("Incomplete A/B heldout8")
                best["heldout8"] = heldout["mean_ndcg@10"]
                result.update(status="completed", finished_at=now(), best_arm=best["tag"])
                save_json(RESULT, result)
                write_report(result)
                paths = [RESULT, EXP / "results/summary_5070_mlm_p1_v2_ab.md"]
                for arm in result["arms"]:
                    paths.extend([EXP / f"results/sparsity/{arm['label']}-index-cost.json",
                                  EXP / f"results-regimes/quantized/{arm['label']}/tiny10.json"])
                    heldout_file = EXP / f"results-regimes/quantized/{arm['label']}/heldout8.json"
                    if heldout_file.is_file():
                        paths.append(heldout_file)
                try:
                    publication = publish_paths(paths, "Add the matched P1 KL and MNRL results", ROOT / "logs/mlm-p1-ab-publish.log")
                except Exception as exc:
                    publication = {"pushed": False, "error": str(exc)}
                status("completed", best_arm=best["tag"], publication=publication)
        except Exception as exc:
            status("failed", error=str(exc))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--approval", type=Path, default=ROOT / "logs/mlm-p1-ab-approval.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        raise SystemExit(worker(args.config, args.tag))
    supervisor(args.approval)


if __name__ == "__main__":
    main()
