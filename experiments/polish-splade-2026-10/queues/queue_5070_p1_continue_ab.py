"""Compare KL with KL+MNRL as matched continuations of the best P1 checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from queue_5070_p1_ab import DATA, primary_ready, worker as instrumented_worker
sys.path.insert(0, str(ROOT))

STATE = ROOT / "logs/mlm-p1-ab-queue.json"
APPROVAL = ROOT / "logs/mlm-p1-continue-ab-approval.json"
RESULT = EXP / "results/data/mlm-p1-v2-continuation-ab.json"
PREFIX = "mlm-p1-continue-ab"
PLACEHOLDER = "__P1_BEST_CHECKPOINT__"


def choose_start(result: dict) -> dict:
    if result.get("status") != "completed" or [r["epoch"] for r in result.get("epochs", [])] != [1, 2, 3]:
        raise ValueError("Three completed KL epochs and their measurements are required")
    for record in result["epochs"]:
        if not math.isfinite(record["tiny10"]):
            raise ValueError("Invalid selection metric")
    # heldout8 is deliberately never used for checkpoint or arm selection.
    return deepcopy(max(result["epochs"], key=lambda row: row["tiny10"]))


def selected_regularizers(source: Path) -> dict:
    state = json.loads((source / "trainer_state.json").read_text())
    history = source / "regularizer_control.json"
    if not history.is_file():
        history = source.parent.parent / "regularizer_control.json"
    events = json.loads(history.read_text())["events"]
    eligible = [e for e in events if e.get("event") == "control" and e["step"] <= state["global_step"]]
    if not eligible:
        raise ValueError("Cannot recover the selected checkpoint's regularizer weights")
    event = max(eligible, key=lambda e: e["step"])
    weights = {"document": event["document_weight"], "query": event["query_weight"],
               "source_step": event["step"], "history": str(history), "history_sha256": digest(history)}
    if any(not math.isfinite(weights[k]) or weights[k] <= 0 for k in ("document", "query")):
        raise ValueError("Invalid selected regularizer weights")
    return weights


def bind_config(template: dict, checkpoint: Path, weights: dict) -> dict:
    config = deepcopy(template)
    if config.get("architectures") != [PLACEHOLDER]:
        raise ValueError("Continuation template must have a deferred checkpoint start")
    if config["hparams"][0]["learning_rate"] != 5e-6:
        raise ValueError("Continuation LR must be 5e-6")
    backend = config["sentence_transformers"]
    if backend.get("splade_decoder_bias_shift") is not None or "regularizer_control" in backend:
        raise ValueError("Continuation must not recalibrate the decoder or restart the regularizer controller")
    config["architectures"] = [str(checkpoint)]
    config["hparams"][0]["document_regularizer_weight"] = weights["document"]
    backend["query_regularizer_weight"] = weights["query"]
    return config


def assert_matched(control: dict, combined: dict) -> None:
    normalized = []
    for config, expected_loss in ((control, "sparse_distill_kl_div"), (combined, "sparse_kl_plus_mnrl")):
        copy = deepcopy(config)
        if copy["hparams"][0].pop("loss") != expected_loss:
            raise ValueError("Unexpected A/B loss")
        copy.pop("runs_dir")
        weight = copy["sentence_transformers"].pop("mnrl_weight", None)
        if expected_loss == "sparse_kl_plus_mnrl" and weight != 1:
            raise ValueError("Only the weight-one MNRL arm is approved")
        normalized.append(copy)
    if normalized[0] != normalized[1]:
        raise ValueError("A/B arms differ in settings other than loss and output directory")


def comparison(base: dict, arms: list[dict]) -> dict:
    control, combined = arms
    costs = {key: combined["density"][key] / control["density"][key] if control["density"][key] > 0 else None
             for key in ("doc_active_mean", "query_active_mean", "postings_per_query")}
    return {"mnrl_minus_kl_tiny10": combined["tiny10"] - control["tiny10"],
            "kl_minus_start_tiny10": control["tiny10"] - base["tiny10"],
            "mnrl_minus_start_tiny10": combined["tiny10"] - base["tiny10"],
            "mnrl_over_kl_cost_ratios": costs,
            "next_stage": "Review tiny10 and cost changes before choosing full relevance>=2 validation (301042 rows) or n5+kj fine-tuning (20292 rows plus a selected MS MARCO supplement)."}


def arm_checkpoint(run: Path, start: dict) -> tuple[Path, dict]:
    import struct
    import zipfile
    from queue_5070_p1_service import intact_checkpoint
    paths = [p for p in run.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
    paths.sort(key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for path in paths:
        try:
            state = intact_checkpoint(path, max_steps=1250)
        except (OSError, ValueError, KeyError, struct.error, zipfile.BadZipFile):
            continue
        if json.loads((path / "continuation-start.json").read_text()) != start:
            raise ValueError("A/B checkpoint has a different starting model or regularizers")
        if not 0 < state["global_step"] < 1250:
            raise ValueError("A/B training already completed; final export needs recovery")
        return path, state
    raise ValueError("No intact optimizer checkpoint for interrupted A/B arm")


def apply_arm_checkpoint_policy(args, state=None):
    from transformers.trainer_utils import IntervalStrategy
    args.save_strategy = IntervalStrategy.STEPS
    args.save_steps = 100
    args.save_total_limit = 2
    if state is not None:
        state.save_steps = 100


def archive_interrupted_telemetry(run: Path, step: int) -> None:
    """Keep discarded branch metrics separate from the restored trajectory."""
    import shutil
    path = run / "p1-ab-telemetry.jsonl"
    if not path.exists():
        return
    stamp = time.time_ns()
    shutil.copy2(path, path.with_name(f"p1-ab-telemetry-before-resume-{stamp}.jsonl"))
    kept = []
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except ValueError:
            if index == len(lines) - 1:
                continue  # Preserve a shutdown-truncated tail in the archive.
            raise
        if record["step"] <= step:
            kept.append(line if line.endswith("\n") else line + "\n")
    temporary = path.with_name(f"{path.name}.resuming-{stamp}")
    temporary.write_text("".join(kept))
    temporary.replace(path)


def worker(config_path: Path, tag: str, start_path: Path, resume_checkpoint: Path | None = None) -> int:
    if os.environ.get("P1_AB_APPROVED") != "yes":
        raise RuntimeError("Continuation A/B approval required")
    sys.path.insert(0, str(ROOT))
    from transformers import TrainerCallback
    from training.backends import sentence_transformers_backend as backend
    from training.backends.splade_checkpoint_compat import add_decoder_bias_alias, clear_single_device_map, restore_and_verify_mlm_weights

    save_json(ROOT / f"logs/{PREFIX}-{tag}-memory-policy.json", {
        "loss_wrapper": "SpladeLoss", "cached_loss": False,
        "full_batch_size": 8, "gradient_accumulation_steps": 4, "time": now(),
    })
    start = json.loads(start_path.read_text())
    checkpoint = Path(start["checkpoint"])
    if digest(checkpoint / "model.safetensors") != start["weights_sha256"]:
        raise ValueError("The common starting checkpoint changed")
    original_builder = backend._build_splade_model

    def verified_builder(*args, **kwargs):
        model = original_builder(*args, **kwargs)
        clear_single_device_map(model[0].auto_model)
        verification = restore_and_verify_mlm_weights(model, checkpoint)
        save_json(ROOT / f"logs/{PREFIX}-{tag}-initial-weights.json", dict(verification, start=start, time=now()))
        print(f"[CONTINUE A/B] Initial checkpoint weights verified: {verification}", flush=True)
        return model

    backend._build_splade_model = verified_builder
    base_args = backend._training_args
    def checkpoint_args(*args, **kwargs):
        result = base_args(*args, **kwargs)
        apply_arm_checkpoint_policy(result)
        return result
    backend._training_args = checkpoint_args
    resumed_step = 0
    if resume_checkpoint is not None:
        from queue_5070_p1_service import intact_checkpoint
        from sentence_transformers import SparseEncoderTrainer
        from training.backends.splade_checkpoint_compat import reload_sparse_model_on_cpu
        restored = intact_checkpoint(resume_checkpoint, max_steps=1250)
        if json.loads((resume_checkpoint / "continuation-start.json").read_text()) != start:
            raise ValueError("Resume provenance differs from the bound A/B start")
        resumed_step = restored["global_step"]
        def verified_reload(trainer, checkpoint_path):
            verification = reload_sparse_model_on_cpu(trainer, Path(checkpoint_path))
            save_json(ROOT / f"logs/{PREFIX}-{tag}-resume-weights.json", dict(
                verification, checkpoint=checkpoint_path, step=resumed_step, time=now()))
            print(f"[CONTINUE A/B] Resumed weights verified: {verification}", flush=True)
        SparseEncoderTrainer._load_from_checkpoint = verified_reload

    class Checkpoints(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if resume_checkpoint is not None and (state.global_step != resumed_step or state.max_steps != 1250
                    or abs(state.epoch - restored["epoch"]) > 0.00001):
                raise ValueError("A/B trainer did not restore the checkpoint state")
            if resume_checkpoint is not None:
                archive_interrupted_telemetry(Path(args.output_dir), resumed_step)
            apply_arm_checkpoint_policy(args, state)
            save_json(Path(args.output_dir) / "continuation-checkpoint-policy.json", {
                "save_steps": 100, "save_total_limit": 2, "restored_step": state.global_step,
                "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
                "time": now(),
            })
            print(f"[CONTINUE A/B] Start at step {state.global_step}; save every 100 steps, keep two", flush=True)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if tag != "smoke" and state.global_step == resumed_step + 1:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            output = Path(args.output_dir)
            current = output / f"checkpoint-{state.global_step}"
            for target in (current, *list((output / "epoch-checkpoints").glob(f"epoch-*-step-{state.global_step}"))):
                add_decoder_bias_alias(target)
                for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
                    if not (target / name).is_file() or not (target / name).stat().st_size:
                        raise ValueError(f"Incomplete continuation checkpoint: {target}/{name}")
                save_json(target / "continuation-start.json", start)
                save_json(target / "continuation-checkpoint-policy.json", {
                    "save_steps": 100, "save_total_limit": 2, "saved_step": state.global_step,
                })
            recent = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if len(recent) > 2:
                raise ValueError("Continuation checkpoint rotation failed")
            save_json(ROOT / f"logs/{PREFIX}-{tag}-checkpoints.json", {
                "step": state.global_step, "updated_at": now(), "checkpoints": [str(p) for p in recent]})
            return control

    attach_checkpoint_observer(Checkpoints)
    options = {"artifact_prefix": PREFIX}
    if resume_checkpoint is not None:
        options["resume_from_checkpoint"] = resume_checkpoint
    rc = instrumented_worker(config_path, tag, **options)
    if digest(checkpoint / "model.safetensors") != start["weights_sha256"]:
        raise ValueError("A/B worker modified the shared source checkpoint")
    return rc


def prepare_run(tag: str, run_name: str, *, resume: bool = False) -> Path:
    parent = (ROOT / "cache/mlm-p1-v2-continuation-ab-runs").resolve()
    target = (parent / tag).resolve()
    if not target.is_relative_to(parent) or target == parent:
        raise ValueError("Invalid continuation output directory")
    link = ROOT / "runs" / run_name
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target:
            raise ValueError("Unexpected existing continuation output")
    else:
        target.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    existing = list(target.glob("**/resolved_config.yaml"))
    if existing and not resume:
        raise RuntimeError("Refusing an implicit continuation restart")
    if len(existing) > 1:
        raise RuntimeError("Ambiguous continuation run")
    return target


def run_arm(spec: dict, tag: str, start_path: Path, env: dict, *, resume: bool = False) -> dict:
    config = Path(spec["snapshot"])
    if digest(config) != spec["sha256"]:
        raise ValueError("Bound continuation config changed")
    target = prepare_run(tag, spec["run_name"], resume=resume)
    result_path = ROOT / f"logs/{PREFIX}-{tag}-result.json"
    if resume and result_path.exists():
        result = json.loads(result_path.read_text())
        expected_steps = 20 if tag == "smoke" else 1250
        final = Path(result["final"]) / "model.safetensors"
        if result["config_sha256"] != spec["sha256"] or result["tag"] != tag or result["steps"] != expected_steps or not final.is_file():
            raise ValueError("Completed A/B artifact does not match the approved arm")
        final_sha = digest(final)
        if result.get("final_weights_sha256", final_sha) != final_sha:
            raise ValueError("Completed A/B weights changed")
        return result
    checkpoint = None
    runs = list(target.glob("**/resolved_config.yaml"))
    if runs:
        checkpoint, _ = arm_checkpoint(runs[0].parent, json.loads(start_path.read_text()))
        checkpoint = ROOT / "runs" / spec["run_name"] / checkpoint.relative_to(target)
    suffix = "" if checkpoint is None else f"-resume-{time.time_ns()}"
    log_path = ROOT / f"logs/{PREFIX}-{tag}{suffix}.log"
    command = [str(PYTHON), str(Path(__file__).resolve()), "--worker", "--tag", tag,
               "--config", str(config), "--start", str(start_path)]
    if checkpoint is not None:
        command.extend(["--resume-checkpoint", str(checkpoint.relative_to(ROOT))])
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        def terminate(sig, frame):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            raise SystemExit(128 + sig)
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        import shutil
        while process.poll() is None:
            if min(shutil.disk_usage(target).free, shutil.disk_usage("/mnt/c").free) < 10_000_000_000:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
                raise RuntimeError("Disk reserve below 10 GB")
            time.sleep(15)
        if process.returncode:
            raise RuntimeError(f"Continuation {tag} failed; see logs/{PREFIX}-{tag}.log")
    return json.loads((ROOT / f"logs/{PREFIX}-{tag}-result.json").read_text())


def write_report(result: dict) -> None:
    lines = ["# P1: A/B kontynuacji KL kontra KL+MNRL", "",
             "Wspólny start: najlepszy tiny10 spośród trzech epok KL; świeży optymalizator w każdym ramieniu.",
             "40 000 tych samych zapytań relevance>=2, jeden pozytyw i 10 negatywów; seed 42, batch 8×4, LR 5e-6, jedna epoka.",
             "Wspólne stałe wagi regularyzacji z wybranego checkpointu; bez ponownej kalibracji biasu. Bez ramienia z wagą 3.", "",
             "| Wariant | tiny10 | heldout8 | wym./dok | wym./zapytanie | postings/zapytanie |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in [result["start"], *result.get("arms", [])]:
        p = row["density"]
        lines.append(f"| {row.get('tag', 'start KL')} | {row['tiny10']:.4f} | {row.get('heldout8', '—')} | {p['doc_active_mean']:.2f} | {p['query_active_mean']:.2f} | {p['postings_per_query']:.2f} |")
    lines += ["", "Heldout8 nie uczestniczy w wyborze checkpointu ani ramienia. Wyższy tiny10 wyznacza wariant do oceny heldout8, z uwzględnieniem niezmienionego startu jako kandydata.",
              "", "```json", json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2), "```", ""]
    (EXP / "results/summary_5070_mlm_p1_continuation_ab.md").write_text("\n".join(lines), encoding="utf-8")


def supervisor(approval_path: Path, *, resume: bool = False) -> None:
    import fcntl
    import yaml
    if os.environ.get("P1_AB_APPROVED") != "yes":
        raise RuntimeError("Continuation A/B approval required")
    approval = json.loads(approval_path.read_text())
    def status(value, **extra):
        save_json(STATE, {"status": value, "updated_at": now(), "pid": os.getpid(), "approved": True,
                          "scope": "KL vs KL+MNRL continuation from best KL checkpoint; 40k rows, LR 5e-6; no weight 3", **extra})
    with (ROOT / "logs/mlm-p1-ab-queue.lock").open("a") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                state = json.loads((ROOT / "logs/mlm-p1-train-queue.json").read_text())
                if primary_ready(state):
                    break
                status("waiting_kl", kl_status=state["status"], mode="continuation")
                time.sleep(30)
            primary = json.loads((EXP / "results/data/mlm-p1-v2-training.json").read_text())
            base = choose_start(primary)
            checkpoint = Path(base["checkpoint"])
            source = Path(base.get("source_checkpoint", base["checkpoint"]))
            pool = json.loads((ROOT / "datasets/arms/mlm-p1-kl-v2.jsonl.manifest.json").read_text())
            side = json.loads((ROOT / "datasets/arms/mlm-p1-kl-v2-tens400.jsonl.teacher.json").read_text())
            validate_teacher(side, pool)
            if pool["sha256"] != approval["pool_sha256"] or digest(DATA) != approval["data_sha256"] or digest(DATA) != side["ab_sha256"] or side["ab_rows"] != 40000:
                raise ValueError("Continuation data differs from the approved 40k subset")
            weights = selected_regularizers(source)
            start = {"checkpoint": str(checkpoint), "source_checkpoint": str(source), "epoch": base["epoch"],
                     "weights_sha256": digest(checkpoint / "model.safetensors"), "regularizers": weights,
                     "selection": "highest tiny10 across all three KL epochs; heldout8 ignored"}
            start_path = ROOT / f"logs/{PREFIX}-start.json"
            existing = start_path.exists()
            saved_bound = None
            if existing:
                if not resume:
                    raise RuntimeError("A continuation start was already bound; explicit resume required")
                saved_bound = json.loads((ROOT / f"logs/{PREFIX}-bound.json").read_text())
                if (json.loads(start_path.read_text()) != start or saved_bound["start"] != start
                        or saved_bound["data_sha256"] != approval["data_sha256"]):
                    raise ValueError("Frozen continuation provenance changed")
            else:
                save_json(start_path, start)
            bound = {}
            configs = {}
            for tag in ("smoke", "kl", "kl-mnrl"):
                spec = approval["templates"][tag]
                template_path = Path(spec["snapshot"])
                if digest(template_path) != spec["sha256"]:
                    raise ValueError("Approved continuation template changed")
                config = bind_config(yaml.safe_load(template_path.read_text()), checkpoint, weights)
                configs[tag] = config
                path = ROOT / f"logs/{PREFIX}-{tag}-bound.yaml"
                if existing:
                    saved = saved_bound["configs"][tag]
                    if (Path(saved["snapshot"]) != path or digest(path) != saved["sha256"]
                            or yaml.safe_load(path.read_text()) != config):
                        raise ValueError("Frozen continuation configuration changed")
                else:
                    with path.open("x") as handle:
                        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
                bound[tag] = {"snapshot": str(path), "sha256": digest(path), "run_name": Path(config["runs_dir"]).name}
                if existing and bound[tag] != saved_bound["configs"][tag]:
                    raise ValueError("Frozen continuation run binding changed")
            assert_matched(configs["kl"], configs["kl-mnrl"])
            if not existing:
                save_json(ROOT / f"logs/{PREFIX}-bound.json", {"start": start, "configs": bound, "data_sha256": approval["data_sha256"]})
            with (ROOT / "logs/gpu_5070.lock").open("a") as gpu:
                status("waiting_gpu_lock")
                fcntl.flock(gpu, fcntl.LOCK_EX)
                while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
                    status("waiting_gpu_idle")
                    time.sleep(15)
                env = dict(os.environ, P1_AB_APPROVED="yes", WANDB_MODE="disabled", ORPHAN_GUARD_EXEMPT="1", PYTHONPATH=str(ROOT),
                           PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           TORCHINDUCTOR_COMPILE_THREADS="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                           REGIMES_ROOT=str(EXP / "results-regimes"), RESULTS_ROOT=str(EXP / "results"))
                result = {"status": "running", "started_at": now(), "start": base, "start_provenance": start,
                          "arms": [], "data_rows": 40000, "data_sha256": approval["data_sha256"]}
                status("smoke", mode="continuation", start=start)
                options = {"resume": True} if resume else {}
                result["smoke"] = run_arm(bound["smoke"], "smoke", start_path, env, **options)
                for tag in ("kl", "kl-mnrl"):
                    status("training_ab", mode="continuation", arm=tag, start=start)
                    arm = run_arm(bound[tag], tag, start_path, env, **options)
                    measurements = {"reuse_tiny": True, "append_log": True} if resume else {}
                    arm.update(measure_model(Path(arm["final"]), f"{PREFIX}-{tag}", env, **measurements))
                    result["arms"].append(arm)
                    save_json(RESULT, result)
                    write_report(result)
                result["comparison"] = comparison(base, result["arms"])
                best = max([base, *result["arms"]], key=lambda r: r["tiny10"])
                heldout_path = EXP / f"results-regimes/quantized/{best['label']}/heldout8.json"
                if best is not base or not heldout_path.exists():
                    subprocess.run(["bash", "scripts/pirb_regime_eval.sh", "quantized", best["label"], best["checkpoint"], "heldout8", "16", "512"],
                                   cwd=ROOT, env=env, check=True)
                heldout = json.loads(heldout_path.read_text())
                from training.backends.splade_checkpoint_compat import same_model_source
                if (heldout.get("n_tasks") != 8 or not math.isfinite(heldout["mean_ndcg@10"])
                        or heldout.get("regime") != "quantized" or heldout.get("max_seq_length") != 512
                        or not same_model_source(Path(heldout["model"]), Path(best["checkpoint"]))):
                    raise ValueError("Incomplete selected heldout8 result")
                best["heldout8"] = heldout["mean_ndcg@10"]
                result.update(status="completed", finished_at=now(), selected_label=best["label"], selected_checkpoint=best["checkpoint"])
                save_json(RESULT, result)
                write_report(result)
                paths = [RESULT, EXP / "results/summary_5070_mlm_p1_continuation_ab.md"]
                for arm in result["arms"]:
                    paths.extend([EXP / f"results/sparsity/{arm['label']}-index-cost.json", EXP / f"results-regimes/quantized/{arm['label']}/tiny10.json"])
                    heldout_file = EXP / f"results-regimes/quantized/{arm['label']}/heldout8.json"
                    if heldout_file.exists():
                        paths.append(heldout_file)
                try:
                    publication = publish_paths(paths, "Add the checkpoint continuation KL and MNRL comparison", ROOT / f"logs/{PREFIX}-publish.log")
                except Exception as exc:
                    publication = {"pushed": False, "error": str(exc)}
                status("completed", mode="continuation", selected_label=best["label"], publication=publication)
        except Exception as exc:
            status("failed", mode="continuation", error=str(exc))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tag", choices=("smoke", "kl", "kl-mnrl"))
    parser.add_argument("--start", type=Path)
    parser.add_argument("--approval", type=Path, default=APPROVAL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        raise SystemExit(worker(args.config, args.tag, args.start, args.resume_checkpoint))
    supervisor(args.approval, resume=args.resume)


if __name__ == "__main__":
    main()
