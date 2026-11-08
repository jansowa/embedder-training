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


def worker(config_path: Path, tag: str, start_path: Path) -> int:
    if os.environ.get("P1_AB_APPROVED") != "yes":
        raise RuntimeError("Continuation A/B approval required")
    sys.path.insert(0, str(ROOT))
    from transformers import TrainerCallback
    from training.backends import sentence_transformers_backend as backend
    from training.backends.splade_checkpoint_compat import add_decoder_bias_alias, restore_and_verify_mlm_weights

    start = json.loads(start_path.read_text())
    checkpoint = Path(start["checkpoint"])
    if digest(checkpoint / "model.safetensors") != start["weights_sha256"]:
        raise ValueError("The common starting checkpoint changed")
    original_builder = backend._build_splade_model

    def verified_builder(*args, **kwargs):
        model = original_builder(*args, **kwargs)
        verification = restore_and_verify_mlm_weights(model, checkpoint)
        save_json(ROOT / f"logs/{PREFIX}-{tag}-initial-weights.json", dict(verification, start=start, time=now()))
        print(f"[CONTINUE A/B] Initial checkpoint weights verified: {verification}", flush=True)
        return model

    backend._build_splade_model = verified_builder

    class Checkpoints(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if tag != "smoke" and state.global_step == 1:
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
            recent = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if len(recent) > 2:
                raise ValueError("Continuation checkpoint rotation failed")
            save_json(ROOT / f"logs/{PREFIX}-{tag}-checkpoints.json", {
                "step": state.global_step, "updated_at": now(), "checkpoints": [str(p) for p in recent]})
            return control

    attach_checkpoint_observer(Checkpoints)
    rc = instrumented_worker(config_path, tag, artifact_prefix=PREFIX)
    if digest(checkpoint / "model.safetensors") != start["weights_sha256"]:
        raise ValueError("A/B worker modified the shared source checkpoint")
    return rc


def prepare_run(tag: str, run_name: str) -> Path:
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
    if list(target.glob("**/resolved_config.yaml")):
        raise RuntimeError("Refusing an implicit continuation restart")
    return target


def run_arm(spec: dict, tag: str, start_path: Path, env: dict) -> dict:
    config = Path(spec["snapshot"])
    if digest(config) != spec["sha256"]:
        raise ValueError("Bound continuation config changed")
    target = prepare_run(tag, spec["run_name"])
    with (ROOT / f"logs/{PREFIX}-{tag}.log").open("x") as log:
        process = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--worker", "--tag", tag,
                                    "--config", str(config), "--start", str(start_path)], cwd=ROOT, env=env,
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


def supervisor(approval_path: Path) -> None:
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
            if start_path.exists():
                raise RuntimeError("A continuation start was already bound")
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
                with path.open("x") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
                bound[tag] = {"snapshot": str(path), "sha256": digest(path), "run_name": Path(config["runs_dir"]).name}
            assert_matched(configs["kl"], configs["kl-mnrl"])
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
                result["smoke"] = run_arm(bound["smoke"], "smoke", start_path, env)
                for tag in ("kl", "kl-mnrl"):
                    status("training_ab", mode="continuation", arm=tag, start=start)
                    arm = run_arm(bound[tag], tag, start_path, env)
                    arm.update(measure_model(Path(arm["final"]), f"{PREFIX}-{tag}", env))
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
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        raise SystemExit(worker(args.config, args.tag, args.start))
    supervisor(args.approval)


if __name__ == "__main__":
    main()
