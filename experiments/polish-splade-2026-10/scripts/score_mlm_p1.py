"""Check the P1 scoring recipe on CPU, or run the explicitly approved GPU block."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "datasets/arms"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_identity(row: dict) -> tuple:
    return row["query_id"], row["query"], tuple(row["pos"]), tuple(row["neg"])


def ensemble_row(sparse: dict, dense: dict) -> dict:
    if row_identity(sparse) != row_identity(dense):
        raise ValueError("Teacher rows differ in query or passage text/order")
    row = dict(sparse)
    for field, passages in (("pos_scores", "pos"), ("neg_scores", "neg")):
        a, b = sparse[field], dense[field]
        if len(a) != len(b) or len(a) != len(row[passages]):
            raise ValueError(f"Unaligned {field}")
        if not all(math.isfinite(float(x)) for x in [*a, *b]):
            raise ValueError("Non-finite teacher score")
        row[field] = [(x + y) / 2 for x, y in zip(a, b)]
    row["teacher_model"] = f"ensemble: mean({sparse['teacher_model']}, {dense['teacher_model']})"
    return row


def combine(sparse: Path, dense: Path, output: Path, ab_output: Path, manifest: dict, ab_source: Path, mnrl_source: Path, mnrl_output: Path) -> None:
    side = json.loads(Path(str(dense) + ".teacher.json").read_text())
    if not math.isfinite(side["scale"]) or side["scale"] <= 0 or side["rows"] != manifest["rows"]:
        raise ValueError("Invalid dense scaling sidecar")
    subset = set()
    with ab_source.open(encoding="utf-8") as handle:
        for line in handle:
            subset.add(row_identity(json.loads(line)))
    mnrl_ids = set((mnrl_source.parent / "p1_mnrl.ids").read_text().splitlines())
    count = examples = ab_count = mnrl_count = 0
    partial = output.with_suffix(output.suffix + ".partial")
    ab_partial = ab_output.with_suffix(ab_output.suffix + ".partial")
    mnrl_partial = mnrl_output.with_suffix(mnrl_output.suffix + ".partial")
    with ExitStack() as stack:
        a = stack.enter_context(sparse.open(encoding="utf-8"))
        b = stack.enter_context(dense.open(encoding="utf-8"))
        out = stack.enter_context(partial.open("x", encoding="utf-8"))
        ab = stack.enter_context(ab_partial.open("x", encoding="utf-8"))
        mnrl = stack.enter_context(mnrl_partial.open("x", encoding="utf-8"))
        for la, lb in zip_longest(a, b):
            if la is None or lb is None:
                raise ValueError("Teacher files have different numbers of rows")
            row = ensemble_row(json.loads(la), json.loads(lb))
            line = json.dumps(row, ensure_ascii=False) + "\n"
            out.write(line)
            count += 1
            examples += len(row["pos"])
            if row["query_id"] in mnrl_ids:
                mnrl.write(line)
                mnrl_count += 1
            if row_identity(row) in subset:
                ab.write(line)
                ab_count += 1
    if (count, examples, ab_count) != (manifest["rows"], manifest["examples"], manifest["ab_subset"]["rows"]):
        raise ValueError(f"Unexpected ensemble counts: {count}, {examples}, {ab_count}")
    if mnrl_count != len(mnrl_ids):
        raise ValueError("The MNRL query ID set is not a subset of the scored KL pool")
    partial.replace(output)
    ab_partial.replace(ab_output)
    mnrl_partial.replace(mnrl_output)
    ensemble_side = {
        "rows": count, "examples": examples, "ab_rows": ab_count,
        "dense_scale": side["scale"], "dense_sidecar": str(dense) + ".teacher.json",
        "recipe": "mean(soup-v12 dot product, already scaled PolDense cosine)",
        "training_score_scale": 1.0, "input_pool_sha256": manifest["sha256"],
        "sha256": sha256(output), "ab_sha256": sha256(ab_output),
        "mnrl_rows": mnrl_count, "mnrl_sha256": sha256(mnrl_output),
    }
    Path(str(output) + ".teacher.json").write_text(json.dumps(ensemble_side, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitchen", type=Path, default=Path.home() / "work/speakleash/retrieval-data-kitchen")
    parser.add_argument("--run", action="store_true", help="requires P1_SCORING_APPROVED=yes")
    args = parser.parse_args()
    os.chdir(ROOT)
    source = DATA / "mlm-p1-kl-v2.jsonl"
    manifest = json.loads(Path(str(source) + ".manifest.json").read_text())
    if sha256(source) != manifest["sha256"]:
        raise ValueError("P1 checksum differs from its reviewed manifest")
    ab_source = ROOT / manifest["ab_subset"]["path"]
    if sha256(ab_source) != manifest["ab_subset"]["sha256"]:
        raise ValueError("A/B subset checksum differs from its manifest")
    python = ROOT / ".venv/bin/python"
    mnrl_source = DATA / "mlm-p1-mnrl-v2.jsonl"
    expected_mnrl = manifest["files"][mnrl_source.name]["sha256"]
    if sha256(mnrl_source) != expected_mnrl:
        raise ValueError("MNRL pool checksum differs from its reviewed manifest")
    if sha256(DATA / "p1_mnrl.ids") != manifest["files"]["p1_mnrl.ids"]["sha256"]:
        raise ValueError("MNRL identifiers differ from their reviewed manifest")
    if sha256(DATA / "p1_kl.ids") != manifest["files"]["p1_kl.ids"]["sha256"]:
        raise ValueError("KL identifiers differ from their reviewed manifest")
    sparse = DATA / "mlm-p1-kl-v2-tv12.jsonl"
    dense = DATA / "mlm-p1-kl-v2-tpd400.jsonl"
    output = DATA / "mlm-p1-kl-v2-tens400.jsonl"
    ab_output = DATA / "mlm-p1-mnrl-v2-ab40k-tens400.jsonl"
    mnrl_output = DATA / "mlm-p1-mnrl-v2-tens400.jsonl"
    bridges = [args.kitchen / "bridge/splade_teacher_scores.py", args.kitchen / "bridge/dense_teacher_scores.py"]
    for path in [python, ROOT / "models/soup-v12/model.safetensors", *bridges]:
        if not path.is_file():
            raise FileNotFoundError(path)
    commands = [
        [str(python), str(bridges[0]), "--teacher", "models/soup-v12", "--rows", str(source), "--output", str(sparse), "--device", "cuda", "--batch-size", "16"],
        [str(python), str(bridges[1]), "--teacher", "OPI-PIB/PolDense-400M", "--preset", "poldense", "--rows", str(source), "--reference", str(sparse), "--output", str(dense), "--dtype", "bf16", "--device", "cuda", "--batch-size", "16"],
    ]
    for command in commands:
        print(shlex.join(command), flush=True)
    free = shutil.disk_usage("/mnt/c").free
    required = 10_000_000_000 + source.stat().st_size * 5
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable"}:
            memory[name + "_GB"] = int(value.split()[0]) * 1024 / 1e9
    print(json.dumps({"rows": manifest["rows"], "examples": manifest["examples"], "texts_including_queries": manifest["examples"] + 11 * manifest["rows"], "free_C_GB": free / 1e9, "required_C_GB": required / 1e9, "dense_script_keeps_input_and_reference_rows_in_RAM": True, **memory}), flush=True)
    if free < required:
        raise RuntimeError("Insufficient C: space for teacher files plus 10 GB reserve")
    if not args.run:
        print("CPU preflight passed. No model was loaded and no scoring was started.")
        return
    if os.environ.get("P1_SCORING_APPROVED") != "yes":
        parser.error("GPU scoring requires explicit approval and P1_SCORING_APPROVED=yes")
    import fcntl

    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "logs/gpu_5070.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if active:
            raise RuntimeError(f"GPU already has compute processes: {active}")
        for command, target in zip(commands, (sparse, dense)):
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing scores: {target}")
            started = time.monotonic()
            with (ROOT / "logs" / f"{target.stem}.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"Completed {target}: {time.monotonic() - started:.1f} seconds", flush=True)
        combine(sparse, dense, output, ab_output, manifest, ab_source, mnrl_source, mnrl_output)
        print(f"Scoring complete: {output}; training has not been started.", flush=True)


if __name__ == "__main__":
    main()
