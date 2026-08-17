#!/usr/bin/env python
"""Prove on real data that an interrupted PIRB benchmark resumes instead of restarting.

The unit tests fake the encoder and the Lucene backend. This drives the real
thing: a real SPLADE checkpoint, real NanoBEIR corpora, real Anserini indexes,
and real SIGKILLs delivered at chosen points of the build. It then reruns the
same benchmark on an empty cache and compares the metrics, which is the only
check that actually proves nothing was lost along the way.

Everything lands under --root (default ~/pirb-resume-e2e) and nothing is written
into the repository, so cleaning up is `rm -rf` of that one directory.

Phases run in order and are individually skippable, because the expensive ones
(env, data, train) only need doing once:

    python scripts/pirb_resume_e2e.py                      # everything
    python scripts/pirb_resume_e2e.py --phases run,verify  # re-run just the test
    python scripts/pirb_resume_e2e.py --keep-cache         # resume a half-done attempt

Kills are timed by watching the artifacts on disk, never by sleeping: the script
waits until the encoding manifest reports a partial corpus, or until phase A is
complete but no index directory exists yet, and only then sends the signal. That
is what makes the scenario reproducible instead of racy.
"""

import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PIRB_ROOT = REPO_ROOT / "third_party" / "pirb"

# pyserini 0.39 ships the Anserini fatjar built for Java 21; the system JDK on a
# workstation is often older, so a private one is fetched instead of touching it.
JDK_URL = ("https://github.com/adoptium/temurin21-binaries/releases/download/"
           "jdk-21.0.5%2B11/OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz")
JDK_DIR_NAME = "jdk-21.0.5+11"
PYTHON_VERSION = "3.10.4"

# Real PIRB tasks, chosen for being the smallest of their kind. NanoBEIR corpora
# are a few thousand documents each and carry the kills. The two PolEval splits are
# the smallest PIRB-native task there is (allegro-faq: 921 passages, 352 KB) and
# they cover what the BEIR-shaped tasks cannot: a corpus outside
# data/<task_id>/passages/ and, between the two of them, a cache directory shared
# by several tasks over one corpus - which is how all seven PolEval tasks are laid
# out in pirb-without-private.json.
TASKS = [
    {"config": {"task_id": "NanoNFCorpus", "type": "beir", "lang": "en"},
     "task_id": "NanoNFCorpus", "cache": "NanoNFCorpus",
     "corpus": "NanoNFCorpus/passages/passages.jsonl"},
    {"config": {"task_id": "NanoSciFact", "type": "beir", "lang": "en"},
     "task_id": "NanoSciFact", "cache": "NanoSciFact",
     "corpus": "NanoSciFact/passages/passages.jsonl"},
    {"config": {"task_id": "NanoFiQA2018", "type": "beir", "lang": "en"},
     "task_id": "NanoFiQA2018", "cache": "NanoFiQA2018",
     "corpus": "NanoFiQA2018/passages/passages.jsonl"},
    {"config": {"split": "test-A", "domain": "allegro-faq", "type": "poleval"},
     "task_id": "poleval-2022-test-A-allegro-faq", "cache": "poleval-2022-allegro-faq",
     "corpus": "poleval-2022/allegro-faq/passages/passages.jsonl"},
    {"config": {"split": "test-B", "domain": "allegro-faq", "type": "poleval"},
     "task_id": "poleval-2022-test-B-allegro-faq", "cache": "poleval-2022-allegro-faq",
     "corpus": "poleval-2022/allegro-faq/passages/passages.jsonl"},
]
TASK_IDS = [task["task_id"] for task in TASKS]


def task_entry(task_id: str) -> Dict:
    for task in TASKS:
        if task["task_id"] == task_id:
            return task
    raise KeyError(task_id)
KILL_TASKS = ["NanoSciFact", "NanoFiQA2018"]
# Tasks whose cache directory is shared with another task, and how many of them
# may encode the corpus: exactly one.
SHARED_CACHE = "poleval-2022-allegro-faq"

# Versions come from the repository's own requirements.lock, which is the only
# resolution that satisfies both halves of the pipeline: SPLADE training needs the
# sparse encoder stack of sentence-transformers 5.x, while PIRB needs pyserini and
# pylate. Pinning pirb/requirements.lock instead gives a venv that cannot train.
PACKAGES = [
    "torch==2.7.1",
    "pyserini==0.39.0",
    "pylate==1.6.0",
    "sentence-transformers==5.3.0",
    "transformers==4.48.2",
    "datasets==3.6.0",
    "numpy==2.2.6",
    "accelerate==1.14.0",
    "scikit-learn",
    "faiss-cpu",
    # pyserini.encode builds an OpenAI client at import time with an empty key,
    # which openai 3.x rejects outright; the lock pins the 1.x that tolerates it.
    "openai==1.102.0",
    "tqdm",
    "pyyaml",
    # The HerBERT tokenizer is a slow XLM tokenizer, hence sentencepiece and
    # sacremoses.
    "sentencepiece",
    "sacremoses",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

# The training config is written into the sandbox rather than taken from configs/,
# so this test cannot be broken by an unrelated edit there - and so the model can
# be an English one, matching the English NanoBEIR corpora it will encode.
#
# Training starts from a real SPLADE checkpoint on purpose. Five steps from a plain
# MLM leave a model whose "sparse" vectors still have tens of thousands of non-zero
# dimensions: passages.jsonl reached 643 MB for 2953 documents when this test was
# first written, and Anserini crawled through it for minutes per task. Fine-tuning
# an already sparse model keeps every artifact the size it is in production.
TRAIN_CONFIG = """backend: sentence-transformers
training_type: splade
train_data: {train_data}
output_dir: {output_dir}

sentence_transformers:
  model_name_or_path: naver/splade-cocondenser-ensembledistil
  max_seq_length: 128
  max_steps: 5
  train_batch_size: 2
  negatives_per_query: 1
  save_strategy: "no"
  report_to: []
  document_regularizer_weight: 0.00003
  query_regularizer_weight: 0.00005
"""

RESULT_TOLERANCE = 1e-9


def log(message: str) -> None:
    print(f"[e2e] {message}", flush=True)


def run(command: List[str], *, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None,
        check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(part) for part in command))
    return subprocess.run(command, env=env, cwd=str(cwd) if cwd else None, check=check)


class Sandbox:
    """Every path this test writes to."""

    def __init__(self, root: Path):
        self.root = root
        self.jdk_home = root / "jdk" / JDK_DIR_NAME
        self.venv = root / ".venv-pirb-test"
        self.data_dir = root / "data"
        self.cache_dir = root / "cache"
        self.control_cache_dir = root / "cache-control"
        self.runs_dir = root / "runs"
        self.model_dir = self.runs_dir / "splade-e2e"
        self.final_model = self.model_dir / "final"
        self.benchmark_config = root / "benchmark-resume-test.json"
        self.models_config = root / "models_config.json"
        self.results_resumed = root / "results-resumed.json"
        self.results_control = root / "results-control.json"
        self.logs = root / "logs"

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    def env(self, **extra: str) -> Dict[str, str]:
        env = os.environ.copy()
        env["JAVA_HOME"] = str(self.jdk_home)
        env["PATH"] = f"{self.jdk_home / 'bin'}:{self.venv / 'bin'}:{env['PATH']}"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONUNBUFFERED"] = "1"
        env["WANDB_MODE"] = "disabled"
        env.pop("PIRB_TRUST_LEGACY_ARTIFACTS", None)
        env.update(extra)
        return env

    def task_paths(self, task_id: str, cache_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Where a task's artifacts live.

        Neither the cache directory nor the corpus can be derived from the task id:
        PolEval keeps its corpus under poleval-2022/<domain>/ and shares one cache
        directory between its splits.
        """
        entry = task_entry(task_id)
        model_name = str(self.final_model).replace("/", "_").replace(".", "_")
        cache_root = (cache_dir or self.cache_dir) / entry["cache"]
        base = cache_root / model_name
        return {
            "corpus": self.data_dir / entry["corpus"],
            "base": base,
            "passages": base / "docs" / "passages.jsonl",
            "manifest": base / "passages_manifest.json",
            "index": base / "lucene_index",
            "results": cache_root / "results",
        }


def phase_env(box: Sandbox) -> None:
    """Fetch a Java 21 runtime and build the test virtualenv."""
    box.root.mkdir(parents=True, exist_ok=True)
    box.logs.mkdir(parents=True, exist_ok=True)

    if not box.jdk_home.is_dir():
        archive = box.root / "jdk21.tar.gz"
        if not archive.exists():
            log(f"downloading a private JDK 21 (the system one is too old for Anserini)")
            urllib.request.urlretrieve(JDK_URL, archive)
        (box.root / "jdk").mkdir(exist_ok=True)
        run(["tar", "-xzf", str(archive), "-C", str(box.root / "jdk")])
        archive.unlink()
    java_version = subprocess.run([str(box.jdk_home / "bin" / "java"), "-version"],
                                  capture_output=True, text=True)
    log(f"java: {java_version.stderr.splitlines()[0]}")

    # A venv on its own proves nothing: a failed install leaves one behind, and
    # skipping the install because of it wastes a run on an unusable environment.
    installed_marker = box.venv / ".e2e-packages-installed"
    if not installed_marker.exists():
        if not box.python.exists():
            run(["uv", "venv", "--python", PYTHON_VERSION, str(box.venv)])
        run(["uv", "pip", "install", "--python", str(box.python),
             "--extra-index-url", TORCH_INDEX, "--index-strategy", "unsafe-best-match", *PACKAGES])
        installed_marker.write_text("\n".join(PACKAGES), encoding="utf-8")
    installed = subprocess.run([str(box.python), "-c",
                                "import torch, pyserini, jnius, pylate; "
                                "print(torch.__version__, torch.cuda.is_available())"],
                               capture_output=True, text=True, env=box.env())
    if installed.returncode != 0:
        raise SystemExit(f"the test environment is not usable:\n{installed.stderr}")
    log(f"python env: torch {installed.stdout.strip()}, pyserini and jnius import cleanly")


def phase_data(box: Sandbox) -> None:
    """Write the one-off benchmark config and download only its tasks."""
    box.benchmark_config.write_text(
        json.dumps([task["config"] for task in TASKS], indent=2),
        encoding="utf-8",
    )
    log(f"benchmark config with {len(TASKS)} task(s): {box.benchmark_config}")
    # PIRB prepares every task of the config before --scope filters anything, which
    # is exactly why this config holds three tiny tasks and not the full benchmark:
    # preparing the real one would download tens of gigabytes to then discard most.
    box.data_dir.mkdir(parents=True, exist_ok=True)
    prepare = (
        "import sys; sys.path.insert(0, %r)\n"
        "from data import Benchmark\n"
        "benchmark = Benchmark.from_config(%r)\n"
        "benchmark.prepare(%r)\n"
    ) % (str(PIRB_ROOT), str(box.benchmark_config), str(box.data_dir))
    run([str(box.python), "-c", prepare], env=box.env(), cwd=PIRB_ROOT)
    for task_id in TASK_IDS:
        corpus = box.task_paths(task_id)["corpus"]
        log(f"{task_id}: {count_lines(corpus)} passages in {corpus.name}, "
            f"{corpus.stat().st_size / 1e6:.1f} MB")


def write_train_config(box: Sandbox) -> Path:
    path = box.root / "train-config.yaml"
    path.write_text(TRAIN_CONFIG.format(
        train_data=REPO_ROOT / "dataset-small-no_in_batch_neg",
        output_dir=box.model_dir,
    ), encoding="utf-8")
    return path


def phase_train(box: Sandbox) -> None:
    """Train a real SPLADE checkpoint, briefly, so that final/ exists."""
    if box.final_model.is_dir():
        log(f"reusing the checkpoint at {box.final_model}")
        return
    if box.model_dir.is_dir():
        # A training run that died leaves run metadata behind, and the CLI then
        # refuses to start a fresh one in the same directory. In the sandbox the
        # answer is simply to start over.
        log(f"clearing the unfinished run directory {box.model_dir}")
        shutil.rmtree(box.model_dir)
    run([str(box.python), "-m", "training.train",
         "--config", str(write_train_config(box)),
         "--no-distributed", "--resume-if-available"],
        env=box.env(), cwd=REPO_ROOT)
    if not box.final_model.is_dir():
        raise SystemExit(f"training did not produce {box.final_model}")
    log(f"checkpoint ready: {box.final_model}")


def phase_skip(box: Sandbox) -> None:
    """Check that a second run skips the training the cluster already paid for."""
    output = subprocess.run(
        [str(box.python), "-m", "training.train",
         "--config", str(write_train_config(box)),
         "--no-distributed", "--resume-if-available"],
        env=box.env(), cwd=str(REPO_ROOT), capture_output=True, text=True)
    marker = "skipping completed training"
    if marker not in output.stdout:
        raise SystemExit(
            "a finished training was not skipped; --resume-if-available would retrain on the cluster:\n"
            + output.stdout[-4000:] + output.stderr[-4000:]
        )
    log("training is skipped when final/ exists, as the cluster job relies on")


def write_models_config(box: Sandbox) -> None:
    box.models_config.write_text(json.dumps([{
        "name": str(box.final_model),
        "type": "splade",
        "fp16": True,
        "max_seq_length": 128,
        # Small batches keep the encoding long enough to interrupt it mid-corpus,
        # and a manifest every batch makes the resume point precise.
        "batch_size": 8,
        "manifest_flush_batches": 1,
        "threads": 4,
    }], indent=2), encoding="utf-8")


def benchmark_command(box: Sandbox, results_json: Path, cache_dir: Path) -> List[str]:
    return [
        str(box.python), "run_benchmark.py",
        "--models_config", str(box.models_config),
        "--benchmark_config", str(box.benchmark_config),
        "--data_dir", str(box.data_dir),
        "--cache_dir", str(cache_dir),
        "--results_json", str(results_json),
        "--scope", "full",
    ]


def start_benchmark(box: Sandbox, name: str, results_json: Path,
                    cache_dir: Path) -> subprocess.Popen:
    box.logs.mkdir(parents=True, exist_ok=True)
    log_path = box.logs / f"{name}.log"
    handle = open(log_path, "w", encoding="utf-8")
    log(f"starting benchmark run '{name}', log: {log_path}")
    # Its own process group: SIGKILL then reaches the Java threads too, the way
    # the batch scheduler kills a whole job.
    return subprocess.Popen(
        benchmark_command(box, results_json, cache_dir),
        cwd=str(PIRB_ROOT), env=box.env(), stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def kill_group(process: subprocess.Popen) -> None:
    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    process.wait()
    log(f"SIGKILL delivered, exit status {process.returncode}")


def wait_until(process: subprocess.Popen, condition: Callable[[], bool], description: str,
               timeout: float = 1800) -> None:
    """Poll the artifacts until the run reaches the state we want to kill in."""
    log(f"waiting until {description}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            log(f"reached: {description}")
            return
        if process.poll() is not None:
            raise SystemExit(f"the run finished before {description}; see the log in {process.args}")
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting until {description}")


def read_manifest(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def count_lines(path: Path) -> int:
    with open(path, "rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b""))


def partially_encoded(box: Sandbox, task: str, at_least: int) -> bool:
    manifest = read_manifest(box.task_paths(task)["manifest"])
    return bool(manifest) and not manifest["complete"] and manifest["written_doc_count"] >= at_least


def encoded_but_not_indexed(box: Sandbox, task: str) -> bool:
    manifest = read_manifest(box.task_paths(task)["manifest"])
    return bool(manifest) and manifest["complete"] and not box.task_paths(task)["index"].is_dir()


def results_exist(box: Sandbox, task_id: str, cache_dir: Optional[Path] = None) -> bool:
    directory = box.task_paths(task_id, cache_dir)["results"]
    return bool(glob.glob(str(directory / f"{task_id}_k*.jsonl.gz")))


def describe_state(box: Sandbox, heading: str) -> Dict[str, Dict]:
    """What is on disk for each task, as the next run will see it."""
    state = {}
    log(f"--- {heading} ---")
    for task_id in TASK_IDS:
        paths = box.task_paths(task_id)
        manifest = read_manifest(paths["manifest"])
        corpus_documents = count_lines(paths["corpus"])
        encoded = manifest["written_doc_count"] if manifest else (
            count_lines(paths["passages"]) if paths["passages"].exists() else 0
        )
        entry = {
            "corpus_documents": corpus_documents,
            "encoded_documents": encoded,
            "encoding_complete": bool(manifest and manifest["complete"]),
            "has_manifest": manifest is not None,
            "index_present": paths["index"].is_dir(),
            "index_manifest": read_manifest(paths["index"] / ".index_manifest.json") is not None,
            "results_cached": results_exist(box, task_id),
        }
        state[task_id] = entry
        log(f"{task_id:32} encoded {entry['encoded_documents']:>5}/{entry['corpus_documents']:<5} "
            f"complete={str(entry['encoding_complete']):5} index={str(entry['index_present']):5} "
            f"results={str(entry['results_cached']):5}")
    return state


def phase_a_decisions(box: Sandbox, log_name: str) -> List[Dict[str, str]]:
    """Every phase A decision in a run's log, in order.

    Keyed by cache directory rather than task: the PolEval splits share one, and
    the point of that pairing is that only the first of them encodes anything.
    """
    decisions = []
    log_path = box.logs / f"{log_name}.log"
    if not log_path.exists():
        return decisions
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Splade encoding for" not in line:
            continue
        _, _, tail = line.partition("Splade encoding for ")
        directory, _, verdict = tail.partition(": ")
        for task in TASKS:
            if f"/{task['cache']}/" in directory:
                decisions.append({"cache": task["cache"], "verdict": verdict.strip()})
                break
    return decisions


def phase_run(box: Sandbox, keep_cache: bool) -> Dict:
    """Interrupt the benchmark twice, then let it finish."""
    write_models_config(box)
    if not keep_cache:
        shutil.rmtree(box.cache_dir, ignore_errors=True)
        shutil.rmtree(box.control_cache_dir, ignore_errors=True)
        box.results_resumed.unlink(missing_ok=True)
        box.results_control.unlink(missing_ok=True)

    second_task, third_task = KILL_TASKS
    documents = count_lines(box.task_paths(second_task)["corpus"])

    # Kill 1: the first task is behind us, the second is halfway through the GPU
    # phase - the expensive state to lose.
    process = start_benchmark(box, "run1-kill-in-encoding", box.results_resumed, box.cache_dir)
    wait_until(process, lambda: partially_encoded(box, second_task, max(8, documents // 3)),
               f"{second_task} has a third of its corpus encoded")
    kill_group(process)
    after_first_kill = describe_state(box, "after the kill during encoding")

    # Kill 2: the third task has finished encoding and Anserini is running, which
    # is where the old code left a directory that looked like a finished index.
    process = start_benchmark(box, "run2-kill-in-indexing", box.results_resumed, box.cache_dir)
    wait_until(process, lambda: encoded_but_not_indexed(box, third_task),
               f"{third_task} is encoded and its index is being built")
    kill_group(process)
    after_second_kill = describe_state(box, "after the kill during indexing")

    # And now the actual point of the exercise.
    process = start_benchmark(box, "run3-finish", box.results_resumed, box.cache_dir)
    if process.wait() != 0:
        raise SystemExit(f"the resuming run failed; see {box.logs / 'run3-finish.log'}")
    final_state = describe_state(box, "after the run that finished")

    log("control run on an empty cache, for the metric comparison")
    process = start_benchmark(box, "run4-control", box.results_control, box.control_cache_dir)
    if process.wait() != 0:
        raise SystemExit(f"the control run failed; see {box.logs / 'run4-control.log'}")

    return {
        "after_first_kill": after_first_kill,
        "after_second_kill": after_second_kill,
        "final_state": final_state,
        "decisions": {name: phase_a_decisions(box, name)
                      for name in ("run1-kill-in-encoding", "run2-kill-in-indexing", "run3-finish")},
    }


def flatten_metrics(value, prefix: str = "") -> Dict[str, float]:
    metrics = {}
    if isinstance(value, dict):
        for key, item in value.items():
            metrics.update(flatten_metrics(item, f"{prefix}{key}."))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            metrics.update(flatten_metrics(item, f"{prefix}{index}."))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        metrics[prefix.rstrip(".")] = float(value)
    return metrics


def phase_verify(box: Sandbox, observations: Optional[Dict]) -> None:
    """Check the claims this whole branch rests on."""
    failures: List[str] = []
    resumed = flatten_metrics(json.loads(box.results_resumed.read_text(encoding="utf-8")))
    control = flatten_metrics(json.loads(box.results_control.read_text(encoding="utf-8")))

    log("--- verification ---")
    if not resumed:
        failures.append("the resuming run produced no metrics")
    for key, value in sorted(resumed.items()):
        if key not in control:
            failures.append(f"metric {key} is missing from the control run")
            continue
        delta = abs(value - control[key])
        verdict = "same" if delta <= RESULT_TOLERANCE else f"DIFFERS by {delta:.6g}"
        log(f"{key:52} resumed={value:<12.6g} control={control[key]:<12.6g} {verdict}")
        if delta > RESULT_TOLERANCE:
            failures.append(f"metric {key} differs by {delta:.6g} after interruptions")

    if observations:
        # A task counts as interrupted when work was done on it and no results were
        # written. The union of both kills is what matters: the second kill lands
        # after the first two tasks have been carried to completion.
        def unfinished(state: Dict[str, Dict]) -> set:
            return {task for task, entry in state.items()
                    if not entry["results_cached"] and (entry["encoded_documents"] > 0 or entry["index_present"])}

        first = unfinished(observations["after_first_kill"])
        second = unfinished(observations["after_second_kill"])
        log(f"unfinished after the first kill: {sorted(first) or 'none'}")
        log(f"unfinished after the second kill: {sorted(second) or 'none'}")
        if len(first | second) < 2:
            failures.append(f"the scenario needs two interrupted tasks, got {sorted(first | second)}")
        for task, state in observations["final_state"].items():
            if not state["encoding_complete"] or not state["index_present"]:
                failures.append(f"{task} is still incomplete after the finishing run")

        every_decision = [decision for run in observations["decisions"].values() for decision in run]
        for run_name, decisions in observations["decisions"].items():
            for decision in decisions:
                log(f"{run_name:22} {decision['cache']:26} {decision['verdict']}")
        if not any(decision["verdict"].startswith("RESUME") for decision in every_decision):
            failures.append("no task resumed a partial corpus; the kills did not land in phase A")
        if not any(decision["verdict"].startswith("REUSE") for decision in every_decision):
            failures.append("no task reused a finished corpus; the kills did not land in phase B")

        # Several PIRB tasks share one cache directory over one corpus - all seven
        # PolEval tasks do. Only the first of them may encode it.
        shared = [decision for decision in every_decision if decision["cache"] == SHARED_CACHE]
        log(f"phase A executions for the shared cache {SHARED_CACHE}: {len(shared)} "
            f"({[decision['verdict'].split(' ')[0] for decision in shared]})")
        encodings = [decision for decision in shared if decision["verdict"].startswith("REBUILD")]
        if len(encodings) > 1:
            failures.append(
                f"the corpus of {SHARED_CACHE} was encoded {len(encodings)} times, once per split"
            )

    if failures:
        log("FAILED:")
        for failure in failures:
            log(f"  - {failure}")
        raise SystemExit(1)
    log("PASSED: interrupted and uninterrupted runs agree, and the work was resumed, not redone")


PHASES = ("env", "data", "train", "skip", "run", "verify")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(Path.home() / "pirb-resume-e2e"),
                        help="Where the JDK, virtualenv, data, caches and logs go")
    parser.add_argument("--phases", default=",".join(PHASES),
                        help=f"Comma separated subset of: {', '.join(PHASES)}")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Do not wipe the index cache before the run phase")
    args = parser.parse_args(argv)

    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    unknown = set(phases) - set(PHASES)
    if unknown:
        raise SystemExit(f"unknown phase(s): {', '.join(sorted(unknown))}")

    box = Sandbox(Path(args.root).expanduser().resolve())
    log(f"sandbox: {box.root}")
    started = time.time()
    observations = None
    observations_path = box.root / "observations.json"

    if "env" in phases:
        phase_env(box)
    if "data" in phases:
        phase_data(box)
    if "train" in phases:
        phase_train(box)
    if "skip" in phases:
        phase_skip(box)
    if "run" in phases:
        observations = phase_run(box, keep_cache=args.keep_cache)
        observations_path.write_text(json.dumps(observations, indent=2), encoding="utf-8")
    elif observations_path.exists():
        observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if "verify" in phases:
        phase_verify(box, observations)

    log(f"done in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
