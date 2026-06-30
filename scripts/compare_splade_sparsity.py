#!/usr/bin/env python
"""Compare SPLADE document sparsity on PIRB passages.

The script mirrors PIRB document encoding:
relu(logits) -> log1p -> max over sequence -> round(weight * qf) > 0.
It does not build a Lucene index and does not import PIRB/pyserini.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_TINY_THRESHOLD = 20_971_520
DEFAULT_SMALL_THRESHOLD = 104_857_600


@dataclass(frozen=True)
class ModelSpec:
    label: str
    name_or_path: str


@dataclass(frozen=True)
class TaskSpec:
    label: str
    passages_path: Path
    queries_path: Path | None = None


@dataclass
class RunningStats:
    values: list[int] = field(default_factory=list)

    def extend(self, batch_values: Iterable[int]) -> None:
        self.values.extend(int(value) for value in batch_values)

    @property
    def count(self) -> int:
        return len(self.values)

    def as_dict(self) -> dict[str, float | int | None]:
        if not self.values:
            return {
                "docs": 0,
                "mean": None,
                "median": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "max": None,
            }
        ordered = sorted(self.values)
        return {
            "docs": len(ordered),
            "mean": sum(ordered) / len(ordered),
            "median": percentile(ordered, 0.50),
            "p90": percentile(ordered, 0.90),
            "p95": percentile(ordered, 0.95),
            "p99": percentile(ordered, 0.99),
            "max": ordered[-1],
        }


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def parse_model_spec(value: str) -> ModelSpec:
    if "=" in value:
        label, name_or_path = value.split("=", 1)
        label = label.strip()
        name_or_path = name_or_path.strip()
    else:
        name_or_path = value.strip()
        label = name_or_path.rstrip("/\\").replace("\\", "/").split("/")[-1]
    if not label or not name_or_path:
        raise argparse.ArgumentTypeError(f"Invalid model spec: {value!r}")
    return ModelSpec(label=label, name_or_path=name_or_path)


def default_models(repo_root: Path) -> list[ModelSpec]:
    models = [ModelSpec("sdadas", "sdadas/polish-splade")]
    local = (
        repo_root
        / "runs"
        / "polish-splade-dataset-small-lr2e-6"
        / "sentence-transformers"
        / "splade"
        / "sdadas_polish-distilroberta-lr-2e-6-ep-1"
        / "final"
    )
    if local.exists():
        models.append(ModelSpec("local", str(local)))
    return models


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare SPLADE active document dimensions on PIRB passages."
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        help=(
            "Model to compare. Use LABEL=HF_OR_PATH. Can be repeated. "
            "Defaults to sdadas/polish-splade and the local small checkpoint if present."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "third_party" / "pirb" / "data",
        help="PIRB data directory.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=repo_root / "third_party" / "pirb" / "config" / "benchmarks" / "pirb-without-private.json",
        help="PIRB benchmark JSON used to derive tasks for --scope.",
    )
    parser.add_argument(
        "--scope",
        choices=("tiny", "small", "full"),
        default="tiny",
        help="Benchmark scope. Uses PIRB file-size thresholds for tiny/small.",
    )
    parser.add_argument(
        "--tasks",
        help=(
            "Comma-separated task labels to use instead of --scope, e.g. "
            "maupqa-gpt3-cc,maupqa-mkqa,scifact-pl."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--quantization-factor", type=int, default=100)
    parser.add_argument(
        "--max-docs-per-task",
        type=int,
        default=None,
        help="Optional quick mode: process only the first N passages per task.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device used for encoding.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use float16 on CUDA. CPU always uses float32.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for machine-readable results.",
    )
    args = parser.parse_args()
    args.repo_root = repo_root
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_seq_length <= 0:
        parser.error("--max-seq-length must be positive")
    if args.quantization_factor <= 0:
        parser.error("--quantization-factor must be positive")
    if args.max_docs_per_task is not None and args.max_docs_per_task <= 0:
        parser.error("--max-docs-per-task must be positive when set")
    if not args.model:
        args.model = default_models(repo_root)
    return args


def task_from_benchmark_entry(entry: dict, data_dir: Path) -> TaskSpec | None:
    entry_type = entry.get("type")
    if entry_type == "maupqa":
        label = f"maupqa-{entry['subset']}"
        return default_task(label, data_dir)
    if entry_type == "beir":
        return default_task(entry["task_id"], data_dir)
    if entry_type == "mfaq":
        return default_task("mfaq", data_dir)
    if entry_type == "gpt-exams":
        return default_task("gpt-exams", data_dir)
    if entry_type == "poleval":
        domain = entry["domain"]
        split = entry["split"]
        return TaskSpec(
            label=f"poleval-2022-{domain}",
            passages_path=data_dir / "poleval-2022" / domain / "passages" / "passages.jsonl",
            queries_path=data_dir / "poleval-2022" / domain / split / "queries.jsonl",
        )
    return None


def default_task(label: str, data_dir: Path) -> TaskSpec:
    return TaskSpec(
        label=label,
        passages_path=data_dir / label / "passages" / "passages.jsonl",
        queries_path=data_dir / label / "queries" / "queries.jsonl",
    )


def load_tasks(args: argparse.Namespace) -> list[TaskSpec]:
    if args.tasks:
        tasks = []
        for label in args.tasks.split(","):
            label = label.strip()
            if not label:
                continue
            if label.startswith("poleval-2022-"):
                domain = label.removeprefix("poleval-2022-")
                tasks.append(
                    TaskSpec(
                        label=label,
                        passages_path=args.data_dir / "poleval-2022" / domain / "passages" / "passages.jsonl",
                    )
                )
            else:
                tasks.append(default_task(label, args.data_dir))
        return filter_existing_tasks(tasks)

    with args.benchmark_config.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    tasks = []
    seen_paths = set()
    for entry in entries:
        task = task_from_benchmark_entry(entry, args.data_dir)
        if task is None:
            continue
        key = task.passages_path.resolve()
        if key in seen_paths:
            continue
        seen_paths.add(key)
        if not task.passages_path.exists():
            continue
        if args.scope != "full" and not is_in_scope(task, args.scope):
            continue
        tasks.append(task)
    return tasks


def filter_existing_tasks(tasks: list[TaskSpec]) -> list[TaskSpec]:
    missing = [task for task in tasks if not task.passages_path.exists()]
    for task in missing:
        print(f"Skipping missing task {task.label}: {task.passages_path}", file=sys.stderr)
    return [task for task in tasks if task.passages_path.exists()]


def is_in_scope(task: TaskSpec, scope: str) -> bool:
    threshold = DEFAULT_TINY_THRESHOLD if scope == "tiny" else DEFAULT_SMALL_THRESHOLD
    size = task.passages_path.stat().st_size
    if task.queries_path is not None and task.queries_path.exists():
        size += task.queries_path.stat().st_size
    return size <= threshold


def iter_passage_texts(path: Path, limit: int | None) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if not line.strip():
                continue
            yield json.loads(line)["contents"]


def batched(values: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class SpladeCounter:
    def __init__(self, model: ModelSpec, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if args.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args.device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

        self.torch = torch
        self.quantization_factor = args.quantization_factor
        self.max_seq_length = args.max_seq_length
        dtype = torch.float16 if args.fp16 and self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        self.tokenizer.model_max_length = self.max_seq_length
        self.model = AutoModelForMaskedLM.from_pretrained(model.name_or_path, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def count_batch(self, texts: list[str]) -> list[int]:
        import numpy as np

        torch = self.torch
        encoded = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.max_seq_length,
        ).to(self.device)
        with torch.inference_mode():
            output = self.model(**encoded)
            logits = output["logits"].detach()
            attention_mask = encoded["attention_mask"].detach().unsqueeze(-1)
            vectors = torch.max(torch.log1p(torch.relu(logits)) * attention_mask, dim=1).values
        dense = vectors.detach().float().cpu().numpy()
        del encoded, output, logits, attention_mask, vectors
        quantized = np.rint(dense * self.quantization_factor).astype(np.int64)
        return (quantized > 0).sum(axis=1).tolist()

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def format_number(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.1f}"


def print_table(rows: list[dict[str, object]]) -> None:
    headers = ["model", "task", "docs", "mean", "median", "p90", "p95", "p99", "max"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def main() -> int:
    args = parse_args()
    tasks = load_tasks(args)
    if not tasks:
        print("No tasks found. Check --data-dir, --scope, or --tasks.", file=sys.stderr)
        return 2

    print("Models:")
    for model in args.model:
        print(f"  {model.label}: {model.name_or_path}")
    print("Tasks:")
    for task in tasks:
        limit = "" if args.max_docs_per_task is None else f" (first {args.max_docs_per_task})"
        print(f"  {task.label}: {task.passages_path}{limit}")
    print()

    all_results: dict[str, dict[str, dict[str, float | int | None]]] = {}
    table_rows = []

    for model_spec in args.model:
        print(f"Loading model {model_spec.label}...", flush=True)
        counter = SpladeCounter(model_spec, args)
        model_results: dict[str, dict[str, float | int | None]] = {}
        overall = RunningStats()
        try:
            for task in tasks:
                print(f"Encoding {model_spec.label} on {task.label}...", flush=True)
                stats = RunningStats()
                texts = iter_passage_texts(task.passages_path, args.max_docs_per_task)
                for batch in batched(texts, args.batch_size):
                    counts = counter.count_batch(batch)
                    stats.extend(counts)
                    overall.extend(counts)
                task_result = stats.as_dict()
                model_results[task.label] = task_result
                table_rows.append(row_from_stats(model_spec.label, task.label, task_result))
        finally:
            counter.close()
        overall_result = overall.as_dict()
        model_results["ALL"] = overall_result
        table_rows.append(row_from_stats(model_spec.label, "ALL", overall_result))
        all_results[model_spec.label] = model_results

    print()
    print_table(table_rows)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON results to {args.output_json}")
    return 0


def row_from_stats(model: str, task: str, stats: dict[str, float | int | None]) -> dict[str, str]:
    return {
        "model": model,
        "task": task,
        "docs": format_number(stats["docs"]),
        "mean": format_number(stats["mean"]),
        "median": format_number(stats["median"]),
        "p90": format_number(stats["p90"]),
        "p95": format_number(stats["p95"]),
        "p99": format_number(stats["p99"]),
        "max": format_number(stats["max"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
