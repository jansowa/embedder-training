#!/usr/bin/env python
"""Run the PIRB benchmark for one model without training it first.

The benchmark itself is driven by the shared helpers in ``training.benchmarks``,
so a standalone run behaves exactly like a post-training run and writes the same
portable ``pirb_results.json`` summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.benchmarks import (  # noqa: E402
    DEFAULT_BENCHMARK_BATCH_SIZE,
    DEFAULT_BENCHMARK_NAME,
    DEFAULT_PIRB_MAX_SEQ_LENGTH,
    DEFAULT_PIRB_RESULTS_DIR,
    BenchmarkSettings,
    BenchmarkTarget,
    run_benchmarks_for_targets,
    safe_label,
)


DEFAULT_BENCHMARK_CONFIG = REPO_ROOT / "third_party" / "pirb" / "config" / "benchmarks" / "pirb-without-private.json"

# "tiny" reuses PIRB's own size-based scope. "quick6" is a fixed, cheap spread
# over three task groups, useful for comparing checkpoints during a weekend run.
QUICK6_TASK_IDS = (
    "arguana-pl",
    "maupqa-mkqa",
    "maupqa-poleval2021-pairs",
    "poleval-2022-test-A-allegro-faq",
    "poleval-2022-test-A-legal-questions",
    "scifact-pl",
)


class BenchmarkModelError(ValueError):
    """Raised when a standalone benchmark cannot be configured."""


def maupqa_task_ids(benchmark_config: Path = DEFAULT_BENCHMARK_CONFIG) -> list[str]:
    """MaupQA task ids of the PIRB benchmark config, in configuration order."""
    try:
        entries = json.loads(benchmark_config.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkModelError(f"Could not read the PIRB benchmark config '{benchmark_config}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkModelError(f"PIRB benchmark config '{benchmark_config}' is not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise BenchmarkModelError(f"PIRB benchmark config '{benchmark_config}' must contain a list of tasks.")
    return [
        f"maupqa-{entry['subset']}"
        for entry in entries
        if isinstance(entry, dict) and entry.get("type") == "maupqa" and entry.get("subset")
    ]


def resolve_tasks(tasks: str, *, benchmark_config: Path = DEFAULT_BENCHMARK_CONFIG) -> str:
    """Turn a preset name or a task list into a PIRB scope string."""
    normalized = tasks.strip()
    if not normalized:
        raise BenchmarkModelError("--tasks must be a preset name or a comma-separated list of PIRB task ids.")
    if normalized == "tiny":
        return "tiny"
    if normalized == "quick6":
        return ",".join(QUICK6_TASK_IDS)
    if normalized == "maupqa":
        task_ids = maupqa_task_ids(benchmark_config)
        if not task_ids:
            raise BenchmarkModelError(f"PIRB benchmark config '{benchmark_config}' defines no MaupQA tasks.")
        return ",".join(task_ids)

    selected = [task_id.strip() for task_id in normalized.split(",") if task_id.strip()]
    if not selected:
        raise BenchmarkModelError("--tasks must be a preset name or a comma-separated list of PIRB task ids.")
    return ",".join(selected)


def default_run_name(model: str) -> str:
    stem = model.rstrip("/\\").replace("\\", "/").split("/")[-1] or model
    return safe_label(stem)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model directory.")
    parser.add_argument(
        "--tasks",
        default="tiny",
        help=(
            "Preset name ('tiny', 'quick6', 'maupqa') or a comma-separated list of PIRB task ids. "
            "PIRB's own 'small' and 'full' scopes work as well."
        ),
    )
    parser.add_argument("--run-name", default=None, help="Run name. Defaults to the last path segment of --model.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_PIRB_RESULTS_DIR,
        help=f"Directory receiving <run-name>/pirb_results.json. Defaults to {DEFAULT_PIRB_RESULTS_DIR}/.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_PIRB_MAX_SEQ_LENGTH,
        help=f"Maximum sequence length used by PIRB. Defaults to {DEFAULT_PIRB_MAX_SEQ_LENGTH}.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode in float16, matching the post-training benchmark default.",
    )
    parser.add_argument(
        "--query-instruction",
        default="",
        help="Query prefix passed to PIRB, for example 'zapytanie: ' for some dense models.",
    )
    parser.add_argument("--pirb-threads", type=int, default=None, help="Lucene/Anserini threads for one PIRB worker.")
    parser.add_argument("--pirb-batch-size", type=int, default=None, help="Encoding batch size used by PIRB.")
    parser.add_argument(
        "--parallel-pirb-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Spread PIRB task groups over the visible GPUs.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=DEFAULT_BENCHMARK_CONFIG,
        help="PIRB benchmark config used to expand the 'maupqa' preset.",
    )
    return parser


def build_target(model: str, label: str) -> BenchmarkTarget:
    """Keep local models as absolute paths and Hugging Face ids as plain names.

    PIRB runs with its own working directory, so a relative local path would not
    resolve there, while a model id must not be turned into a path at all.
    """
    path = Path(model)
    if path.exists():
        return BenchmarkTarget(label=label, path=path)
    return BenchmarkTarget(label=label, path=path, model=model)


def build_settings(args: argparse.Namespace, *, scope: str, run_name: str) -> BenchmarkSettings:
    return BenchmarkSettings(
        run_mteb=False,
        run_pirb=True,
        benchmark_name=DEFAULT_BENCHMARK_NAME,
        pirb_scope=scope,
        batch_size=DEFAULT_BENCHMARK_BATCH_SIZE,
        pirb_max_seq_length=int(args.max_seq_length),
        query_instruction_for_retrieval=str(args.query_instruction or ""),
        output_dir=Path(args.results_dir) / run_name,
        log_to_wandb=False,
        parallel_checkpoints=False,
        parallel_pirb_tasks=bool(args.parallel_pirb_tasks),
        pirb_threads=args.pirb_threads,
        pirb_batch_size=args.pirb_batch_size,
        pirb_fp16=bool(args.fp16),
        pirb_results_dir=Path(args.results_dir),
        pirb_results_group_by_label=False,
        run_name=run_name,
        tasks_preset=str(args.tasks),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_seq_length <= 0:
        print("[ERROR] --max-seq-length must be positive.", file=sys.stderr)
        return 2

    try:
        scope = resolve_tasks(str(args.tasks), benchmark_config=Path(args.benchmark_config))
    except BenchmarkModelError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    run_name = safe_label(args.run_name) if args.run_name else default_run_name(str(args.model))
    settings = build_settings(args, scope=scope, run_name=run_name)
    target = build_target(str(args.model), run_name)

    print(f"[INFO] Benchmarking {target.model_ref} on PIRB scope '{scope}'.", flush=True)
    metrics = run_benchmarks_for_targets([target], settings)
    for key, value in sorted(metrics.get(target.label, {}).items()):
        if "average_ndcg" in key:
            print(f"[INFO] {key}: {float(value):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
