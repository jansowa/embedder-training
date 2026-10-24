#!/usr/bin/env python
"""Compare NDCG@10 across benchmark metrics.json files and leaderboard columns.

The inputs are the `metrics.json` files written by the post-training benchmark
and by `run_experiments.py --mode benchmark --benchmark-output-dir ...`. Only the
per-task `pirb_<task_id>` keys are read; aggregates and metadata are ignored.

Every table is built on the intersection of the tasks that all compared columns
provide. Means are never computed over different task sets, because a mean over
a different selection of tasks is not comparable with another one.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import glob
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


NDCG_METRIC = "NDCG@10"
METRICS_FILENAME = "metrics.json"
PIRB_METRIC_PREFIX = "pirb_"
# Keys that share the 'pirb_' prefix but describe the run instead of one task.
PIRB_AGGREGATE_KEYS = frozenset({"model", "ndcg_tasks", "datasets", "task_selection", "queries"})
PIRB_AGGREGATE_PREFIXES = ("average_ndcg@",)
DEFAULT_LEADERBOARD_MODELS = ("./models/polish-splade", "./models/mmlw/mmlw-retrieval-roberta-base")

# Group names follow the PIRB leaderboard. When no leaderboard file is given,
# the group of a task is derived from the task id instead.
BEIR_TASK_IDS = frozenset(
    {
        "arguana-pl",
        "dbpedia-pl",
        "fiqa-pl",
        "hotpotqa-pl",
        "msmarco-pl",
        "nfcorpus-pl",
        "nq-pl",
        "quora-pl",
        "scidocs-pl",
        "scifact-pl",
        "trec-covid-pl",
    }
)
WEB_TASK_IDS = frozenset(
    {"abczdrowie", "eprawnik", "gemini", "odi", "onet", "pwn", "specprawnik", "techpedia", "zapytajfizyka"}
)
GROUP_ORDER = ("MaupQA", "PolEval-2022", "BEIR-PL", "Web Datasets", "Other")


class CompareResultsError(ValueError):
    """Raised when the requested comparison cannot be built."""


@dataclass(frozen=True)
class Column:
    name: str
    model: str
    source: str
    scores: dict[str, float]


@dataclass
class ComparisonRow:
    kind: str
    name: str
    group: str
    tasks: int
    values: dict[str, float | None] = field(default_factory=dict)


def task_group(task_id: str, groups: dict[str, str] | None = None) -> str:
    if groups and task_id in groups:
        return groups[task_id]
    if task_id.startswith("maupqa-"):
        return "MaupQA"
    if task_id.startswith("poleval-2022-"):
        return "PolEval-2022"
    if task_id in BEIR_TASK_IDS:
        return "BEIR-PL"
    if task_id in WEB_TASK_IDS:
        return "Web Datasets"
    return "Other"


def pirb_task_id(metric_key: str) -> str | None:
    """Task id behind a `pirb_<task_id>` metric key, or None for other keys.

    Benchmark metrics are prefixed with the checkpoint label, so the `pirb_`
    marker is looked up in the last path segment of the key.
    """
    name = metric_key.rsplit("/", 1)[-1]
    if not name.startswith(PIRB_METRIC_PREFIX):
        return None
    task_id = name[len(PIRB_METRIC_PREFIX) :]
    if not task_id or task_id in PIRB_AGGREGATE_KEYS:
        return None
    if any(task_id.startswith(prefix) for prefix in PIRB_AGGREGATE_PREFIXES):
        return None
    return task_id


def pirb_task_scores(metrics: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, value in metrics.items():
        task_id = pirb_task_id(str(key))
        if task_id is None or isinstance(value, (list, dict, str, bool)) or value is None:
            continue
        scores[task_id] = float(value)
    return scores


def expand_metrics_paths(patterns: Sequence[str]) -> list[Path]:
    """Resolve CLI arguments that may be files, directories, or glob patterns."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = [Path(match) for match in sorted(glob.glob(pattern, recursive=True))]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        if not matches:
            raise CompareResultsError(f"No metrics files matched '{pattern}'.")
        for match in matches:
            candidates = sorted(match.rglob(METRICS_FILENAME)) if match.is_dir() else [match]
            if match.is_dir() and not candidates:
                raise CompareResultsError(f"Directory '{match}' contains no {METRICS_FILENAME}.")
            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                paths.append(candidate)
    return paths


def default_column_label(path: Path) -> str:
    return path.parent.name or path.stem


def _unique_name(preferred: str, path: Path, used: set[str]) -> str:
    if preferred not in used:
        return preferred
    qualified = f"{path.parent.parent.name}/{preferred}" if path.parent.parent.name else preferred
    if qualified not in used:
        return qualified
    index = 2
    while f"{preferred}#{index}" in used:
        index += 1
    return f"{preferred}#{index}"


def load_metrics_column(path: Path, used_names: set[str], *, label: str | None = None) -> Column:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CompareResultsError(f"Could not read '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CompareResultsError(f"'{path}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CompareResultsError(f"'{path}' does not look like a {METRICS_FILENAME} file.")

    scores = pirb_task_scores(payload)
    if not scores:
        raise CompareResultsError(f"'{path}' has no per-task 'pirb_<task_id>' metrics.")

    name = _unique_name(label or default_column_label(path), path, used_names)
    used_names.add(name)
    model = str(payload.get("model") or payload.get("pirb_model") or "")
    return Column(name=name, model=model, source=str(path), scores=scores)


def load_leaderboard(
    path: Path,
    model_ids: Sequence[str],
    used_names: set[str],
) -> tuple[list[Column], dict[str, str]]:
    """Load reference columns and the task group mapping from a PIRB leaderboard file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CompareResultsError(f"Could not read the leaderboard '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CompareResultsError(f"Leaderboard '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise CompareResultsError(f"Leaderboard '{path}' must contain a 'results' list.")

    groups = {
        str(task["id"]): str(task.get("groupId", "Other"))
        for task in payload.get("tasks", [])
        if isinstance(task, dict) and task.get("id")
    }
    by_id = {
        str(entry["id"]): entry
        for entry in payload["results"]
        if isinstance(entry, dict) and entry.get("id")
    }

    columns: list[Column] = []
    for model_id in model_ids:
        entry = by_id.get(model_id)
        if entry is None:
            print(f"[WARN] Leaderboard '{path}' has no model '{model_id}'; skipping that column.", file=sys.stderr)
            continue
        scores = {
            str(task_id): float(metrics[NDCG_METRIC])
            for task_id, metrics in (entry.get("results") or {}).items()
            if isinstance(metrics, dict) and NDCG_METRIC in metrics
        }
        if not scores:
            print(f"[WARN] Leaderboard model '{model_id}' has no {NDCG_METRIC} scores; skipping.", file=sys.stderr)
            continue
        name = _unique_name(model_id, path, used_names)
        used_names.add(name)
        columns.append(Column(name=name, model=model_id, source=f"{path} ({model_id})", scores=scores))

    if not columns:
        raise CompareResultsError(f"Leaderboard '{path}' provided none of the requested models: {', '.join(model_ids)}.")
    return columns, groups


def shared_tasks(columns: Sequence[Column]) -> tuple[list[str], list[str]]:
    """Tasks every column scored, plus the tasks that had to be dropped."""
    if not columns:
        return [], []
    shared = set(columns[0].scores)
    union: set[str] = set()
    for column in columns:
        shared &= set(column.scores)
        union |= set(column.scores)
    return sorted(shared), sorted(union - shared)


def sorted_tasks(task_ids: Iterable[str], groups: dict[str, str] | None) -> list[str]:
    order = {name: index for index, name in enumerate(GROUP_ORDER)}
    return sorted(
        task_ids,
        key=lambda task_id: (order.get(task_group(task_id, groups), len(order)), task_group(task_id, groups), task_id),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_rows(
    columns: Sequence[Column],
    tasks: Sequence[str],
    *,
    reference: Column,
    groups: dict[str, str] | None = None,
) -> list[ComparisonRow]:
    """Build the per-task rows plus the mean, per-group mean, and delta rows."""
    rows = [
        ComparisonRow(
            kind="task",
            name=task_id,
            group=task_group(task_id, groups),
            tasks=1,
            values={column.name: column.scores[task_id] for column in columns},
        )
        for task_id in tasks
    ]

    means = {column.name: _mean([column.scores[task_id] for task_id in tasks]) for column in columns}
    rows.append(ComparisonRow(kind="mean", name="mean", group="", tasks=len(tasks), values=means))

    grouped: dict[str, list[str]] = {}
    for task_id in tasks:
        grouped.setdefault(task_group(task_id, groups), []).append(task_id)
    for group in sorted(
        grouped,
        key=lambda name: (GROUP_ORDER.index(name) if name in GROUP_ORDER else len(GROUP_ORDER), name),
    ):
        group_tasks = grouped[group]
        rows.append(
            ComparisonRow(
                kind="group_mean",
                name="group mean",
                group=group,
                tasks=len(group_tasks),
                values={
                    column.name: _mean([column.scores[task_id] for task_id in group_tasks]) for column in columns
                },
            )
        )

    rows.append(
        ComparisonRow(
            kind="delta",
            name=f"delta vs {reference.name}",
            group="",
            tasks=len(tasks),
            values={column.name: means[column.name] - means[reference.name] for column in columns},
        )
    )
    return rows


def _format_value(row: ComparisonRow, value: float | None) -> str:
    if value is None:
        return "-"
    if row.kind == "delta":
        return f"{value:+.2f}"
    return f"{value:.2f}"


def render_markdown(
    columns: Sequence[Column],
    rows: Sequence[ComparisonRow],
    *,
    tasks: Sequence[str],
    reference: Column,
    skipped: Sequence[str],
) -> str:
    out = io.StringIO()
    out.write(f"# PIRB comparison ({NDCG_METRIC})\n\n")
    out.write(f"Tasks in the intersection: {len(tasks)}\n\n")
    out.write(f"Reference column: `{reference.name}`\n\n")
    if skipped:
        out.write(f"Skipped tasks (missing in at least one column): {', '.join(skipped)}\n\n")

    headers = ["row", "group", "tasks", *[column.name for column in columns]]
    out.write("| " + " | ".join(headers) + " |\n")
    out.write("| " + " | ".join(["---", "---", "---:", *["---:" for _ in columns]]) + " |\n")
    for row in rows:
        cells = [row.name, row.group, str(row.tasks)]
        cells.extend(_format_value(row, row.values.get(column.name)) for column in columns)
        out.write("| " + " | ".join(cells) + " |\n")

    out.write("\n## Columns\n\n")
    out.write("| column | model | source |\n| --- | --- | --- |\n")
    for column in columns:
        out.write(f"| {column.name} | {column.model} | {column.source} |\n")
    return out.getvalue()


def render_csv(columns: Sequence[Column], rows: Sequence[ComparisonRow]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["kind", "row", "group", "tasks", *[column.name for column in columns]])
    for row in rows:
        writer.writerow(
            [
                row.kind,
                row.name,
                row.group,
                row.tasks,
                *[_format_value(row, row.values.get(column.name)) for column in columns],
            ]
        )
    return out.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "metrics",
        nargs="+",
        metavar="PATH",
        help=(
            f"{METRICS_FILENAME} files, directories to search recursively, or glob patterns. "
            "Quote a pattern to let the script expand it."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Column labels, one per metrics file, in input order. Defaults to each file's parent directory name.",
    )
    parser.add_argument("--leaderboard", type=Path, default=None, help="PIRB leaderboard data.json.")
    parser.add_argument(
        "--leaderboard-models",
        nargs="+",
        default=list(DEFAULT_LEADERBOARD_MODELS),
        help="Leaderboard model ids added as reference columns.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help=(
            "Column the delta row is computed against. Defaults to the first leaderboard column, "
            "or to the first metrics column when no leaderboard is given."
        ),
    )
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown", help="Output format.")
    parser.add_argument("--output", type=Path, default=None, help="Write the table to this file instead of stdout.")
    return parser


def build_comparison(args: argparse.Namespace) -> tuple[str, int]:
    """Return the rendered table and the number of compared tasks."""
    used_names: set[str] = set()
    leaderboard_columns: list[Column] = []
    groups: dict[str, str] | None = None
    if args.leaderboard is not None:
        leaderboard_columns, groups = load_leaderboard(Path(args.leaderboard), args.leaderboard_models, used_names)

    paths = expand_metrics_paths(args.metrics)
    labels = args.labels
    if labels is not None and len(labels) != len(paths):
        raise CompareResultsError(f"--labels has {len(labels)} values but {len(paths)} metrics file(s) were given.")
    metrics_columns = [
        load_metrics_column(path, used_names, label=labels[index] if labels else None)
        for index, path in enumerate(paths)
    ]

    columns = [*leaderboard_columns, *metrics_columns]
    reference = columns[0]
    if args.reference is not None:
        matching = [column for column in columns if column.name == args.reference]
        if not matching:
            available = ", ".join(column.name for column in columns)
            raise CompareResultsError(f"Unknown --reference '{args.reference}'. Available columns: {available}.")
        reference = matching[0]

    shared, skipped = shared_tasks(columns)
    if skipped:
        print(
            "[WARN] Comparing on the intersection of "
            f"{len(shared)} tasks; {len(skipped)} task(s) are missing in at least one column: "
            f"{', '.join(skipped)}.",
            file=sys.stderr,
        )
    if not shared:
        raise CompareResultsError("The compared columns share no task; there is nothing to compare.")

    tasks = sorted_tasks(shared, groups)
    rows = build_rows(columns, tasks, reference=reference, groups=groups)
    if args.format == "csv":
        return render_csv(columns, rows), len(tasks)
    return render_markdown(columns, rows, tasks=tasks, reference=reference, skipped=skipped), len(tasks)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rendered, _ = build_comparison(args)
    except CompareResultsError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"[INFO] Wrote the comparison to: {args.output}", flush=True)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
