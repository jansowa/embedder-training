"""Utilities for training runs that combine multiple JSONL datasets."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Iterator

from training.dataset_sources import resolve_train_data_entry
from training.dataset_filters import apply_dataset_filter_if_configured
from training.distributed import is_main_process, is_torchrun_child, wait_for_files


CONCAT_SHUFFLE = "concat_shuffle"
PROPORTIONAL_BATCH_BEST_EFFORT = "proportional_batch_best_effort"

DATASET_MIX_STRATEGIES = {CONCAT_SHUFFLE, PROPORTIONAL_BATCH_BEST_EFFORT}
DATASET_MIX_STRATEGY_ALIASES = {
    "concat": CONCAT_SHUFFLE,
    "concat-shuffle": CONCAT_SHUFFLE,
    "concat_shuffle": CONCAT_SHUFFLE,
    "proportional": PROPORTIONAL_BATCH_BEST_EFFORT,
    "proportional-batch": PROPORTIONAL_BATCH_BEST_EFFORT,
    "proportional_batch": PROPORTIONAL_BATCH_BEST_EFFORT,
    "proportional-batch-best-effort": PROPORTIONAL_BATCH_BEST_EFFORT,
    "proportional_batch_best_effort": PROPORTIONAL_BATCH_BEST_EFFORT,
}

DEFAULT_MIXED_DATASET_CACHE_DIR = Path("cache/mixed_datasets")
DEFAULT_SAMPLED_DATASET_CACHE_DIR = Path("cache/sampled_datasets")

BEFORE_FILTER = "before_filter"
AFTER_FILTER = "after_filter"
SAMPLE_STAGES = (BEFORE_FILTER, AFTER_FILTER)
# Matches the training seed default in the SentenceTransformers backend, so an
# unset per-source seed follows the run instead of a separate constant.
DEFAULT_SAMPLING_SEED = 42
TRAIN_DATA_PATH_KEYS = ("path", "train_data")


class MultiDatasetConfigError(ValueError):
    """Raised when multi-dataset training configuration is invalid."""


@dataclass(frozen=True)
class DatasetSampling:
    """Deterministic per-source subsampling of a training JSONL file."""

    max_samples: int
    seed: int
    stage: str

    def as_cache_payload(self) -> dict[str, Any]:
        return {"max_samples": self.max_samples, "seed": self.seed, "sample_stage": self.stage}


@dataclass(frozen=True)
class TrainDataEntry:
    path: str
    sampling: DatasetSampling | None = None


@dataclass(frozen=True)
class SampledDatasetResult:
    input_path: Path
    output_path: Path
    total: int
    kept: int
    cache_hit: bool


@dataclass(frozen=True)
class DatasetSource:
    name: str
    original: str
    path: Path
    sampling: DatasetSampling | None = None


@dataclass(frozen=True)
class MixedDatasetResult:
    sources: list[DatasetSource]
    output_dir: Path
    output_path: Path
    cache_hit: bool


@dataclass
class LoadedTrainingRows:
    sources: list[DatasetSource]
    rows: list[dict[str, Any]]
    dataset_ids: list[str]
    dedupe_values: list[set[str]]

    @property
    def is_multi_source(self) -> bool:
        return len(self.sources) > 1

    @property
    def source_label(self) -> str:
        return describe_sources(self.sources)


def normalize_dataset_mix_strategy(value: Any) -> str:
    if value is None:
        return CONCAT_SHUFFLE
    normalized = str(value).strip().lower().replace(" ", "_")
    strategy = DATASET_MIX_STRATEGY_ALIASES.get(normalized)
    if strategy is None:
        supported = ", ".join(sorted(DATASET_MIX_STRATEGIES))
        raise MultiDatasetConfigError(f"'dataset_mix_strategy' must be one of: {supported}.")
    return strategy


def _setting_value(settings: tuple[dict[str, Any] | None, ...], *keys: str) -> Any:
    for section in settings:
        if not isinstance(section, dict):
            continue
        for key in keys:
            if key in section:
                return section[key]
    return None


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise MultiDatasetConfigError(f"'{field_name}' must be a positive integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise MultiDatasetConfigError(f"'{field_name}' must be a positive integer.") from exc
    if number <= 0:
        raise MultiDatasetConfigError(f"'{field_name}' must be a positive integer.")
    return number


def _sampling_from_mapping(item: dict[str, Any], *, field_name: str, default_seed: int) -> DatasetSampling | None:
    max_samples = item.get("max_samples")
    if max_samples is None:
        for key in ("seed", "sample_stage"):
            if key in item:
                raise MultiDatasetConfigError(f"'{field_name}.{key}' requires '{field_name}.max_samples'.")
        return None

    stage = str(item.get("sample_stage", AFTER_FILTER)).strip().lower().replace("-", "_")
    if stage not in SAMPLE_STAGES:
        supported = ", ".join(SAMPLE_STAGES)
        raise MultiDatasetConfigError(f"'{field_name}.sample_stage' must be one of: {supported}.")
    seed = item.get("seed", default_seed)
    if isinstance(seed, bool):
        raise MultiDatasetConfigError(f"'{field_name}.seed' must be an integer.")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise MultiDatasetConfigError(f"'{field_name}.seed' must be an integer.") from exc
    return DatasetSampling(
        max_samples=_positive_int(max_samples, field_name=f"{field_name}.max_samples"),
        seed=seed,
        stage=stage,
    )


def _train_data_entry(item: Any, *, field_name: str, default_seed: int) -> TrainDataEntry:
    if isinstance(item, (str, Path)):
        value = str(item)
        if not value:
            raise MultiDatasetConfigError(f"'{field_name}' must not be empty.")
        return TrainDataEntry(path=value)
    if isinstance(item, dict):
        path = next((item[key] for key in TRAIN_DATA_PATH_KEYS if item.get(key)), None)
        if path is None:
            supported = " or ".join(f"'{key}'" for key in TRAIN_DATA_PATH_KEYS)
            raise MultiDatasetConfigError(f"'{field_name}' must set {supported}.")
        if not isinstance(path, (str, Path)):
            raise MultiDatasetConfigError(f"'{field_name}.path' must be a string.")
        return TrainDataEntry(
            path=str(path),
            sampling=_sampling_from_mapping(item, field_name=field_name, default_seed=default_seed),
        )
    raise MultiDatasetConfigError(f"'{field_name}' must be a string or a mapping with a 'path'.")


def normalize_train_data_specs(train_data: Any, *settings: dict[str, Any] | None) -> list[TrainDataEntry]:
    """Normalize `train_data` into one entry per source, with optional sampling.

    A source is either a plain path/URL or a mapping that additionally sets
    `max_samples`, `seed`, and `sample_stage`.
    """
    seed_value = _setting_value(settings, "seed")
    default_seed = DEFAULT_SAMPLING_SEED if seed_value is None else int(seed_value)

    if isinstance(train_data, (str, Path, dict)):
        return [_train_data_entry(train_data, field_name="train_data", default_seed=default_seed)]
    if isinstance(train_data, list) and train_data:
        return [
            _train_data_entry(item, field_name=f"train_data[{index}]", default_seed=default_seed)
            for index, item in enumerate(train_data)
        ]
    raise MultiDatasetConfigError("'train_data' must be a non-empty string, mapping, or list of sources.")


def normalize_train_data_entries(train_data: Any) -> list[str]:
    return [entry.path for entry in normalize_train_data_specs(train_data)]


def is_multi_train_data(train_data: Any) -> bool:
    return len(normalize_train_data_specs(train_data)) > 1


def resolve_jsonl_train_data_path(train_data: str | Path, *settings: dict[str, Any] | None) -> Path:
    path = Path(resolve_train_data_entry(train_data, *settings))
    if path.is_dir():
        for filename in ("dataset.jsonl", "mixed_dataset.jsonl"):
            candidate = path / filename
            if candidate.exists():
                return candidate
        jsonl_files = sorted(path.glob("*.jsonl"))
        if jsonl_files:
            return jsonl_files[0]
        raise MultiDatasetConfigError(f"No JSONL training file found under '{path}'.")
    if not path.exists():
        raise MultiDatasetConfigError(f"Training data path '{path}' does not exist.")
    return path


def _safe_slug(value: Any, *, max_length: int = 80) -> str:
    raw_value = str(value)
    allowed = []
    for char in raw_value.strip().replace("\\", "/"):
        if char.isalnum() or char in {"_", "-"}:
            allowed.append(char)
        else:
            allowed.append("-")
    slug = "".join(allowed).strip("-_") or "dataset"
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(raw_value.encode("utf-8")).hexdigest()[:10]
    return f"{slug[: max_length - 11].rstrip('-_')}-{digest}"


def _source_name(train_data: str, index: int) -> str:
    path = Path(train_data)
    candidate = path.name or str(path)
    return f"{index}-{_safe_slug(candidate, max_length=60)}"


def _sampled_cache_key(input_path: Path, sampling: DatasetSampling) -> str:
    payload = {
        "input_path": str(input_path.resolve()),
        "input_hash": _file_sha256(input_path),
        **sampling.as_cache_payload(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _count_jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for raw_line in fh if raw_line.strip())


def _print_sample_report(report: dict[str, Any], *, cache_hit: bool) -> None:
    print(
        "[INFO] Dataset subsample "
        f"'{report['label']}' ({'cache hit' if cache_hit else 'materialized'}): "
        f"input={report['input_path']} output={report['output_path']} "
        f"rows_before={report['total']} rows_after={report['kept']} "
        f"max_samples={report['max_samples']} seed={report['seed']} sample_stage={report['sample_stage']}",
        flush=True,
    )


def materialize_sampled_jsonl_dataset(
    input_path: Path,
    sampling: DatasetSampling,
    *,
    cache_dir: str | Path | None = None,
    label: str = "dataset",
) -> SampledDatasetResult:
    """Keep a deterministic subset of a JSONL file, cached by content and settings.

    The selected rows keep their original order, and the same seed always picks
    the same rows for the same input file.
    """
    key = _sampled_cache_key(input_path, sampling)
    output_root = Path(cache_dir) if cache_dir is not None else DEFAULT_SAMPLED_DATASET_CACHE_DIR
    output_dir = output_root / f"{_safe_slug(label, max_length=60)}-{key}"
    output_path = output_dir / "dataset.jsonl"
    report_path = output_dir / "sample_report.json"

    def cached_result() -> SampledDatasetResult | None:
        if not (output_path.exists() and report_path.exists()):
            return None
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        _print_sample_report(report, cache_hit=True)
        return SampledDatasetResult(
            input_path=input_path,
            output_path=output_path,
            total=int(report["total"]),
            kept=int(report["kept"]),
            cache_hit=True,
        )

    result = cached_result()
    if result is not None:
        return result

    if is_torchrun_child() and not is_main_process():
        wait_for_files([output_path, report_path])
        result = cached_result()
        if result is None:
            raise MultiDatasetConfigError(f"Subsample cache under '{output_dir}' is incomplete after rank 0 finished.")
        return result

    total = _count_jsonl_rows(input_path)
    if total <= sampling.max_samples:
        print(
            f"[INFO] Dataset subsample '{label}' skipped: {input_path} has {total} rows, "
            f"which is not more than max_samples={sampling.max_samples}.",
            flush=True,
        )
        return SampledDatasetResult(
            input_path=input_path,
            output_path=input_path,
            total=total,
            kept=total,
            cache_hit=False,
        )

    selected = set(random.Random(sampling.seed).sample(range(total), sampling.max_samples))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with input_path.open(encoding="utf-8") as in_fh, output_path.open("w", encoding="utf-8", newline="\n") as out_fh:
            row_index = 0
            for raw_line in in_fh:
                if not raw_line.strip():
                    continue
                if row_index in selected:
                    out_fh.write(raw_line if raw_line.endswith("\n") else f"{raw_line}\n")
                row_index += 1
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise

    report = {
        "label": label,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total": total,
        "kept": len(selected),
        **sampling.as_cache_payload(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_sample_report(report, cache_hit=False)
    return SampledDatasetResult(
        input_path=input_path,
        output_path=output_path,
        total=total,
        kept=len(selected),
        cache_hit=False,
    )


def _sampled_dataset_cache_dir(*settings: dict[str, Any] | None) -> Any:
    return _setting_value(settings, "sampled_dataset_cache_dir")


def resolve_dataset_sources(
    train_data: Any,
    *settings: dict[str, Any] | None,
    config_path: str | None = None,
    apply_filters: bool = True,
) -> list[DatasetSource]:
    entries = normalize_train_data_specs(train_data, *settings)
    sample_cache_dir = _sampled_dataset_cache_dir(*settings)
    sources: list[DatasetSource] = []
    for index, entry in enumerate(entries):
        name = _source_name(entry.path, index)
        sampling = entry.sampling
        selected: str | Path = entry.path

        if sampling is not None and sampling.stage == BEFORE_FILTER:
            selected = materialize_sampled_jsonl_dataset(
                resolve_jsonl_train_data_path(entry.path, *settings),
                sampling,
                cache_dir=sample_cache_dir,
                label=name,
            ).output_path

        filtered = (
            apply_dataset_filter_if_configured(selected, *settings, config_path=config_path)
            if apply_filters
            else None
        )
        path = filtered.output_path if filtered is not None else resolve_jsonl_train_data_path(selected, *settings)

        if sampling is not None and sampling.stage == AFTER_FILTER:
            path = materialize_sampled_jsonl_dataset(
                path,
                sampling,
                cache_dir=sample_cache_dir,
                label=name,
            ).output_path

        sources.append(DatasetSource(name=name, original=entry.path, path=path, sampling=sampling))
    return sources


def describe_sources(sources: Iterable[DatasetSource]) -> str:
    values = [str(source.path) for source in sources]
    if not values:
        return "<no datasets>"
    if len(values) == 1:
        return values[0]
    return "[" + ", ".join(values) + "]"


def dedupe_values_from_training_row(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key, value in row.items():
        if key in {"label", "dataset_name"} or key.startswith("_"):
            continue
        if value is None:
            continue
        if isinstance(value, list):
            values.update(str(item) for item in value)
        else:
            values.add(str(value))
    return values


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mixed_cache_key(sources: list[DatasetSource]) -> str:
    payload = [
        {
            "name": source.name,
            "path": str(source.path.resolve()),
            "sha256": _file_sha256(source.path),
            "sampling": source.sampling.as_cache_payload() if source.sampling is not None else None,
        }
        for source in sources
    ]
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _print_mix_report(report: dict[str, Any], *, cache_hit: bool) -> None:
    print(
        f"[INFO] Mixed dataset ({'cache hit' if cache_hit else 'materialized'}): "
        f"total_rows={report['total_lines']}",
        flush=True,
    )
    for source in report.get("sources", []):
        print(f"[INFO]   {source['name']}: rows={source['lines']} path={source['path']}", flush=True)


def materialize_mixed_jsonl_dataset(
    train_data: Any,
    *settings: dict[str, Any] | None,
    cache_dir: str | Path | None = None,
    config_path: str | None = None,
    group_name: str | None = None,
) -> MixedDatasetResult:
    sources = resolve_dataset_sources(train_data, *settings, config_path=config_path)
    if len(sources) < 2:
        raise MultiDatasetConfigError("Mixed dataset materialization requires at least two training data sources.")

    key = _mixed_cache_key(sources)
    label = _safe_slug(group_name or "-".join(source.name for source in sources), max_length=80)
    output_root = Path(cache_dir) if cache_dir is not None else DEFAULT_MIXED_DATASET_CACHE_DIR
    output_dir = output_root / f"{label}-{key}"
    output_path = output_dir / "dataset.jsonl"
    report_path = output_dir / "mix_report.json"

    if output_path.exists() and report_path.exists():
        try:
            _print_mix_report(json.loads(report_path.read_text(encoding="utf-8")), cache_hit=True)
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        return MixedDatasetResult(sources=sources, output_dir=output_dir, output_path=output_path, cache_hit=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    line_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8", newline="\n") as out_fh:
        for source in sources:
            count = 0
            with source.path.open(encoding="utf-8") as in_fh:
                for raw_line in in_fh:
                    if not raw_line.strip():
                        continue
                    out_fh.write(raw_line)
                    if not raw_line.endswith("\n"):
                        out_fh.write("\n")
                    count += 1
            line_counts[source.name] = count

    report = {
        "sources": [
            {
                "name": source.name,
                "original": source.original,
                "path": str(source.path),
                "lines": line_counts[source.name],
            }
            for source in sources
        ],
        "total_lines": sum(line_counts.values()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _print_mix_report(report, cache_hit=False)
    return MixedDatasetResult(sources=sources, output_dir=output_dir, output_path=output_path, cache_hit=False)


def _shuffle_values(values: list[int], *, generator: Any = None, seed: int = 0) -> list[int]:
    if len(values) <= 1:
        return list(values)
    if generator is not None:
        try:
            import torch

            order = torch.randperm(len(values), generator=generator).tolist()
            return [values[index] for index in order]
        except Exception:
            pass
    rng = random.Random(seed)
    shuffled = list(values)
    rng.shuffle(shuffled)
    return shuffled


def _quota_for_batch(batch_size: int, remaining_counts: dict[str, int]) -> dict[str, int]:
    total_remaining = sum(remaining_counts.values())
    if total_remaining <= 0:
        return {}

    desired = min(batch_size, total_remaining)
    quotas: dict[str, int] = {}
    fractions: list[tuple[float, int, str]] = []
    for dataset_id, remaining in remaining_counts.items():
        if remaining <= 0:
            quotas[dataset_id] = 0
            continue
        raw = desired * remaining / total_remaining
        base = min(remaining, int(math.floor(raw)))
        quotas[dataset_id] = base
        fractions.append((raw - base, remaining, dataset_id))

    missing = desired - sum(quotas.values())
    for _, _, dataset_id in sorted(fractions, reverse=True):
        if missing <= 0:
            break
        if quotas[dataset_id] < remaining_counts[dataset_id]:
            quotas[dataset_id] += 1
            missing -= 1
    return quotas


class ProportionalNoDuplicatesBatchSampler:
    """Best-effort proportional sampler with hard no-duplicate batch checks."""

    def __init__(
        self,
        dataset_ids: list[str],
        dedupe_values: list[set[str]],
        *,
        batch_size: int,
        drop_last: bool,
        generator: Any = None,
        seed: int = 0,
    ) -> None:
        if len(dataset_ids) != len(dedupe_values):
            raise MultiDatasetConfigError("'dataset_ids' and 'dedupe_values' must have the same length.")
        if batch_size <= 0:
            raise MultiDatasetConfigError("'batch_size' must be greater than zero.")
        self.dataset_ids = list(dataset_ids)
        self.dedupe_values = [set(values) for values in dedupe_values]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.generator = generator
        self.seed = int(seed or 0)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_queues(self) -> dict[str, deque[int]]:
        grouped: dict[str, list[int]] = {}
        for index, dataset_id in enumerate(self.dataset_ids):
            grouped.setdefault(dataset_id, []).append(index)
        return {
            dataset_id: deque(_shuffle_values(indices, generator=self.generator, seed=self.seed + self.epoch + offset))
            for offset, (dataset_id, indices) in enumerate(grouped.items())
        }

    def _take_one(self, queue: deque[int], used_values: set[str]) -> int | None:
        for _ in range(len(queue)):
            index = queue.popleft()
            if self.dedupe_values[index].isdisjoint(used_values):
                return index
            queue.append(index)
        return None

    def __iter__(self) -> Iterator[list[int]]:
        queues = self._build_queues()
        dataset_order = list(queues)
        while any(queues.values()):
            remaining_counts = {dataset_id: len(queue) for dataset_id, queue in queues.items()}
            quotas = _quota_for_batch(self.batch_size, remaining_counts)
            used_values: set[str] = set()
            batch: list[int] = []

            quota_order = sorted(dataset_order, key=lambda dataset_id: (quotas.get(dataset_id, 0), len(queues[dataset_id])), reverse=True)
            for dataset_id in quota_order:
                target = quotas.get(dataset_id, 0)
                while target > 0 and len(batch) < self.batch_size:
                    index = self._take_one(queues[dataset_id], used_values)
                    if index is None:
                        break
                    batch.append(index)
                    used_values.update(self.dedupe_values[index])
                    target -= 1

            while len(batch) < self.batch_size:
                progress = False
                fill_order = sorted(dataset_order, key=lambda dataset_id: len(queues[dataset_id]), reverse=True)
                for dataset_id in fill_order:
                    if len(batch) >= self.batch_size:
                        break
                    index = self._take_one(queues[dataset_id], used_values)
                    if index is None:
                        continue
                    batch.append(index)
                    used_values.update(self.dedupe_values[index])
                    progress = True
                if not progress:
                    break

            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset_ids) // self.batch_size
        return (len(self.dataset_ids) + self.batch_size - 1) // self.batch_size


class ProportionalNoDuplicatesBatchSamplerFactory:
    def __init__(self, dataset_ids: list[str], dedupe_values: list[set[str]]) -> None:
        self.dataset_ids = list(dataset_ids)
        self.dedupe_values = [set(values) for values in dedupe_values]

    def __call__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        drop_last: bool,
        generator: Any = None,
        seed: int = 0,
        **_: Any,
    ) -> ProportionalNoDuplicatesBatchSampler:
        if len(dataset) != len(self.dataset_ids):
            raise MultiDatasetConfigError(
                "Proportional batch sampler metadata does not match the training dataset length."
            )
        return ProportionalNoDuplicatesBatchSampler(
            self.dataset_ids,
            self.dedupe_values,
            batch_size=batch_size,
            drop_last=drop_last,
            generator=generator,
            seed=seed,
        )


def dataset_counts(dataset_ids: Iterable[str]) -> dict[str, int]:
    return dict(Counter(dataset_ids))
