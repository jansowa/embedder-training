"""Declarative JSONL dataset filtering shared by training backends."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from training.backends.registry import TrainingCliError
from training.dataset_sources import HF_DATASET_CACHE_CONFIG_KEYS, resolve_train_data_entry
from training.distributed import is_main_process, is_torchrun_child, wait_for_files


DEFAULT_CACHE_DIR = Path("cache/filtered_datasets")
FILTER_CONFIG_KEYS = {"dataset_filter", "dataset_filter_cache_dir"}
POLICIES = {"fail", "include", "exclude"}
NULL_POSITIVE_SCORE_STRATEGIES = {"fail", "include", "include_if_any_scored_positive_kept"}
AGGREGATES = {"min", "max", "mean", "sum", "count"}
EMPTY_PASSAGE_WARNING_ID_LIMIT = 20
MISSING = object()


class DatasetFilterError(TrainingCliError):
    """Raised when a declarative dataset filter cannot be applied."""


@dataclass(frozen=True)
class FilteredDatasetResult:
    input_path: Path
    output_dir: Path
    output_path: Path
    report_path: Path
    report: dict[str, Any]
    cache_hit: bool


@dataclass
class _EvalContext:
    profile_name: str
    line_no: int
    scope: str
    missing_counts: Counter[str] = field(default_factory=Counter)
    type_mismatch_counts: Counter[str] = field(default_factory=Counter)


def _load_yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise DatasetFilterError("Dataset filters require PyYAML. Install it before using 'dataset_filter'.") from exc
    return yaml


def _validate_policy(policy: Any, *, key: str, profile_path: Path) -> str:
    if policy is None:
        return "fail"
    policy = str(policy)
    if policy not in POLICIES:
        supported = ", ".join(sorted(POLICIES))
        raise DatasetFilterError(f"Invalid {key} '{policy}' in {profile_path}. Supported values: {supported}.")
    return policy


def _validate_null_positive_score_strategy(value: Any, *, profile_path: Path) -> str:
    if value is None:
        return "fail"
    strategy = str(value)
    if strategy not in NULL_POSITIVE_SCORE_STRATEGIES:
        supported = ", ".join(sorted(NULL_POSITIVE_SCORE_STRATEGIES))
        raise DatasetFilterError(
            f"Invalid null_positive_score_strategy '{strategy}' in {profile_path}. Supported values: {supported}."
        )
    return strategy


def _null_positive_score_fields(profile: dict[str, Any], *, profile_path: Path) -> tuple[str, ...]:
    fields = profile.get("null_positive_score_fields", ["pos_scores"])
    if not isinstance(fields, list) or not fields or any(not isinstance(field, str) or not field for field in fields):
        raise DatasetFilterError(
            f"Dataset filter profile '{profile_path}' must set null_positive_score_fields as a non-empty list of field names."
        )
    if len(set(fields)) != len(fields):
        raise DatasetFilterError(f"Dataset filter profile '{profile_path}' must not repeat null_positive_score_fields.")
    return tuple(fields)


def _resolve_relative_path(path_value: str | Path, *, config_path: str | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    if config_path:
        candidate = Path(config_path).parent / path
        if candidate.exists():
            return candidate
    return path


def resolve_jsonl_input_path(train_data: str | Path, *settings: dict[str, Any] | None) -> Path:
    path = Path(resolve_train_data_entry(train_data, *settings))
    if path.is_dir():
        for filename in ("dataset.jsonl", "mixed_dataset.jsonl"):
            candidate = path / filename
            if candidate.exists():
                return candidate
        jsonl_files = sorted(path.glob("*.jsonl"))
        if jsonl_files:
            return jsonl_files[0]
        raise DatasetFilterError(f"No JSONL training file found under '{path}'.")
    if not path.exists():
        raise DatasetFilterError(f"Training data path '{path}' does not exist.")
    if path.suffix != ".jsonl":
        raise DatasetFilterError(f"Training data path '{path}' must be a JSONL file or a directory containing one.")
    return path


def _load_profile(profile_path: str | Path, *, config_path: str | None = None) -> tuple[Path, dict[str, Any]]:
    resolved_path = _resolve_relative_path(profile_path, config_path=config_path)
    if not resolved_path.exists():
        raise DatasetFilterError(f"Dataset filter profile '{profile_path}' does not exist.")

    yaml = _load_yaml_module()
    try:
        with resolved_path.open(encoding="utf-8") as fh:
            profile = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise DatasetFilterError(f"Could not read dataset filter profile '{resolved_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise DatasetFilterError(f"Could not parse dataset filter profile '{resolved_path}': {exc}") from exc

    if not isinstance(profile, dict):
        raise DatasetFilterError(f"Dataset filter profile '{resolved_path}' must contain a YAML mapping.")
    try:
        version = int(profile.get("version", 0))
    except (TypeError, ValueError) as exc:
        raise DatasetFilterError(f"Dataset filter profile '{resolved_path}' must set version: 1.") from exc
    if version != 1:
        raise DatasetFilterError(f"Dataset filter profile '{resolved_path}' must set version: 1.")
    if not profile.get("name"):
        raise DatasetFilterError(f"Dataset filter profile '{resolved_path}' must set a non-empty name.")
    for key in ("rules", "sample_rules", "positive_rules", "negative_rules"):
        if key in profile and not isinstance(profile[key], list):
            raise DatasetFilterError(f"Dataset filter profile '{resolved_path}' must set {key} as a list when provided.")
    if not any(key in profile for key in ("rules", "sample_rules", "positive_rules", "negative_rules")):
        raise DatasetFilterError(
            f"Dataset filter profile '{resolved_path}' must set at least one of "
            "rules, sample_rules, positive_rules, or negative_rules."
        )
    for key in ("min_positives", "min_negatives"):
        if key in profile:
            try:
                minimum = int(profile[key])
            except (TypeError, ValueError) as exc:
                raise DatasetFilterError(f"Dataset filter profile '{resolved_path}' must set {key} as an integer.") from exc
            if minimum < 0:
                raise DatasetFilterError(f"Dataset filter profile '{resolved_path}' must set {key} to zero or greater.")
    if "drop_samples_with_empty_passages" in profile and not isinstance(
        profile["drop_samples_with_empty_passages"], bool
    ):
        raise DatasetFilterError(
            f"Dataset filter profile '{resolved_path}' must set drop_samples_with_empty_passages to true or false."
        )

    _validate_policy(profile.get("missing_policy", "fail"), key="missing_policy", profile_path=resolved_path)
    _validate_policy(profile.get("type_mismatch_policy", "fail"), key="type_mismatch_policy", profile_path=resolved_path)
    _validate_null_positive_score_strategy(profile.get("null_positive_score_strategy"), profile_path=resolved_path)
    if "null_positive_score_fields" in profile:
        _null_positive_score_fields(profile, profile_path=resolved_path)
    return resolved_path, profile


def _canonical_profile(profile: dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(input_path: Path, input_hash: str, profile: dict[str, Any]) -> str:
    payload = {
        "input_path": str(input_path.resolve()),
        "input_hash": input_hash,
        "profile": _canonical_profile(profile),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "dataset-filter"


def _policy_from_node(node: dict[str, Any], key: str, inherited: str, profile_path: Path) -> str:
    if key not in node:
        return inherited
    return _validate_policy(node.get(key), key=key, profile_path=profile_path)


def _field_name(node: dict[str, Any]) -> str:
    field_name = node.get("field")
    if not isinstance(field_name, str) or not field_name:
        raise DatasetFilterError("Dataset filter rule must set a non-empty 'field'.")
    if any(part == "" for part in field_name.split(".")):
        raise DatasetFilterError(f"Dataset filter rule field '{field_name}' contains an empty path segment.")
    return field_name


def _resolve_field(sample: dict[str, Any], field_name: str) -> Any:
    current: Any = sample
    for part in field_name.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return MISSING
    return current


def _scoped_field_name(ctx: _EvalContext, field_name: str) -> str:
    return f"{ctx.scope}:{field_name}"


def _handle_missing(field_name: str, policy: str, ctx: _EvalContext) -> bool:
    scoped_field = _scoped_field_name(ctx, field_name)
    ctx.missing_counts[scoped_field] += 1
    if policy == "fail":
        raise DatasetFilterError(
            f"Dataset filter '{ctx.profile_name}' failed on line {ctx.line_no}: "
            f"missing field '{scoped_field}'. Set missing_policy to include or exclude if this is expected."
        )
    return policy == "include"


def _handle_type_mismatch(field_name: str, policy: str, ctx: _EvalContext, detail: str) -> bool:
    scoped_field = _scoped_field_name(ctx, field_name)
    ctx.type_mismatch_counts[scoped_field] += 1
    if policy == "fail":
        raise DatasetFilterError(
            f"Dataset filter '{ctx.profile_name}' failed on line {ctx.line_no}: "
            f"type mismatch for field '{scoped_field}' ({detail}). "
            "Set type_mismatch_policy to include or exclude if this is expected."
        )
    return policy == "include"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compatible_types(left: Any, right: Any) -> bool:
    if _is_number(left) and _is_number(right):
        return True
    return type(left) is type(right)


def _ensure_values(node: dict[str, Any], *, op: str) -> list[Any]:
    if "values" in node:
        values = node["values"]
    elif op == "between" and "value" in node:
        values = node["value"]
    else:
        raise DatasetFilterError(f"Dataset filter operator '{op}' requires 'values'.")
    if not isinstance(values, list):
        raise DatasetFilterError(f"Dataset filter operator '{op}' requires 'values' to be a list.")
    return values


def _ensure_value(node: dict[str, Any], *, op: str) -> Any:
    if "value" not in node:
        raise DatasetFilterError(f"Dataset filter operator '{op}' requires 'value'.")
    return node["value"]


def _aggregate_value(value: Any, aggregate: str, field_name: str, policy: str, ctx: _EvalContext) -> Any:
    if aggregate not in AGGREGATES:
        supported = ", ".join(sorted(AGGREGATES))
        raise DatasetFilterError(f"Unsupported aggregate '{aggregate}' for field '{field_name}'. Supported: {supported}.")
    if not isinstance(value, (list, dict)):
        return _handle_type_mismatch(field_name, policy, ctx, f"aggregate '{aggregate}' requires a list or mapping")

    values = list(value.values()) if isinstance(value, dict) else list(value)
    if aggregate == "count":
        return len(values)

    if any(not _is_number(item) for item in values):
        return _handle_type_mismatch(field_name, policy, ctx, f"aggregate '{aggregate}' requires numeric values")
    if not values and aggregate in {"min", "max", "mean"}:
        return _handle_type_mismatch(field_name, policy, ctx, f"aggregate '{aggregate}' requires at least one value")

    if aggregate == "min":
        return min(values)
    if aggregate == "max":
        return max(values)
    if aggregate == "mean":
        return sum(values) / len(values)
    if aggregate == "sum":
        return sum(values)
    raise AssertionError(f"unreachable aggregate: {aggregate}")


def _evaluate_leaf(
    node: dict[str, Any],
    sample: dict[str, Any],
    *,
    missing_policy: str,
    type_mismatch_policy: str,
    ctx: _EvalContext,
) -> bool:
    field_name = _field_name(node)
    op = str(node.get("op", ""))
    if not op:
        raise DatasetFilterError(f"Dataset filter rule for field '{field_name}' must set 'op'.")

    value = _resolve_field(sample, field_name)
    if op == "exists":
        return value is not MISSING
    if op == "missing":
        return value is MISSING
    if value is MISSING:
        return _handle_missing(field_name, missing_policy, ctx)

    aggregate = node.get("aggregate")
    if aggregate is not None:
        value = _aggregate_value(value, str(aggregate), field_name, type_mismatch_policy, ctx)
        if isinstance(value, bool):
            return value

    if op in {"gt", "gte", "lt", "lte"}:
        expected = _ensure_value(node, op=op)
        if not _is_number(value) or not _is_number(expected):
            return _handle_type_mismatch(field_name, type_mismatch_policy, ctx, f"operator '{op}' requires numbers")
        if op == "gt":
            return value > expected
        if op == "gte":
            return value >= expected
        if op == "lt":
            return value < expected
        return value <= expected

    if op == "between":
        values = _ensure_values(node, op=op)
        if len(values) != 2 or not _is_number(values[0]) or not _is_number(values[1]) or not _is_number(value):
            return _handle_type_mismatch(field_name, type_mismatch_policy, ctx, "operator 'between' requires two numeric bounds")
        low, high = values
        return low <= value <= high

    if op in {"eq", "neq"}:
        expected = _ensure_value(node, op=op)
        if not _compatible_types(value, expected):
            return _handle_type_mismatch(field_name, type_mismatch_policy, ctx, f"operator '{op}' compares incompatible types")
        result = value == expected
        return result if op == "eq" else not result

    if op in {"in", "not_in"}:
        values = _ensure_values(node, op=op)
        if isinstance(value, (list, dict)):
            return _handle_type_mismatch(field_name, type_mismatch_policy, ctx, f"operator '{op}' requires a scalar field value")
        result = value in values
        return result if op == "in" else not result

    if op in {"intersects", "contains_any", "contains_all", "contains_none"}:
        values = _ensure_values(node, op=op)
        if not isinstance(value, list):
            return _handle_type_mismatch(field_name, type_mismatch_policy, ctx, f"operator '{op}' requires a list field value")
        try:
            actual = set(value)
            expected = set(values)
        except TypeError:
            return _handle_type_mismatch(field_name, type_mismatch_policy, ctx, f"operator '{op}' requires hashable list values")
        if op in {"intersects", "contains_any"}:
            return bool(actual & expected)
        if op == "contains_all":
            return expected.issubset(actual)
        return not bool(actual & expected)

    supported_ops = (
        "gt, gte, lt, lte, between, eq, neq, in, not_in, intersects, contains_any, "
        "contains_all, contains_none, exists, missing"
    )
    raise DatasetFilterError(f"Unsupported dataset filter operator '{op}' for field '{field_name}'. Supported: {supported_ops}.")


def _evaluate_node(
    node: Any,
    sample: dict[str, Any],
    *,
    missing_policy: str,
    type_mismatch_policy: str,
    ctx: _EvalContext,
    profile_path: Path,
) -> bool:
    if not isinstance(node, dict):
        raise DatasetFilterError(f"Dataset filter profile '{profile_path}' contains a rule that is not a mapping.")

    missing_policy = _policy_from_node(node, "missing_policy", missing_policy, profile_path)
    type_mismatch_policy = _policy_from_node(node, "type_mismatch_policy", type_mismatch_policy, profile_path)

    logical_keys = [key for key in ("all", "any", "not") if key in node]
    has_leaf = "field" in node
    if len(logical_keys) + int(has_leaf) != 1:
        raise DatasetFilterError("Dataset filter rule must contain exactly one of 'field', 'all', 'any', or 'not'.")

    if has_leaf:
        return _evaluate_leaf(
            node,
            sample,
            missing_policy=missing_policy,
            type_mismatch_policy=type_mismatch_policy,
            ctx=ctx,
        )

    if "not" in node:
        return not _evaluate_node(
            node["not"],
            sample,
            missing_policy=missing_policy,
            type_mismatch_policy=type_mismatch_policy,
            ctx=ctx,
            profile_path=profile_path,
        )

    key = logical_keys[0]
    children = node[key]
    if not isinstance(children, list):
        raise DatasetFilterError(f"Dataset filter logical rule '{key}' must contain a list.")
    results = [
        _evaluate_node(
            child,
            sample,
            missing_policy=missing_policy,
            type_mismatch_policy=type_mismatch_policy,
            ctx=ctx,
            profile_path=profile_path,
        )
        for child in children
    ]
    return all(results) if key == "all" else any(results)


def _evaluate_rules(
    rules: list[Any],
    profile: dict[str, Any],
    profile_path: Path,
    sample: Any,
    ctx: _EvalContext,
) -> bool:
    missing_policy = _validate_policy(profile.get("missing_policy", "fail"), key="missing_policy", profile_path=profile_path)
    type_mismatch_policy = _validate_policy(
        profile.get("type_mismatch_policy", "fail"),
        key="type_mismatch_policy",
        profile_path=profile_path,
    )
    results = [
        _evaluate_node(
            rule,
            sample,
            missing_policy=missing_policy,
            type_mismatch_policy=type_mismatch_policy,
            ctx=ctx,
            profile_path=profile_path,
        )
        for rule in rules
    ]
    return all(results)


def _sample_rules(profile: dict[str, Any]) -> list[Any]:
    if "sample_rules" in profile:
        return profile["sample_rules"]
    return profile.get("rules", [])


def _min_count(profile: dict[str, Any], key: str, default: int) -> int:
    return int(profile.get(key, default))


def _empty_passage_positions(sample: dict[str, Any]) -> dict[str, list[int]]:
    """Return indexes of empty strings in the training passage lists.

    Type validation remains the responsibility of the normal filter pipeline. This
    helper deliberately identifies only literal empty strings, so whitespace-only
    passages retain the existing behavior.
    """
    empty_positions: dict[str, list[int]] = {}
    for field_name in ("pos", "neg"):
        value = sample.get(field_name)
        if isinstance(value, list):
            positions = [index for index, passage in enumerate(value) if passage == ""]
            if positions:
                empty_positions[field_name] = positions
    return empty_positions


def _empty_passage_report_entry(
    sample: dict[str, Any], line_no: int, empty_positions: dict[str, list[int]]
) -> dict[str, Any]:
    return {
        "line_no": line_no,
        "query_id": sample.get("query_id"),
        "empty_passage_indexes": empty_positions,
    }


def _ensure_text_items(sample: dict[str, Any], key: str, line_no: int) -> list[Any]:
    value = sample.get(key)
    if isinstance(value, list):
        return value
    raise DatasetFilterError(f"Line {line_no}: field '{key}' must be a list before passage-level filtering.")


def _ensure_parallel_list(sample: dict[str, Any], field_name: str, expected_len: int, line_no: int) -> list[Any] | None:
    if field_name not in sample:
        return None
    value = sample[field_name]
    if not isinstance(value, list):
        raise DatasetFilterError(f"Line {line_no}: field '{field_name}' must be a list when provided.")
    if len(value) != expected_len:
        raise DatasetFilterError(
            f"Line {line_no}: field '{field_name}' has {len(value)} items but expected {expected_len} "
            "to match its parallel passage list."
        )
    return value


def _feature_list(sample: dict[str, Any], key: str, expected_len: int, line_no: int) -> list[Any] | None:
    features = sample.get("features")
    if not isinstance(features, dict) or key not in features:
        return None
    value = features[key]
    if not isinstance(value, list):
        raise DatasetFilterError(f"Line {line_no}: field 'features.{key}' must be a list when provided.")
    if len(value) != expected_len:
        raise DatasetFilterError(
            f"Line {line_no}: field 'features.{key}' has {len(value)} items but expected {expected_len} "
            f"to match '{key}'."
        )
    return value


def _has_only_null_score_values(
    parallel_values: dict[str, list[Any] | None],
    score_fields: tuple[str, ...],
    index: int,
) -> bool:
    values = [parallel_values[field_name][index] for field_name in score_fields if parallel_values.get(field_name) is not None]
    return bool(values) and all(value is None for value in values)


def _filter_passages(
    sample: dict[str, Any],
    *,
    passage_key: str,
    feature_key: str,
    parallel_fields: tuple[str, ...],
    rules: list[Any],
    profile: dict[str, Any],
    profile_path: Path,
    line_no: int,
    profile_name: str,
    missing_counts: Counter[str],
    type_mismatch_counts: Counter[str],
    null_positive_score_strategy: str = "fail",
    null_positive_score_fields: tuple[str, ...] = (),
) -> tuple[int, int]:
    passages = _ensure_text_items(sample, passage_key, line_no)
    total = len(passages)
    feature_items = _feature_list(sample, feature_key, total, line_no)
    parallel_values = {
        field_name: _ensure_parallel_list(sample, field_name, total, line_no)
        for field_name in parallel_fields
    }

    if not rules:
        return total, total

    null_score_indices = {
        idx
        for idx in range(total)
        if null_positive_score_strategy != "fail"
        and _has_only_null_score_values(parallel_values, null_positive_score_fields, idx)
    }
    scored_kept_indices: set[int] = set()
    for idx in range(total):
        if idx in null_score_indices:
            continue
        item_features = feature_items[idx] if feature_items is not None else MISSING
        if isinstance(item_features, dict):
            rule_item = dict(item_features)
        else:
            rule_item = {}
        rule_item["parallel"] = {
            field_name: values[idx]
            for field_name, values in parallel_values.items()
            if values is not None
        }
        ctx = _EvalContext(profile_name=profile_name, line_no=line_no, scope=feature_key)
        keep = _evaluate_rules(rules, profile, profile_path, rule_item, ctx)
        missing_counts.update(ctx.missing_counts)
        type_mismatch_counts.update(ctx.type_mismatch_counts)
        if keep:
            scored_kept_indices.add(idx)

    include_null_scores = null_positive_score_strategy == "include" or (
        null_positive_score_strategy == "include_if_any_scored_positive_kept" and bool(scored_kept_indices)
    )
    kept_indices = [
        idx
        for idx in range(total)
        if idx in scored_kept_indices or (include_null_scores and idx in null_score_indices)
    ]

    sample[passage_key] = [passages[idx] for idx in kept_indices]
    if feature_items is not None:
        features = sample.get("features")
        if isinstance(features, dict):
            features[feature_key] = [feature_items[idx] for idx in kept_indices]
    for field_name, values in parallel_values.items():
        if values is not None:
            sample[field_name] = [values[idx] for idx in kept_indices]

    return total, len(kept_indices)


def _filter_sample(
    sample: dict[str, Any],
    *,
    profile: dict[str, Any],
    profile_path: Path,
    line_no: int,
    profile_name: str,
) -> tuple[bool, dict[str, int], Counter[str], Counter[str]]:
    missing_counts: Counter[str] = Counter()
    type_mismatch_counts: Counter[str] = Counter()
    stats = {
        "positives_total": 0,
        "positives_kept": 0,
        "negatives_total": 0,
        "negatives_kept": 0,
        "removed_by_min_positives": 0,
        "removed_by_min_negatives": 0,
    }

    sample_ctx = _EvalContext(profile_name=profile_name, line_no=line_no, scope="sample")
    sample_keep = _evaluate_rules(_sample_rules(profile), profile, profile_path, sample, sample_ctx)
    missing_counts.update(sample_ctx.missing_counts)
    type_mismatch_counts.update(sample_ctx.type_mismatch_counts)
    if not sample_keep:
        return False, stats, missing_counts, type_mismatch_counts

    positives_total, positives_kept = _filter_passages(
        sample,
        passage_key="pos",
        feature_key="pos",
        parallel_fields=("pos_scores", "pos_scores_stronger_reranker", "pos_id"),
        rules=profile.get("positive_rules", []),
        profile=profile,
        profile_path=profile_path,
        line_no=line_no,
        profile_name=profile_name,
        missing_counts=missing_counts,
        type_mismatch_counts=type_mismatch_counts,
        null_positive_score_strategy=_validate_null_positive_score_strategy(
            profile.get("null_positive_score_strategy"), profile_path=profile_path
        ),
        null_positive_score_fields=_null_positive_score_fields(profile, profile_path=profile_path),
    )
    negatives_total, negatives_kept = _filter_passages(
        sample,
        passage_key="neg",
        feature_key="neg",
        parallel_fields=("neg_scores", "neg_id"),
        rules=profile.get("negative_rules", []),
        profile=profile,
        profile_path=profile_path,
        line_no=line_no,
        profile_name=profile_name,
        missing_counts=missing_counts,
        type_mismatch_counts=type_mismatch_counts,
    )
    stats.update(
        {
            "positives_total": positives_total,
            "positives_kept": positives_kept,
            "negatives_total": negatives_total,
            "negatives_kept": negatives_kept,
        }
    )

    if positives_kept < _min_count(profile, "min_positives", 1):
        stats["removed_by_min_positives"] = 1
        stats["positives_kept"] = 0
        stats["negatives_kept"] = 0
        return False, stats, missing_counts, type_mismatch_counts
    if negatives_kept < _min_count(profile, "min_negatives", 1):
        stats["removed_by_min_negatives"] = 1
        stats["positives_kept"] = 0
        stats["negatives_kept"] = 0
        return False, stats, missing_counts, type_mismatch_counts
    return True, stats, missing_counts, type_mismatch_counts


def _print_report(report: dict[str, Any], *, cache_hit: bool) -> None:
    print(
        "[INFO] Dataset filter "
        f"'{report['profile_name']}' ({'cache hit' if cache_hit else 'materialized'}): "
        f"input={report['input_path']} output={report['output_path']} "
        f"total={report['total']} kept={report['kept']} removed={report['removed']} "
        f"positives={report.get('positives_kept', 0)}/{report.get('positives_total', 0)} "
        f"negatives={report.get('negatives_kept', 0)}/{report.get('negatives_total', 0)}",
        flush=True,
    )
    for field_name, count in sorted(report.get("missing_counts", {}).items()):
        if count:
            print(f"[WARN] Dataset filter missing field '{field_name}' affected {count} samples.", file=sys.stderr, flush=True)
    for field_name, count in sorted(report.get("type_mismatch_counts", {}).items()):
        if count:
            print(
                f"[WARN] Dataset filter type mismatch for field '{field_name}' affected {count} samples.",
                file=sys.stderr,
                flush=True,
            )
    empty_passages = report.get("empty_passages", {})
    removed_samples = int(empty_passages.get("removed_samples", 0))
    if removed_samples:
        print("=" * 78, file=sys.stderr, flush=True)
        print(
            "[WARNING] DATASET QUALITY: skipped "
            f"{removed_samples} samples containing an empty string in 'pos' or 'neg'.",
            file=sys.stderr,
            flush=True,
        )
        identifiers = empty_passages.get("identifiers", [])
        if removed_samples <= EMPTY_PASSAGE_WARNING_ID_LIMIT:
            formatted_identifiers = ", ".join(
                f"query_id={entry['query_id']!r} (line {entry['line_no']})" for entry in identifiers
            )
            print(f"[WARNING] Affected samples: {formatted_identifiers}", file=sys.stderr, flush=True)
        else:
            print(
                "[WARNING] Too many affected samples to list here; see the detailed report.",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"[WARNING] Detailed report: {empty_passages['detail_report_path']}",
            file=sys.stderr,
            flush=True,
        )
        print("=" * 78, file=sys.stderr, flush=True)


def materialize_filtered_dataset(
    train_data: str | Path,
    dataset_filter: str | Path,
    *,
    cache_dir: str | Path | None = None,
    config_path: str | None = None,
    source_settings: dict[str, Any] | None = None,
) -> FilteredDatasetResult:
    profile_path, profile = _load_profile(dataset_filter, config_path=config_path)
    input_path = resolve_jsonl_input_path(train_data, source_settings)
    input_hash = _file_sha256(input_path)
    key = _cache_key(input_path, input_hash, profile)

    output_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    output_dir = output_root / f"{_slug(str(profile['name']))}-{key}"
    output_path = output_dir / "dataset.jsonl"
    report_path = output_dir / "filter_report.json"
    empty_passage_report_path = output_dir / "empty_passage_report.jsonl"
    drop_empty_passages = bool(profile.get("drop_samples_with_empty_passages", False))

    def cached_result(*, cache_hit: bool) -> FilteredDatasetResult | None:
        if not (output_path.exists() and report_path.exists()):
            return None
        if drop_empty_passages and not empty_passage_report_path.exists():
            return None
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not report:
            return None
        _print_report(report, cache_hit=cache_hit)
        return FilteredDatasetResult(input_path, output_dir, output_path, report_path, report, cache_hit=cache_hit)

    result = cached_result(cache_hit=True)
    if result is not None:
        return result

    if is_torchrun_child() and not is_main_process():
        wait_for_files([output_path, report_path])
        result = cached_result(cache_hit=True)
        if result is None:
            raise DatasetFilterError(f"Filtered dataset cache under '{output_dir}' is incomplete after rank 0 finished.")
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_name = str(profile["name"])
    total = 0
    kept = 0
    positives_total = 0
    positives_kept = 0
    negatives_total = 0
    negatives_kept = 0
    removed_by_min_positives = 0
    removed_by_min_negatives = 0
    missing_counts: Counter[str] = Counter()
    type_mismatch_counts: Counter[str] = Counter()
    empty_passage_samples = 0
    empty_positive_samples = 0
    empty_negative_samples = 0
    empty_passage_identifiers: list[dict[str, Any]] = []

    try:
        with (
            input_path.open(encoding="utf-8") as in_fh,
            output_path.open("w", encoding="utf-8") as out_fh,
            empty_passage_report_path.open("w", encoding="utf-8") as empty_report_fh,
        ):
            for line_no, raw_line in enumerate(in_fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                total += 1
                try:
                    sample = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DatasetFilterError(f"Line {line_no} in '{input_path}' is invalid JSON: {exc}") from exc
                if not isinstance(sample, dict):
                    raise DatasetFilterError(f"Line {line_no} in '{input_path}' must be a JSON object.")

                empty_positions = _empty_passage_positions(sample)
                if drop_empty_passages and empty_positions:
                    entry = _empty_passage_report_entry(sample, line_no, empty_positions)
                    empty_report_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    empty_passage_samples += 1
                    empty_positive_samples += int("pos" in empty_positions)
                    empty_negative_samples += int("neg" in empty_positions)
                    if len(empty_passage_identifiers) < EMPTY_PASSAGE_WARNING_ID_LIMIT:
                        empty_passage_identifiers.append(entry)
                    continue

                keep, sample_stats, sample_missing_counts, sample_type_mismatch_counts = _filter_sample(
                    sample,
                    profile=profile,
                    profile_path=profile_path,
                    line_no=line_no,
                    profile_name=profile_name,
                )
                positives_total += sample_stats["positives_total"]
                positives_kept += sample_stats["positives_kept"]
                negatives_total += sample_stats["negatives_total"]
                negatives_kept += sample_stats["negatives_kept"]
                removed_by_min_positives += sample_stats["removed_by_min_positives"]
                removed_by_min_negatives += sample_stats["removed_by_min_negatives"]
                missing_counts.update(sample_missing_counts)
                type_mismatch_counts.update(sample_type_mismatch_counts)
                if keep:
                    kept += 1
                    out_fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    except Exception:
        if output_path.exists():
            output_path.unlink()
        if empty_passage_report_path.exists():
            empty_passage_report_path.unlink()
        raise

    report = {
        "profile_name": profile_name,
        "profile_path": str(profile_path),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total": total,
        "kept": kept,
        "removed": total - kept,
        "positives_total": positives_total,
        "positives_kept": positives_kept,
        "positives_removed": positives_total - positives_kept,
        "negatives_total": negatives_total,
        "negatives_kept": negatives_kept,
        "negatives_removed": negatives_total - negatives_kept,
        "removed_by_min_positives": removed_by_min_positives,
        "removed_by_min_negatives": removed_by_min_negatives,
        "missing_counts": dict(sorted(missing_counts.items())),
        "type_mismatch_counts": dict(sorted(type_mismatch_counts.items())),
        "empty_passages": {
            "enabled": drop_empty_passages,
            "removed_samples": empty_passage_samples,
            "samples_with_empty_pos": empty_positive_samples,
            "samples_with_empty_neg": empty_negative_samples,
            "detail_report_path": str(empty_passage_report_path),
            "identifiers": empty_passage_identifiers,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_report(report, cache_hit=False)
    return FilteredDatasetResult(input_path, output_dir, output_path, report_path, report, cache_hit=False)


def _merged_filter_settings(*settings: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for section in settings:
        if isinstance(section, dict):
            for key in FILTER_CONFIG_KEYS | HF_DATASET_CACHE_CONFIG_KEYS:
                if key in section:
                    merged[key] = section[key]
    return merged


def apply_dataset_filter_if_configured(
    train_data: str | Path,
    *settings: dict[str, Any] | None,
    config_path: str | None = None,
) -> FilteredDatasetResult | None:
    merged = _merged_filter_settings(*settings)
    dataset_filter = merged.get("dataset_filter")
    if not dataset_filter:
        return None
    return materialize_filtered_dataset(
        train_data,
        dataset_filter,
        cache_dir=merged.get("dataset_filter_cache_dir"),
        config_path=config_path,
        source_settings=merged,
    )
