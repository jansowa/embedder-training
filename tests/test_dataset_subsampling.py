import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.multi_dataset import (  # noqa: E402
    AFTER_FILTER,
    BEFORE_FILTER,
    DatasetSampling,
    MultiDatasetConfigError,
    materialize_mixed_jsonl_dataset,
    materialize_sampled_jsonl_dataset,
    normalize_train_data_specs,
    resolve_dataset_sources,
)


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def _training_rows(count, *, prefix="q", keep=True):
    return [
        {
            "query": f"{prefix}-{index}",
            "pos": [f"{prefix}-pos-{index}"],
            "neg": [f"{prefix}-neg-{index}"],
            "pos_scores": [1.0 if keep or index % 2 == 0 else 0.0],
        }
        for index in range(count)
    ]


def _queries(path):
    return [json.loads(line)["query"] for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_score_filter(path):
    path.write_text(
        "version: 1\n"
        "name: keep-scored-positives\n"
        "positive_rules:\n"
        "  - field: parallel.pos_scores\n"
        "    op: gt\n"
        "    value: 0.5\n",
        encoding="utf-8",
    )
    return path


def test_plain_string_sources_have_no_sampling():
    entries = normalize_train_data_specs(["a.jsonl", "b.jsonl"])
    assert [entry.path for entry in entries] == ["a.jsonl", "b.jsonl"]
    assert [entry.sampling for entry in entries] == [None, None]


def test_mapping_sources_carry_their_sampling_settings():
    entries = normalize_train_data_specs(
        [
            "a.jsonl",
            {"path": "b.jsonl", "max_samples": 10, "seed": 7, "sample_stage": "before_filter"},
        ]
    )
    assert entries[1].path == "b.jsonl"
    assert entries[1].sampling == DatasetSampling(max_samples=10, seed=7, stage=BEFORE_FILTER)


def test_sampling_defaults_to_the_run_seed_and_after_filter():
    entries = normalize_train_data_specs([{"path": "b.jsonl", "max_samples": 10}], {"seed": 123})
    assert entries[0].sampling == DatasetSampling(max_samples=10, seed=123, stage=AFTER_FILTER)


def test_sampling_seed_falls_back_to_the_training_default():
    entries = normalize_train_data_specs([{"path": "b.jsonl", "max_samples": 10}])
    assert entries[0].sampling.seed == 42


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"path": "b.jsonl", "max_samples": 0}, "positive integer"),
        ({"path": "b.jsonl", "max_samples": "many"}, "positive integer"),
        ({"path": "b.jsonl", "max_samples": 5, "sample_stage": "midway"}, "sample_stage"),
        ({"path": "b.jsonl", "seed": 1}, "requires"),
        ({"max_samples": 5}, "must set"),
    ],
)
def test_invalid_sampling_settings_are_rejected(entry, match):
    with pytest.raises(MultiDatasetConfigError, match=match):
        normalize_train_data_specs([entry])


def test_subsampling_is_deterministic_for_a_seed(tmp_path):
    source = _write_jsonl(tmp_path / "source.jsonl", _training_rows(20))
    sampling = DatasetSampling(max_samples=5, seed=11, stage=AFTER_FILTER)

    first = materialize_sampled_jsonl_dataset(source, sampling, cache_dir=tmp_path / "cache-a", label="source")
    second = materialize_sampled_jsonl_dataset(source, sampling, cache_dir=tmp_path / "cache-b", label="source")

    assert first.total == 20
    assert first.kept == 5
    assert _queries(first.output_path) == _queries(second.output_path)


def test_a_different_seed_selects_different_rows(tmp_path):
    source = _write_jsonl(tmp_path / "source.jsonl", _training_rows(50))

    first = materialize_sampled_jsonl_dataset(
        source, DatasetSampling(10, 1, AFTER_FILTER), cache_dir=tmp_path / "cache", label="source"
    )
    second = materialize_sampled_jsonl_dataset(
        source, DatasetSampling(10, 2, AFTER_FILTER), cache_dir=tmp_path / "cache", label="source"
    )

    assert _queries(first.output_path) != _queries(second.output_path)


def test_selected_rows_keep_their_original_order(tmp_path):
    source = _write_jsonl(tmp_path / "source.jsonl", _training_rows(20))
    result = materialize_sampled_jsonl_dataset(
        source, DatasetSampling(6, 3, AFTER_FILTER), cache_dir=tmp_path / "cache", label="source"
    )

    selected = _queries(result.output_path)
    original = _queries(source)
    assert selected == [query for query in original if query in set(selected)]


def test_subsampling_is_skipped_when_the_source_is_small_enough(tmp_path):
    source = _write_jsonl(tmp_path / "source.jsonl", _training_rows(3))
    result = materialize_sampled_jsonl_dataset(
        source, DatasetSampling(10, 1, AFTER_FILTER), cache_dir=tmp_path / "cache", label="source"
    )

    assert result.output_path == source
    assert (result.total, result.kept) == (3, 3)


def test_subsampling_is_cached_per_input_and_settings(tmp_path, capsys):
    source = _write_jsonl(tmp_path / "source.jsonl", _training_rows(20))
    sampling = DatasetSampling(5, 11, AFTER_FILTER)

    materialize_sampled_jsonl_dataset(source, sampling, cache_dir=tmp_path / "cache", label="source")
    capsys.readouterr()
    second = materialize_sampled_jsonl_dataset(source, sampling, cache_dir=tmp_path / "cache", label="source")

    assert second.cache_hit is True
    assert "cache hit" in capsys.readouterr().out


def test_sampling_report_records_the_row_counts(tmp_path):
    source = _write_jsonl(tmp_path / "source.jsonl", _training_rows(20))
    result = materialize_sampled_jsonl_dataset(
        source, DatasetSampling(5, 11, AFTER_FILTER), cache_dir=tmp_path / "cache", label="source"
    )

    report = json.loads((result.output_path.parent / "sample_report.json").read_text(encoding="utf-8"))
    assert report["total"] == 20
    assert report["kept"] == 5
    assert report["seed"] == 11
    assert report["sample_stage"] == AFTER_FILTER


def test_row_counts_before_and_after_are_logged(tmp_path, capsys):
    source = _write_jsonl(tmp_path / "source.jsonl", _training_rows(20))
    materialize_sampled_jsonl_dataset(
        source, DatasetSampling(5, 11, AFTER_FILTER), cache_dir=tmp_path / "cache", label="source"
    )

    output = capsys.readouterr().out
    assert "rows_before=20" in output
    assert "rows_after=5" in output


def test_after_filter_sampling_counts_the_filtered_rows(tmp_path):
    source = _write_jsonl(tmp_path / "source.jsonl", _training_rows(20, keep=False))
    settings = {
        "dataset_filter": str(_write_score_filter(tmp_path / "filter.yaml")),
        "dataset_filter_cache_dir": str(tmp_path / "filtered"),
        "sampled_dataset_cache_dir": str(tmp_path / "sampled"),
    }

    sources = resolve_dataset_sources(
        [{"path": str(source), "max_samples": 4, "seed": 5, "sample_stage": AFTER_FILTER}],
        settings,
    )

    # The filter keeps the 10 even-indexed rows, and the subsample takes 4 of them.
    assert len(_queries(sources[0].path)) == 4
    report = json.loads((sources[0].path.parent / "sample_report.json").read_text(encoding="utf-8"))
    assert report["total"] == 10


def test_before_filter_sampling_counts_the_unfiltered_rows(tmp_path):
    source = _write_jsonl(tmp_path / "source.jsonl", _training_rows(20, keep=False))
    settings = {
        "dataset_filter": str(_write_score_filter(tmp_path / "filter.yaml")),
        "dataset_filter_cache_dir": str(tmp_path / "filtered"),
        "sampled_dataset_cache_dir": str(tmp_path / "sampled"),
    }

    sources = resolve_dataset_sources(
        [{"path": str(source), "max_samples": 4, "seed": 5, "sample_stage": BEFORE_FILTER}],
        settings,
    )

    report_path = next((tmp_path / "sampled").glob("*/sample_report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total"] == 20
    assert report["kept"] == 4
    # Filtering runs after the subsample, so at most the sampled rows survive.
    assert len(_queries(sources[0].path)) <= 4


def test_only_the_configured_source_is_subsampled(tmp_path):
    first = _write_jsonl(tmp_path / "first.jsonl", _training_rows(8, prefix="a"))
    second = _write_jsonl(tmp_path / "second.jsonl", _training_rows(30, prefix="b"))

    sources = resolve_dataset_sources(
        [str(first), {"path": str(second), "max_samples": 6}],
        {"sampled_dataset_cache_dir": str(tmp_path / "sampled")},
    )

    assert sources[0].path == first
    assert sources[0].sampling is None
    assert len(_queries(sources[1].path)) == 6


def test_mix_cache_key_changes_with_the_sampling_settings(tmp_path):
    first = _write_jsonl(tmp_path / "first.jsonl", _training_rows(8, prefix="a"))
    second = _write_jsonl(tmp_path / "second.jsonl", _training_rows(30, prefix="b"))
    settings = {"sampled_dataset_cache_dir": str(tmp_path / "sampled")}
    cache_dir = tmp_path / "mixed"

    small = materialize_mixed_jsonl_dataset(
        [str(first), {"path": str(second), "max_samples": 6}], settings, cache_dir=cache_dir
    )
    large = materialize_mixed_jsonl_dataset(
        [str(first), {"path": str(second), "max_samples": 12}], settings, cache_dir=cache_dir
    )

    assert small.output_dir != large.output_dir
    assert len(_queries(small.output_path)) == 14
    assert len(_queries(large.output_path)) == 20


def test_mix_reuses_its_cache_for_identical_sampling(tmp_path):
    first = _write_jsonl(tmp_path / "first.jsonl", _training_rows(8, prefix="a"))
    second = _write_jsonl(tmp_path / "second.jsonl", _training_rows(30, prefix="b"))
    settings = {"sampled_dataset_cache_dir": str(tmp_path / "sampled")}
    train_data = [str(first), {"path": str(second), "max_samples": 6, "seed": 9}]

    materialize_mixed_jsonl_dataset(train_data, settings, cache_dir=tmp_path / "mixed")
    again = materialize_mixed_jsonl_dataset(train_data, settings, cache_dir=tmp_path / "mixed")

    assert again.cache_hit is True


def test_mix_logs_the_rows_per_source(tmp_path, capsys):
    first = _write_jsonl(tmp_path / "first.jsonl", _training_rows(8, prefix="a"))
    second = _write_jsonl(tmp_path / "second.jsonl", _training_rows(30, prefix="b"))

    materialize_mixed_jsonl_dataset(
        [str(first), {"path": str(second), "max_samples": 6}],
        {"sampled_dataset_cache_dir": str(tmp_path / "sampled")},
        cache_dir=tmp_path / "mixed",
    )

    output = capsys.readouterr().out
    assert "0-first-jsonl: rows=8" in output
    assert "1-second-jsonl: rows=6" in output
