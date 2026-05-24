import builtins
import importlib
import json
from pathlib import Path
import sys
from textwrap import dedent
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OPTIONAL_BACKENDS = {"FlagEmbedding", "sentence_transformers", "pylate"}


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _write_filter_profile(path, body):
    path.write_text(dedent(body).strip() + "\n", encoding="utf-8")
    return path


def _filtered_queries(output_path):
    return [json.loads(line)["query"] for line in output_path.read_text(encoding="utf-8").splitlines() if line]


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_cli_import_does_not_import_optional_backends(monkeypatch):
    for module_name in list(sys.modules):
        if module_name.split(".", 1)[0] in OPTIONAL_BACKENDS:
            sys.modules.pop(module_name)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in OPTIONAL_BACKENDS:
            raise AssertionError(f"Unexpected optional backend import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("training.train")
    importlib.reload(module)


@pytest.mark.parametrize(
    ("backend", "training_type"),
    [
        ("flagembedding", "embedder"),
        ("sentence-transformers", "matryoshka"),
        ("sentence-transformers", "splade"),
        ("pylate", "colbert"),
    ],
)
def test_parser_accepts_supported_backend_and_training_type(backend, training_type):
    from training.train import build_parser

    args = build_parser().parse_args(
        [
            "--backend",
            backend,
            "--training-type",
            training_type,
            "--config",
            "configs/grid.yaml",
        ]
    )

    assert args.backend == backend
    assert args.training_type == training_type


def test_unsupported_backend_training_type_combination_is_rejected():
    from training.backends.registry import UnsupportedTrainingCombination, validate_backend_training_type

    with pytest.raises(UnsupportedTrainingCombination) as exc:
        validate_backend_training_type("pylate", "splade")

    assert "Training type 'splade' is not supported by backend 'pylate'" in str(exc.value)
    assert "Supported types: colbert, late-interaction" in str(exc.value)


def test_backend_module_is_loaded_lazily_for_selected_backend(monkeypatch, tmp_path):
    import training.train as train

    config = tmp_path / "train.yaml"
    config.write_text("backend: pylate\ntraining_type: colbert\n", encoding="utf-8")
    args = train.build_parser().parse_args(["--config", str(config)])
    loaded_modules = []
    requests = []

    def fake_load_backend_module(spec):
        loaded_modules.append(spec.module)
        return SimpleNamespace(run_training=lambda request: requests.append(request) or 0)

    monkeypatch.setattr(train, "load_backend_module", fake_load_backend_module)

    assert train.run_training(args) == 0
    assert loaded_modules == ["training.backends.pylate_backend"]
    assert requests[0].backend == "pylate"
    assert requests[0].training_type == "colbert"


def test_shared_grid_expands_for_sentence_transformers(tmp_path):
    from training.config_grid import expand_config_grid

    variants = expand_config_grid(
        {
            "backend": "sentence-transformers",
            "training_type": "splade",
            "runs_dir": str(tmp_path / "runs"),
            "train_data": "dataset-small-no_in_batch_neg",
            "architectures": ["sdadas/polish-distilroberta", "allegro/herbert-base-cased"],
            "hparams": [{"learning_rate": 2e-6, "num_train_epochs": 1}],
            "sentence_transformers": {"train_batch_size": 2},
        },
        backend="sentence-transformers",
        training_type="splade",
    )

    assert len(variants) == 2
    assert variants[0]["model_name_or_path"] == "sdadas/polish-distilroberta"
    assert variants[0]["learning_rate"] == 2e-6
    assert variants[0]["sentence_transformers"]["model_name_or_path"] == "sdadas/polish-distilroberta"
    assert variants[0]["sentence_transformers"]["learning_rate"] == 2e-6
    assert variants[0]["sentence_transformers"]["train_batch_size"] == 2
    assert variants[0]["output_dir"].startswith(str(tmp_path / "runs" / "sentence-transformers" / "splade"))
    assert variants[1]["model_name_or_path"] == "allegro/herbert-base-cased"


def test_run_training_expands_grid_before_backend_call(monkeypatch, tmp_path):
    import training.train as train

    config = tmp_path / "train.yaml"
    config.write_text(
        dedent(
            """
            backend: sentence-transformers
            training_type: splade
            runs_dir: runs/test-grid
            train_data: dataset-small-no_in_batch_neg
            architectures:
              - model-a
              - model-b
            hparams:
              - learning_rate: 2e-6
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    args = train.build_parser().parse_args(["--config", str(config)])
    requests = []

    def fake_load_backend_module(spec):
        return SimpleNamespace(run_training=lambda request: requests.append(request) or 0)

    monkeypatch.setattr(train, "load_backend_module", fake_load_backend_module)

    assert train.run_training(args) == 0
    assert [request.config["model_name_or_path"] for request in requests] == ["model-a", "model-b"]
    assert all(float(request.config["learning_rate"]) == 2e-6 for request in requests)
    assert requests[0].config["output_dir"] != requests[1].config["output_dir"]


def test_missing_backend_dependency_has_readable_error(monkeypatch):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import BackendDependencyError, TrainingRequest

    original_find_spec = sentence_transformers_backend.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "sentence_transformers":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(sentence_transformers_backend.importlib.util, "find_spec", fake_find_spec)

    request = TrainingRequest(
        backend="sentence-transformers",
        training_type="embedder",
        config={},
        config_path="configs/st_embedder.yaml",
        cli_args=SimpleNamespace(),
    )
    with pytest.raises(BackendDependencyError) as exc:
        sentence_transformers_backend.run_training(request)

    assert "Backend 'sentence-transformers' requires the sentence-transformers dependency" in str(exc.value)
    assert "pip install -r requirements/requirements-sentence-transformers.txt" in str(exc.value)


def test_flagembedding_backend_rewrites_train_data_with_dataset_filter(monkeypatch, tmp_path):
    from training.backends import flagembedding_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_jsonl(
        data_dir / "dataset.jsonl",
        [
            {
                "query": "keep",
                "pos": ["p1", "p2"],
                "neg": ["n1", "n2"],
                "features": {
                    "pos": [
                        {"ranks": {"cross_encoder_v1": 0.9}},
                        {"ranks": {"cross_encoder_v1": 0.7}},
                    ],
                    "neg": [
                        {"ranks": {"hard_negative_score": 0.8}},
                        {"ranks": {"hard_negative_score": 0.4}},
                    ],
                },
            },
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: flag_passage
        version: 1
        positive_rules:
          - field: ranks.cross_encoder_v1
            op: gte
            value: 0.8
        negative_rules:
          - field: ranks.hard_negative_score
            op: gte
            value: 0.6
        """,
    )
    calls = {"cmd": None}

    class FakeRun:
        def finish(self):
            pass

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            return FakeRun()

    def fake_subprocess_run(cmd, check, env):
        calls["cmd"] = cmd

    monkeypatch.setattr(flagembedding_backend, "_require_flagembedding", lambda: None)
    monkeypatch.setattr(flagembedding_backend, "_load_wandb", lambda: FakeWandb)
    monkeypatch.setattr(flagembedding_backend.subprocess, "run", fake_subprocess_run)

    request = TrainingRequest(
        backend="flagembedding",
        training_type="embedder",
        config={
            "runs_dir": str(tmp_path / "runs"),
            "architectures": ["tiny-model"],
            "hparams": [
                {
                    "learning_rate": 1e-5,
                    "num_train_epochs": 1,
                    "train_data": str(data_dir),
                    "dataset_filter": str(profile),
                    "dataset_filter_cache_dir": str(tmp_path / "cache"),
                    "knowledge_distillation": False,
                }
            ],
        },
        config_path=str(tmp_path / "config.yaml"),
        cli_args=SimpleNamespace(run_mteb=False, run_pirb=False, remove_checkpoints=False),
    )

    assert flagembedding_backend.run_training(request) == 0
    cmd = calls["cmd"]
    assert cmd is not None
    assert "--dataset-filter" not in cmd
    assert "--dataset-filter-cache-dir" not in cmd
    train_data_arg = cmd[cmd.index("--train-data") + 1]
    filtered_dir = tmp_path / "cache" / Path(train_data_arg).name
    assert Path(train_data_arg) == filtered_dir
    assert _filtered_queries(filtered_dir / "dataset.jsonl") == ["keep"]
    filtered_record = _read_jsonl(filtered_dir / "dataset.jsonl")[0]
    assert filtered_record["pos"] == ["p1"]
    assert filtered_record["neg"] == ["n1"]


def test_sentence_transformers_dataset_loader_expands_flagembedding_jsonl(tmp_path):
    from training.backends.sentence_transformers_backend import load_flagembedding_jsonl_dataset

    data_file = tmp_path / "dataset.jsonl"
    data_file.write_text(
        json.dumps(
            {
                "query": "question",
                "pos": ["positive one", "positive two"],
                "neg": ["negative one", "negative two", "negative three"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_flagembedding_jsonl_dataset(
        data_file,
        negatives_per_query=2,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )

    assert rows == [
        {
            "anchor": "query: question",
            "positive": "passage: positive one",
            "negative_1": "passage: negative one",
            "negative_2": "passage: negative two",
        },
        {
            "anchor": "query: question",
            "positive": "passage: positive two",
            "negative_1": "passage: negative one",
            "negative_2": "passage: negative two",
        },
    ]


def test_dataset_filter_gte_rank(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "keep", "pos": ["p"], "neg": ["n"], "features": {"ranks": {"cross_encoder_v1": 0.91}}},
            {"query": "drop", "pos": ["p"], "neg": ["n"], "features": {"ranks": {"cross_encoder_v1": 0.79}}},
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: rank_gte
        version: 1
        rules:
          - field: features.ranks.cross_encoder_v1
            op: gte
            value: 0.8
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert _filtered_queries(result.output_path) == ["keep"]
    assert result.report["total"] == 2
    assert result.report["kept"] == 1


def test_dataset_filter_eq_flag(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "keep", "pos": ["p"], "neg": ["n"], "features": {"flags": {"is_synthetic": False}}},
            {"query": "drop", "pos": ["p"], "neg": ["n"], "features": {"flags": {"is_synthetic": True}}},
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: flag_eq
        version: 1
        rules:
          - field: features.flags.is_synthetic
            op: eq
            value: false
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert _filtered_queries(result.output_path) == ["keep"]


def test_dataset_filter_in_category(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "keep", "pos": ["p"], "neg": ["n"], "features": {"categories": {"language": "pl"}}},
            {"query": "drop", "pos": ["p"], "neg": ["n"], "features": {"categories": {"language": "de"}}},
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: category_in
        version: 1
        rules:
          - field: features.categories.language
            op: in
            values: ["pl", "en"]
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert _filtered_queries(result.output_path) == ["keep"]


def test_dataset_filter_intersects_category_list(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "query": "keep",
                "pos": ["p"],
                "neg": ["n"],
                "features": {"category_lists": {"domains": ["medical", "qa"]}},
            },
            {
                "query": "drop",
                "pos": ["p"],
                "neg": ["n"],
                "features": {"category_lists": {"domains": ["finance"]}},
            },
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: domains_intersects
        version: 1
        rules:
          - field: features.category_lists.domains
            op: intersects
            values: ["medical", "legal"]
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert _filtered_queries(result.output_path) == ["keep"]


def test_dataset_filter_any_two_ranks(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "cross", "pos": ["p"], "neg": ["n"], "features": {"ranks": {"cross_encoder_v1": 0.82}}},
            {"query": "teacher", "pos": ["p"], "neg": ["n"], "features": {"ranks": {"teacher_score": 0.76}}},
            {
                "query": "drop",
                "pos": ["p"],
                "neg": ["n"],
                "features": {"ranks": {"cross_encoder_v1": 0.7, "teacher_score": 0.7}},
            },
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: any_ranks
        version: 1
        rules:
          - any:
              - field: features.ranks.cross_encoder_v1
                op: gte
                value: 0.8
              - field: features.ranks.teacher_score
                op: gte
                value: 0.75
            missing_policy: exclude
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert _filtered_queries(result.output_path) == ["cross", "teacher"]
    assert result.report["missing_counts"] == {
        "sample:features.ranks.cross_encoder_v1": 1,
        "sample:features.ranks.teacher_score": 1,
    }


def test_dataset_filter_aggregate_max(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "keep", "pos": ["p"], "neg": ["n"], "features": {"ranks": {"a": 0.86, "b": 0.2}}},
            {"query": "drop", "pos": ["p"], "neg": ["n"], "features": {"ranks": {"a": 0.4, "b": 0.7}}},
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: aggregate_max
        version: 1
        rules:
          - field: features.ranks
            aggregate: max
            op: gte
            value: 0.85
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert _filtered_queries(result.output_path) == ["keep"]


def test_dataset_filter_missing_policy_fail(tmp_path):
    from training.dataset_filters import DatasetFilterError, materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(data_file, [{"query": "q", "pos": ["p"], "neg": ["n"], "features": {"ranks": {}}}])
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: missing_fail
        version: 1
        rules:
          - field: features.ranks.teacher_score
            op: gte
            value: 0.75
        """,
    )

    with pytest.raises(DatasetFilterError) as exc:
        materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert "missing field 'sample:features.ranks.teacher_score'" in str(exc.value)


def test_dataset_filter_missing_policy_exclude(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "missing", "pos": ["p"], "neg": ["n"], "features": {"ranks": {}}},
            {"query": "keep", "pos": ["p"], "neg": ["n"], "features": {"ranks": {"teacher_score": 0.8}}},
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: missing_exclude
        version: 1
        missing_policy: exclude
        rules:
          - field: features.ranks.teacher_score
            op: gte
            value: 0.75
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert _filtered_queries(result.output_path) == ["keep"]
    assert result.report["missing_counts"] == {"sample:features.ranks.teacher_score": 1}


def test_dataset_filter_positive_rules_trim_passages_and_parallel_metadata(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "query": "q",
                "pos": ["p1", "p2", "p3"],
                "neg": ["n1"],
                "pos_scores": [0.9, 0.7, 0.95],
                "pos_id": ["p1-id", "p2-id", "p3-id"],
                "features": {
                    "pos": [
                        {"ranks": {"cross_encoder_v1": 0.9}},
                        {"ranks": {"cross_encoder_v1": 0.7}},
                        {"ranks": {"cross_encoder_v1": 0.95}},
                    ],
                    "neg": [{"ranks": {"hard_negative_score": 0.5}}],
                },
            }
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: positive_trim
        version: 1
        positive_rules:
          - field: ranks.cross_encoder_v1
            op: gte
            value: 0.8
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")
    record = _read_jsonl(result.output_path)[0]

    assert record["pos"] == ["p1", "p3"]
    assert record["pos_scores"] == [0.9, 0.95]
    assert record["pos_id"] == ["p1-id", "p3-id"]
    assert record["features"]["pos"] == [
        {"ranks": {"cross_encoder_v1": 0.9}},
        {"ranks": {"cross_encoder_v1": 0.95}},
    ]
    assert result.report["positives_total"] == 3
    assert result.report["positives_kept"] == 2
    assert result.report["positives_removed"] == 1


def test_dataset_filter_negative_rules_trim_passages_and_parallel_metadata(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "query": "q",
                "pos": ["p1"],
                "neg": ["n1", "n2", "n3"],
                "neg_scores": [0.7, 0.4, 0.9],
                "neg_id": ["n1-id", "n2-id", "n3-id"],
                "features": {
                    "pos": [{"ranks": {"cross_encoder_v1": 0.9}}],
                    "neg": [
                        {"ranks": {"hard_negative_score": 0.7}},
                        {"ranks": {"hard_negative_score": 0.4}},
                        {"ranks": {"hard_negative_score": 0.9}},
                    ],
                },
            }
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: negative_trim
        version: 1
        negative_rules:
          - field: ranks.hard_negative_score
            op: gte
            value: 0.6
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")
    record = _read_jsonl(result.output_path)[0]

    assert record["neg"] == ["n1", "n3"]
    assert record["neg_scores"] == [0.7, 0.9]
    assert record["neg_id"] == ["n1-id", "n3-id"]
    assert record["features"]["neg"] == [
        {"ranks": {"hard_negative_score": 0.7}},
        {"ranks": {"hard_negative_score": 0.9}},
    ]
    assert result.report["negatives_total"] == 3
    assert result.report["negatives_kept"] == 2
    assert result.report["negatives_removed"] == 1


def test_dataset_filter_drops_sample_below_min_positives(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "query": "q",
                "pos": ["p1", "p2"],
                "neg": ["n1"],
                "features": {
                    "pos": [
                        {"ranks": {"cross_encoder_v1": 0.9}},
                        {"ranks": {"cross_encoder_v1": 0.7}},
                    ]
                },
            }
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: min_pos
        version: 1
        min_positives: 2
        positive_rules:
          - field: ranks.cross_encoder_v1
            op: gte
            value: 0.8
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert _read_jsonl(result.output_path) == []
    assert result.report["removed_by_min_positives"] == 1


def test_dataset_filter_drops_sample_below_min_negatives(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "query": "q",
                "pos": ["p1"],
                "neg": ["n1", "n2"],
                "features": {
                    "neg": [
                        {"ranks": {"hard_negative_score": 0.8}},
                        {"ranks": {"hard_negative_score": 0.2}},
                    ]
                },
            }
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: min_neg
        version: 1
        min_negatives: 2
        negative_rules:
          - field: ranks.hard_negative_score
            op: gte
            value: 0.6
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert _read_jsonl(result.output_path) == []
    assert result.report["removed_by_min_negatives"] == 1


def test_dataset_filter_rules_still_alias_sample_rules(tmp_path):
    from training.dataset_filters import materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "keep", "pos": ["p"], "neg": ["n"], "features": {"query": {"categories": {"language": "pl"}}}},
            {"query": "drop", "pos": ["p"], "neg": ["n"], "features": {"query": {"categories": {"language": "de"}}}},
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: rules_alias
        version: 1
        rules:
          - field: features.query.categories.language
            op: eq
            value: pl
        """,
    )

    result = materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert _filtered_queries(result.output_path) == ["keep"]


def test_dataset_filter_passage_missing_policy_fail(tmp_path):
    from training.dataset_filters import DatasetFilterError, materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [{"query": "q", "pos": ["p1"], "neg": ["n1"], "features": {"pos": [{}]}}],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: passage_missing
        version: 1
        positive_rules:
          - field: ranks.cross_encoder_v1
            op: gte
            value: 0.8
        """,
    )

    with pytest.raises(DatasetFilterError) as exc:
        materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert "missing field 'pos:ranks.cross_encoder_v1'" in str(exc.value)


def test_dataset_filter_rejects_mismatched_parallel_metadata(tmp_path):
    from training.dataset_filters import DatasetFilterError, materialize_filtered_dataset

    data_file = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_file,
        [{"query": "q", "pos": ["p1", "p2"], "neg": ["n1"], "features": {"pos": [{"ranks": {"a": 1.0}}]}}],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: mismatch
        version: 1
        positive_rules:
          - field: ranks.a
            op: gte
            value: 0.5
        """,
    )

    with pytest.raises(DatasetFilterError) as exc:
        materialize_filtered_dataset(data_file, profile, cache_dir=tmp_path / "cache")

    assert "field 'features.pos' has 1 items but expected 2" in str(exc.value)


def test_sentence_transformers_embedder_runs_training_with_mocks(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1", "n2"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    calls = {"trained": False, "saved": None, "rows": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            calls["rows"] = rows
            return rows

    class FakeModel:
        def __init__(self, model_name_or_path, **model_kwargs):
            self.model_name_or_path = model_name_or_path
            self.model_kwargs = model_kwargs
            self.max_seq_length = None

        def save_pretrained(self, output_path):
            calls["saved"] = output_path

    class FakeArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLoss:
        def __init__(self, model):
            self.model = model

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.loss = loss

        def train(self):
            calls["trained"] = True

    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_sentence_transformers",
        lambda: (FakeDataset, FakeModel, FakeTrainer, FakeArgs, FakeLoss, object),
    )

    request = TrainingRequest(
        backend="sentence-transformers",
        training_type="embedder",
        config={
            "train_data": str(data_dir),
            "output_dir": str(output_dir),
            "sentence_transformers": {
                "model_name_or_path": "tiny-model",
                "max_steps": 1,
                "train_batch_size": 1,
                "negatives_per_query": 1,
                "max_seq_length": 64,
                "save_strategy": "no",
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["trained"] is True
    assert calls["saved"] == str(output_dir / "final")
    assert calls["rows"] == [{"anchor": "q", "positive": "p", "negative_1": "n1"}]


def test_sentence_transformers_embedder_uses_filtered_records(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_jsonl(
        data_dir / "dataset.jsonl",
        [
            {
                "query": "keep",
                "pos": ["p1", "p2"],
                "neg": ["n1", "n2"],
                "features": {
                    "query": {"categories": {"language": "pl"}},
                    "pos": [
                        {"ranks": {"cross_encoder_v1": 0.9}},
                        {"ranks": {"cross_encoder_v1": 0.7}},
                    ],
                    "neg": [
                        {"ranks": {"hard_negative_score": 0.8}},
                        {"ranks": {"hard_negative_score": 0.4}},
                    ],
                },
            },
            {
                "query": "drop",
                "pos": ["p"],
                "neg": ["n1"],
                "features": {"query": {"categories": {"language": "de"}}},
            },
        ],
    )
    profile = _write_filter_profile(
        tmp_path / "filter.yaml",
        """
        name: st_language
        version: 1
        sample_rules:
          - field: features.query.categories.language
            op: in
            values: ["pl"]
        positive_rules:
          - field: ranks.cross_encoder_v1
            op: gte
            value: 0.8
        negative_rules:
          - field: ranks.hard_negative_score
            op: gte
            value: 0.6
        """,
    )
    output_dir = tmp_path / "out"
    calls = {"trained": False, "saved": None, "rows": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            calls["rows"] = rows
            return rows

    class FakeModel:
        def __init__(self, model_name_or_path, **model_kwargs):
            self.max_seq_length = None

        def save_pretrained(self, output_path):
            calls["saved"] = output_path

    class FakeArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLoss:
        def __init__(self, model):
            self.model = model

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            self.train_dataset = train_dataset

        def train(self):
            calls["trained"] = True

    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_sentence_transformers",
        lambda: (FakeDataset, FakeModel, FakeTrainer, FakeArgs, FakeLoss, object),
    )

    request = TrainingRequest(
        backend="sentence-transformers",
        training_type="embedder",
        config={
            "train_data": str(data_dir),
            "dataset_filter": str(profile),
            "dataset_filter_cache_dir": str(tmp_path / "cache"),
            "output_dir": str(output_dir),
            "sentence_transformers": {
                "model_name_or_path": "tiny-model",
                "max_steps": 1,
                "train_batch_size": 1,
                "negatives_per_query": 1,
                "save_strategy": "no",
            },
        },
        config_path=str(tmp_path / "config.yaml"),
        cli_args=SimpleNamespace(),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["trained"] is True
    assert calls["saved"] == str(output_dir / "final")
    assert calls["rows"] == [{"anchor": "keep", "positive": "p1", "negative_1": "n1"}]


def test_sentence_transformers_matryoshka_runs_training_with_mocks(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1", "n2"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    calls = {"trained": False, "saved": None, "dims": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeModel:
        def __init__(self, model_name_or_path, **model_kwargs):
            self.max_seq_length = None

        def get_sentence_embedding_dimension(self):
            return 4

        def save_pretrained(self, output_path):
            calls["saved"] = output_path

    class FakeArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRankingLoss:
        def __init__(self, model):
            self.model = model

    class FakeMatryoshkaLoss:
        def __init__(self, model, loss, matryoshka_dims, **kwargs):
            calls["dims"] = matryoshka_dims

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            self.loss = loss

        def train(self):
            calls["trained"] = True

    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_sentence_transformers",
        lambda: (FakeDataset, FakeModel, FakeTrainer, FakeArgs, FakeRankingLoss, FakeMatryoshkaLoss),
    )

    request = TrainingRequest(
        backend="sentence-transformers",
        training_type="matryoshka",
        config={
            "train_data": str(data_dir),
            "output_dir": str(output_dir),
            "sentence_transformers": {
                "model_name_or_path": "tiny-model",
                "max_steps": 1,
                "train_batch_size": 1,
                "negatives_per_query": 1,
                "matryoshka_dims": [4, 2],
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["trained"] is True
    assert calls["saved"] == str(output_dir / "final")
    assert calls["dims"] == [4, 2]


def test_sentence_transformers_splade_runs_training_with_mocks(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    calls = {"trained": False, "saved": None, "regularizers": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeMLMTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            self.model_name_or_path = model_name_or_path
            self.kwargs = kwargs

    class FakeSpladePooling:
        def __init__(self, pooling_strategy):
            self.pooling_strategy = pooling_strategy

    class FakeSparseEncoder:
        def __init__(self, modules):
            calls["modules"] = modules
            self.max_seq_length = None

        def save_pretrained(self, output_path):
            calls["saved"] = output_path

    class FakeArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSparseRankingLoss:
        def __init__(self, model, scale, gather_across_devices):
            self.model = model

    class FakeSpladeLoss:
        def __init__(self, model, loss, document_regularizer_weight, query_regularizer_weight):
            calls["regularizers"] = (document_regularizer_weight, query_regularizer_weight)

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            self.loss = loss

        def train(self):
            calls["trained"] = True

    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_sparse_sentence_transformers",
        lambda: (
            FakeDataset,
            FakeSparseEncoder,
            FakeTrainer,
            FakeArgs,
            FakeSparseRankingLoss,
            FakeSpladeLoss,
            FakeMLMTransformer,
            FakeSpladePooling,
        ),
    )

    request = TrainingRequest(
        backend="sentence-transformers",
        training_type="splade",
        config={
            "train_data": str(data_dir),
            "output_dir": str(output_dir),
            "sentence_transformers": {
                "model_name_or_path": "tiny-mlm",
                "max_steps": 1,
                "train_batch_size": 1,
                "processor_kwargs": {"use_fast": False},
                "document_regularizer_weight": 0.1,
                "query_regularizer_weight": 0.2,
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["trained"] is True
    assert calls["saved"] == str(output_dir / "final")
    assert calls["regularizers"] == (0.1, 0.2)
    assert calls["modules"][0].model_name_or_path == "tiny-mlm"
    assert calls["modules"][0].kwargs["processor_kwargs"] == {"use_fast": False}
    assert calls["modules"][1].pooling_strategy == "max"


def test_pylate_colbert_runs_training_with_mocks(monkeypatch, tmp_path):
    from training.backends import pylate_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1", "n2"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    calls = {"trained": False, "saved": None, "rows": None, "collator": False}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            calls["rows"] = rows
            return rows

    class FakeColBERT:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def tokenize(self, texts, **kwargs):
            return {"input_ids": texts}

        def save_pretrained(self, output_path):
            calls["saved"] = output_path

    class FakeLoss:
        def __init__(self, model, gather_across_devices, temperature):
            self.model = model

    class FakeCollator:
        def __init__(self, tokenize_fn):
            calls["collator"] = True

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss, data_collator):
            self.train_dataset = train_dataset

        def train(self):
            calls["trained"] = True

    class FakeArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.remove_unused_columns = True

    monkeypatch.setattr(
        pylate_backend,
        "_load_pylate_training_stack",
        lambda: (FakeDataset, FakeColBERT, FakeLoss, FakeCollator, FakeTrainer, FakeArgs),
    )

    request = TrainingRequest(
        backend="pylate",
        training_type="colbert",
        config={
            "train_data": str(data_dir),
            "output_dir": str(output_dir),
            "pylate": {
                "model_name_or_path": "tiny-model",
                "max_steps": 1,
                "train_batch_size": 1,
                "negatives_per_query": 1,
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )

    assert pylate_backend.run_training(request) == 0
    assert calls["trained"] is True
    assert calls["collator"] is True
    assert calls["saved"] == str(output_dir / "final")
    assert calls["rows"] == [{"anchor": "q", "positive": "p", "negative_1": "n1"}]


def test_run_experiments_train_mode_delegates_to_new_cli(monkeypatch):
    import run_experiments

    calls = []

    def fake_subprocess_run(cmd, check):
        calls.append((cmd, check))

    monkeypatch.setattr(run_experiments.subprocess, "run", fake_subprocess_run)

    result = run_experiments.main(
        [
            "--mode",
            "train",
            "--grid-configuration-file",
            "configs/grid.yaml",
            "--run_pirb",
            "--remove_checkpoints",
            "--pirb_scope",
            "small",
        ]
    )

    assert result == 0
    cmd, check = calls[0]
    assert check is True
    assert cmd[:3] == [sys.executable, "-m", "training.train"]
    assert "--backend" in cmd
    assert "flagembedding" in cmd
    assert "--training-type" in cmd
    assert "embedder" in cmd
    assert "--run-pirb" in cmd
    assert "--remove-checkpoints" in cmd
