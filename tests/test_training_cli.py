import builtins
import importlib
import json
import sys
from types import SimpleNamespace

import pytest


OPTIONAL_BACKENDS = {"FlagEmbedding", "sentence_transformers", "pylate"}


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

    class FakeSparseEncoder:
        def __init__(self, model_name_or_path, **model_kwargs):
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
        lambda: (FakeDataset, FakeSparseEncoder, FakeTrainer, FakeArgs, FakeSparseRankingLoss, FakeSpladeLoss),
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
