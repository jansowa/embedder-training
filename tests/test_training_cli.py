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


def _write_complete_checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": 1}), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


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


def test_parser_accepts_resume_flags():
    from training.train import build_parser

    resume_args = build_parser().parse_args(["--config", "configs/grid.yaml", "--resume"])
    checkpoint_args = build_parser().parse_args(
        ["--config", "configs/grid.yaml", "--resume-from-checkpoint", "runs/model/checkpoint-10"]
    )

    assert resume_args.resume is True
    assert resume_args.resume_from_checkpoint is None
    assert checkpoint_args.resume is False
    assert checkpoint_args.resume_from_checkpoint == "runs/model/checkpoint-10"


def test_parser_accepts_distributed_gpu_flags():
    from training.train import build_parser

    gpu_args = build_parser().parse_args(["--config", "configs/grid.yaml", "--gpus", "2,3"])
    count_args = build_parser().parse_args(["--config", "configs/grid.yaml", "--num-gpus", "2"])
    disabled_args = build_parser().parse_args(["--config", "configs/grid.yaml", "--no-distributed"])

    assert gpu_args.gpus == "2,3"
    assert gpu_args.num_gpus is None
    assert count_args.gpus is None
    assert count_args.num_gpus == 2
    assert disabled_args.no_distributed is True


def test_parser_accepts_config_override_flags():
    from training.train import build_parser

    args = build_parser().parse_args(
        [
            "--config",
            "configs/grid.yaml",
            "--set",
            "train_data=dataset-a",
            "--set",
            "sentence_transformers.train_batch_size=4",
            "--set-str",
            "sentence_transformers.save_strategy=no",
        ]
    )

    assert args.config_overrides == ["train_data=dataset-a", "sentence_transformers.train_batch_size=4"]
    assert args.config_string_overrides == ["sentence_transformers.save_strategy=no"]


def test_parser_accepts_benchmark_parameter_flags():
    from training.train import build_parser

    args = build_parser().parse_args(
        [
            "--config",
            "configs/grid.yaml",
            "--run-pirb",
            "--run-mteb",
            "--benchmark-name",
            "NanoBEIR",
            "--benchmark-output-dir",
            "runs/benchmarks",
            "--benchmark-batch-size",
            "32",
            "--benchmark-query-instruction",
            "Pytanie: ",
            "--pirb-scope",
            "small",
            "--pirb-max-seq-length",
            "384",
        ]
    )

    assert args.run_pirb is True
    assert args.run_mteb is True
    assert args.benchmark_name == "NanoBEIR"
    assert args.benchmark_output_dir == "runs/benchmarks"
    assert args.benchmark_batch_size == 32
    assert args.benchmark_query_instruction == "Pytanie: "
    assert args.pirb_scope == "small"
    assert args.pirb_max_seq_length == 384


def test_parser_accepts_dry_run_and_resolved_config_flags():
    from training.train import build_parser

    args = build_parser().parse_args(
        [
            "--config",
            "configs/grid.yaml",
            "--dry-run",
            "--print-config",
            "--no-save-resolved-config",
        ]
    )

    assert args.dry_run is True
    assert args.print_config is True
    assert args.save_resolved_config is False


def test_parser_rejects_conflicting_gpu_flags():
    from training.train import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--gpus", "0,1", "--num-gpus", "2"])


def test_parser_rejects_conflicting_resume_flags():
    from training.train import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--resume", "--resume-from-checkpoint", "runs/model/checkpoint-10"])


def test_apply_config_overrides_updates_and_adds_nested_values():
    import training.train as train

    config = {
        "sentence_transformers": {"train_batch_size": 8},
        "hparams": [{"learning_rate": "2e-6"}],
    }

    train.apply_config_overrides(
        config,
        [
            "train_data=dataset-a",
            "sentence_transformers.train_batch_size=4",
            "sentence_transformers.report_to=[wandb]",
            "hparams.0.num_train_epochs=3",
            "new_section.enabled=true",
        ],
        ["sentence_transformers.save_strategy=no"],
    )

    assert config["train_data"] == "dataset-a"
    assert config["sentence_transformers"]["train_batch_size"] == 4
    assert config["sentence_transformers"]["report_to"] == ["wandb"]
    assert config["sentence_transformers"]["save_strategy"] == "no"
    assert config["hparams"][0]["num_train_epochs"] == 3
    assert config["new_section"]["enabled"] is True


def test_apply_config_overrides_rejects_missing_assignment():
    import training.train as train

    with pytest.raises(train.ConfigError, match="PATH=VALUE"):
        train.apply_config_overrides({}, ["train_data"])


def test_benchmark_settings_resolve_from_config_and_cli(tmp_path):
    from training.benchmarks import resolve_benchmark_settings

    settings = resolve_benchmark_settings(
        {
            "benchmark": {
                "run_pirb": True,
                "name": "ConfigBenchmark",
                "scope": "small",
                "batch_size": 16,
                "max_seq_length": 384,
                "output_dir": str(tmp_path / "bench"),
                "query_instruction": "Config query: ",
            }
        },
        {},
        SimpleNamespace(
            run_mteb=True,
            run_pirb=False,
            benchmark_name="CliBenchmark",
            pirb_scope=None,
            benchmark_batch_size=None,
            pirb_max_seq_length=256,
            benchmark_output_dir=None,
            benchmark_query_instruction=None,
        ),
    )

    assert settings.run_mteb is True
    assert settings.run_pirb is True
    assert settings.benchmark_name == "CliBenchmark"
    assert settings.pirb_scope == "small"
    assert settings.batch_size == 16
    assert settings.pirb_max_seq_length == 256
    assert settings.output_dir == tmp_path / "bench"
    assert settings.query_instruction_for_retrieval == "Config query: "


def test_run_benchmarks_for_model_uses_explicit_parameters(monkeypatch, tmp_path):
    import convert_utils
    import training.benchmarks as benchmarks
    from training.benchmarks import BenchmarkSettings

    calls = {}

    def fake_run_mteb(st_dir, tasks, batch_size=64, output_folder=None):
        calls["mteb"] = {
            "st_dir": st_dir,
            "tasks": tasks,
            "batch_size": batch_size,
            "output_folder": output_folder,
        }
        return {"mean_ndcg_at_10": 0.5}

    def fake_run_pirb(st_dir, query_instruction_for_retrieval, max_seq_length=512, scope="tiny", output_dir=None, **kwargs):
        calls["pirb"] = {
            "st_dir": st_dir,
            "query_instruction_for_retrieval": query_instruction_for_retrieval,
            "max_seq_length": max_seq_length,
            "scope": scope,
            "output_dir": output_dir,
        }
        return {"pirb_average_ndcg@10": 0.25}

    monkeypatch.setattr(benchmarks, "_load_mteb_tasks", lambda benchmark_name: [f"task:{benchmark_name}"])
    monkeypatch.setattr(convert_utils, "run_mteb", fake_run_mteb)
    monkeypatch.setattr(convert_utils, "run_pirb", fake_run_pirb)

    metrics = benchmarks.run_benchmarks_for_model(
        "runs/model/final",
        BenchmarkSettings(
            run_mteb=True,
            run_pirb=True,
            benchmark_name="NanoBEIR",
            pirb_scope="small",
            batch_size=32,
            pirb_max_seq_length=384,
            query_instruction_for_retrieval="Pytanie: ",
            output_dir=tmp_path / "bench",
            log_to_wandb=False,
        ),
        metric_prefix="final/",
        label="final",
    )

    assert calls["mteb"]["batch_size"] == 32
    assert calls["mteb"]["output_folder"] == str(tmp_path / "bench" / "final" / "mteb")
    assert calls["pirb"]["query_instruction_for_retrieval"] == "Pytanie: "
    assert calls["pirb"]["max_seq_length"] == 384
    assert calls["pirb"]["scope"] == "small"
    assert calls["pirb"]["output_dir"] == str(tmp_path / "bench" / "final" / "pirb")
    assert metrics == {
        "final/mean_ndcg_at_10": 0.5,
        "final/pirb_average_ndcg@10": 0.25,
    }
    assert json.loads((tmp_path / "bench" / "final" / "metrics.json").read_text(encoding="utf-8")) == metrics


def test_resolve_benchmark_targets_selects_configured_checkpoints(tmp_path):
    from training.benchmarks import BenchmarkSettings, resolve_benchmark_targets

    output_dir = tmp_path / "out"
    (output_dir / "final").mkdir(parents=True)
    (output_dir / "epoch-checkpoints" / "epoch-0001-step-100").mkdir(parents=True)
    (output_dir / "epoch-checkpoints" / "epoch-0001-step-123").mkdir(parents=True)
    (output_dir / "epoch-checkpoints" / "epoch-0002-step-456").mkdir(parents=True)
    (output_dir / "checkpoint-20000").mkdir(parents=True)

    settings = BenchmarkSettings(
        run_mteb=False,
        run_pirb=True,
        benchmark_name="NanoBEIR",
        pirb_scope="small",
        batch_size=32,
        pirb_max_seq_length=384,
        query_instruction_for_retrieval="",
        checkpoints=("final", {"epoch": 1}, {"epoch": 2}, {"step": 20000}),
    )

    targets = resolve_benchmark_targets(output_dir, settings)

    assert [(target.label, target.path.relative_to(output_dir), target.step) for target in targets] == [
        ("final", Path("final"), 0),
        ("epoch-0001", Path("epoch-checkpoints") / "epoch-0001-step-123", 123),
        ("epoch-0002", Path("epoch-checkpoints") / "epoch-0002-step-456", 456),
        ("step-20000", Path("checkpoint-20000"), 20000),
    ]


def test_resolve_benchmark_targets_warns_and_skips_missing_checkpoints(caplog, tmp_path):
    import logging

    from training.benchmarks import BenchmarkSettings, resolve_benchmark_targets

    output_dir = tmp_path / "out"
    (output_dir / "final").mkdir(parents=True)
    settings = BenchmarkSettings(
        run_mteb=False,
        run_pirb=True,
        benchmark_name="NanoBEIR",
        pirb_scope="small",
        batch_size=32,
        pirb_max_seq_length=384,
        query_instruction_for_retrieval="",
        checkpoints=("final", {"epoch": 2}, {"step": 20000}),
    )

    caplog.set_level(logging.WARNING, logger="training.benchmarks")

    targets = resolve_benchmark_targets(output_dir, settings)

    assert [target.label for target in targets] == ["final"]
    assert "Selected benchmark checkpoint 'epoch-0002' was not found" in caplog.text
    assert "Selected benchmark checkpoint 'step-20000' was not found" in caplog.text


def test_distributed_config_selects_specific_gpu_ids(monkeypatch):
    from training.distributed import resolve_distributed_config

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    config = {"distributed": {"gpus": [2, 3]}}
    launch_config = resolve_distributed_config(
        backend="sentence-transformers",
        config=config,
        cli_args=SimpleNamespace(gpus=None, num_gpus=None, no_distributed=False),
    )

    assert launch_config.enabled is True
    assert launch_config.nproc_per_node == 2
    assert launch_config.cuda_visible_devices == "2,3"


def test_distributed_config_selects_first_visible_gpu_count(monkeypatch):
    from training.distributed import resolve_distributed_config

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    config = {"distributed": {"num_gpus": 2}}
    launch_config = resolve_distributed_config(
        backend="pylate",
        config=config,
        cli_args=SimpleNamespace(gpus=None, num_gpus=None, no_distributed=False),
    )

    assert launch_config.nproc_per_node == 2
    assert launch_config.cuda_visible_devices == "4,5"


def test_run_training_relaunches_sentence_transformers_with_torchrun(monkeypatch, tmp_path):
    import training.distributed as distributed
    import training.train as train

    config = tmp_path / "train.yaml"
    config.write_text(
        dedent(
            """
            backend: sentence-transformers
            training_type: splade
            timestamp_output_dir: true
            train_data: dataset-small-no_in_batch_neg
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    args = train.build_parser().parse_args(["--config", str(config), "--gpus", "2,3"])
    args._raw_argv = ["--config", str(config), "--gpus", "2,3"]
    calls = {}

    def fake_run(cmd, check, env):
        calls["cmd"] = cmd
        calls["check"] = check
        calls["env"] = env
        return SimpleNamespace(returncode=0)

    def fail_load_backend_module(spec):
        raise AssertionError("backend module should not be loaded by the parent launcher")

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(distributed.subprocess, "run", fake_run)
    monkeypatch.setattr(train, "load_backend_module", fail_load_backend_module)

    assert train.run_training(args) == 0
    assert calls["check"] is False
    assert calls["cmd"][:5] == ["torchrun", "--standalone", "--nproc_per_node", "2", "-m"]
    assert calls["cmd"][5:7] == ["training.train", "--config"]
    assert calls["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert "EMBEDDER_TRAINING_RUN_TIMESTAMP" in calls["env"]


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


def test_grid_can_append_timestamp_to_output_dirs(monkeypatch, tmp_path):
    import training.config_grid as config_grid

    monkeypatch.setattr(config_grid, "_timestamp_slug", lambda: "20260525-101112")

    variants = config_grid.expand_config_grid(
        {
            "backend": "sentence-transformers",
            "training_type": "splade",
            "runs_dir": str(tmp_path / "runs"),
            "timestamp_output_dir": True,
            "architectures": ["sdadas/polish-distilroberta"],
            "hparams": [{"learning_rate": 2e-6, "num_train_epochs": 1}],
            "sentence_transformers": {},
        },
        backend="sentence-transformers",
        training_type="splade",
    )

    assert variants[0]["run_timestamp"] == "20260525-101112"
    assert variants[0]["run_name"].endswith("-20260525-101112")
    assert variants[0]["output_dir"].endswith("sdadas_polish-distilroberta-lr-2e-06-ep-1-20260525-101112")
    assert variants[0]["sentence_transformers"]["output_dir"] == variants[0]["output_dir"]


def test_grid_resume_reuses_timestamped_output_dir(tmp_path):
    from training.config_grid import expand_config_grid

    run_dir = (
        tmp_path
        / "runs"
        / "sentence-transformers"
        / "splade"
        / "sdadas_polish-distilroberta-lr-2e-06-ep-1-20260525-101112"
    )
    (run_dir / "checkpoint-100").mkdir(parents=True)

    variants = expand_config_grid(
        {
            "backend": "sentence-transformers",
            "training_type": "splade",
            "runs_dir": str(tmp_path / "runs"),
            "timestamp_output_dir": True,
            "architectures": ["sdadas/polish-distilroberta"],
            "hparams": [{"learning_rate": 2e-6, "num_train_epochs": 1}],
            "sentence_transformers": {},
        },
        backend="sentence-transformers",
        training_type="splade",
        resume=True,
    )

    assert "run_timestamp" not in variants[0]
    assert variants[0]["run_name"] == run_dir.name
    assert variants[0]["output_dir"] == str(run_dir)
    assert variants[0]["sentence_transformers"]["output_dir"] == str(run_dir)


def test_grid_expands_train_data_groups(tmp_path):
    from training.config_grid import expand_config_grid

    variants = expand_config_grid(
        {
            "backend": "sentence-transformers",
            "training_type": "splade",
            "runs_dir": str(tmp_path / "runs"),
            "architectures": ["tiny-model"],
            "hparams": [{"learning_rate": 2e-6, "num_train_epochs": 1}],
            "train_data_groups": [
                {"name": "ab", "train_data": ["dataset-a", "dataset-b"]},
                {"name": "cd", "train_data": ["dataset-c", "dataset-d"]},
            ],
            "sentence_transformers": {"train_batch_size": 2},
        },
        backend="sentence-transformers",
        training_type="splade",
    )

    assert len(variants) == 2
    assert variants[0]["train_data"] == ["dataset-a", "dataset-b"]
    assert variants[0]["sentence_transformers"]["train_data"] == ["dataset-a", "dataset-b"]
    assert variants[0]["grid_train_data_group"]["name"] == "ab"
    assert "-data-ab" in variants[0]["run_name"]
    assert "-data-cd" in variants[1]["run_name"]
    assert variants[0]["output_dir"] != variants[1]["output_dir"]


def test_grid_rejects_train_data_groups_with_hparam_train_data(tmp_path):
    from training.config_grid import expand_config_grid

    with pytest.raises(ValueError, match="train_data_groups"):
        expand_config_grid(
            {
                "backend": "sentence-transformers",
                "training_type": "splade",
                "runs_dir": str(tmp_path / "runs"),
                "architectures": ["tiny-model"],
                "hparams": [{"learning_rate": 2e-6, "train_data": "dataset-a"}],
                "train_data_groups": ["dataset-b"],
            },
            backend="sentence-transformers",
            training_type="splade",
        )


def test_proportional_batch_sampler_keeps_quota_when_possible():
    from training.multi_dataset import ProportionalNoDuplicatesBatchSampler

    sampler = ProportionalNoDuplicatesBatchSampler(
        ["a"] * 6 + ["b"] * 2,
        [{f"a-{idx}"} for idx in range(6)] + [{f"b-{idx}"} for idx in range(2)],
        batch_size=4,
        drop_last=False,
        seed=13,
    )

    batches = list(sampler)
    assert len(batches) == 2
    for batch in batches:
        counts = {"a": 0, "b": 0}
        for index in batch:
            counts[sampler.dataset_ids[index]] += 1
        assert counts == {"a": 3, "b": 1}


def test_proportional_batch_sampler_breaks_quota_to_avoid_duplicates():
    from training.multi_dataset import ProportionalNoDuplicatesBatchSampler

    sampler = ProportionalNoDuplicatesBatchSampler(
        ["a", "a", "b", "b"],
        [{"x"}, {"y"}, {"x"}, {"y"}],
        batch_size=4,
        drop_last=False,
        seed=0,
    )

    batches = list(sampler)
    assert len(batches) == 2
    assert any({sampler.dataset_ids[index] for index in batch} == {"a"} for batch in batches)
    for batch in batches:
        seen = set()
        for index in batch:
            assert sampler.dedupe_values[index].isdisjoint(seen)
            seen.update(sampler.dedupe_values[index])


def test_checkpoint_resolver_finds_latest_regular_and_epoch_checkpoints(tmp_path):
    from training.checkpoints import find_latest_checkpoint

    output_dir = tmp_path / "out"
    for checkpoint in (
        output_dir / "checkpoint-100",
        output_dir / "checkpoint-300",
        output_dir / "epoch-checkpoints" / "epoch-0001-step-200",
        output_dir / "epoch-checkpoints" / "epoch-0002-step-400",
    ):
        _write_complete_checkpoint(checkpoint)
    (output_dir / "checkpoint-500").mkdir()

    assert find_latest_checkpoint(output_dir) == output_dir / "epoch-checkpoints" / "epoch-0002-step-400"


def test_explicit_resume_rejects_incomplete_checkpoint(tmp_path):
    from training.checkpoints import CheckpointError, resolve_resume_checkpoint

    output_dir = tmp_path / "out"
    incomplete = output_dir / "checkpoint-500"
    incomplete.mkdir(parents=True)

    with pytest.raises(CheckpointError) as exc:
        resolve_resume_checkpoint(
            output_dir,
            {"resume_from_checkpoint": str(incomplete)},
            {},
            SimpleNamespace(),
        )

    assert "is incomplete" in str(exc.value)
    assert "trainer_state.json" in str(exc.value)


def test_epoch_checkpoint_callback_preserves_epoch_checkpoint(tmp_path):
    from training.checkpoints import EpochCheckpointCallback

    output_dir = tmp_path / "out"
    source = output_dir / "checkpoint-5"
    source.mkdir(parents=True)
    (source / "trainer_state.json").write_text(json.dumps({"global_step": 5}), encoding="utf-8")
    callback = EpochCheckpointCallback(output_dir)
    control = SimpleNamespace(should_save=False)

    callback.on_epoch_end(None, SimpleNamespace(global_step=5, epoch=1.0), control)
    callback.on_save(None, SimpleNamespace(global_step=5, epoch=1.0), control)

    preserved = output_dir / "epoch-checkpoints" / "epoch-0001-step-5"
    assert control.should_save is True
    assert (preserved / "trainer_state.json").exists()


def test_epoch_checkpoint_callback_ignores_unhandled_trainer_events(tmp_path):
    from training.checkpoints import EpochCheckpointCallback

    callback = EpochCheckpointCallback(tmp_path / "out")
    control = SimpleNamespace(should_save=False)

    assert callback.on_train_begin(None, SimpleNamespace(global_step=0), control) is control
    assert control.should_save is False


def test_grid_run_names_include_distinct_hparams(monkeypatch, tmp_path):
    import training.config_grid as config_grid

    monkeypatch.setattr(config_grid, "_timestamp_slug", lambda: "20260525-101112")

    variants = config_grid.expand_config_grid(
        {
            "backend": "sentence-transformers",
            "training_type": "splade",
            "runs_dir": str(tmp_path / "runs"),
            "timestamp_output_dir": True,
            "architectures": ["sdadas/polish-distilroberta"],
            "hparams": [
                {
                    "learning_rate": 2e-6,
                    "num_train_epochs": 1,
                    "loss": "sparse_multiple_negatives_ranking",
                    "document_regularizer_weight": 0.0003,
                },
                {
                    "learning_rate": 2e-6,
                    "num_train_epochs": 1,
                    "loss": "sparse_margin_mse",
                    "document_regularizer_weight": 0.0003,
                    "score_normalization": "per_query_minmax",
                },
            ],
            "sentence_transformers": {},
        },
        backend="sentence-transformers",
        training_type="splade",
    )

    assert len({variant["run_name"] for variant in variants}) == 2
    assert len({variant["output_dir"] for variant in variants}) == 2
    assert "loss-sparse_multiple_negatives_ranking" in variants[0]["run_name"]
    assert "loss-sparse_margin_mse" in variants[1]["run_name"]
    assert "docreg-0_0003" in variants[1]["run_name"]
    assert variants[1]["sentence_transformers"]["score_normalization"] == "per_query_minmax"


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


def test_run_training_applies_config_overrides_before_backend_call(monkeypatch, tmp_path):
    import training.train as train

    config = tmp_path / "train.yaml"
    config.write_text(
        dedent(
            """
            backend: sentence-transformers
            training_type: splade
            runs_dir: runs/test-overrides

            sentence_transformers:
              train_batch_size: 8
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    args = train.build_parser().parse_args(
        [
            "--config",
            str(config),
            "--no-distributed",
            "--set",
            "train_data=dataset-from-cli",
            "--set",
            "hparams.0.learning_rate=0.00002",
            "--set",
            "hparams.0.num_train_epochs=1",
            "--set",
            "sentence_transformers.negatives_per_query=2",
            "--set-str",
            "model_cache_dir=cache/from-cli",
            "--set-str",
            "sentence_transformers.save_strategy=no",
        ]
    )
    requests = []

    def fake_load_backend_module(spec):
        return SimpleNamespace(run_training=lambda request: requests.append(request) or 0)

    monkeypatch.setattr(train, "load_backend_module", fake_load_backend_module)

    assert train.run_training(args) == 0
    assert len(requests) == 1
    request_config = requests[0].config
    assert request_config["train_data"] == "dataset-from-cli"
    assert request_config["learning_rate"] == 0.00002
    assert request_config["num_train_epochs"] == 1
    assert request_config["model_cache_dir"] == "cache/from-cli"
    assert request_config["sentence_transformers"]["train_batch_size"] == 8
    assert request_config["sentence_transformers"]["negatives_per_query"] == 2
    assert request_config["sentence_transformers"]["save_strategy"] == "no"


def test_run_training_dry_run_prints_plan_and_config_without_backend(monkeypatch, tmp_path, capsys):
    import yaml
    import training.train as train

    config = tmp_path / "train.yaml"
    output_dir = tmp_path / "runs"
    config.write_text(
        dedent(
            f"""
            backend: sentence-transformers
            training_type: splade
            output_dir: {output_dir}
            train_data: dataset-from-yaml
            architectures:
              - tiny-model
            hparams:
              - learning_rate: 2e-6
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    args = train.build_parser().parse_args(
        [
            "--config",
            str(config),
            "--dry-run",
            "--print-config",
            "--set-str",
            "train_data=dataset-from-cli",
        ]
    )

    def fail_load_backend_module(spec):
        raise AssertionError("dry-run should not load the backend module")

    monkeypatch.setattr(train, "load_backend_module", fail_load_backend_module)

    assert train.run_training(args) == 0
    stdout = capsys.readouterr().out
    assert "[DRY-RUN] 1 run(s) would execute." in stdout
    assert "train_data=dataset-from-cli" in stdout
    printed_yaml = "\n".join(line for line in stdout.splitlines() if not line.startswith("[DRY-RUN]"))
    printed_config = yaml.safe_load(printed_yaml)
    assert printed_config["train_data"] == "dataset-from-cli"
    assert printed_config["output_dir"] == str(output_dir)
    assert not (output_dir / "resolved_config.yaml").exists()


def test_run_training_writes_resolved_config_artifacts(monkeypatch, tmp_path):
    import yaml
    import training.train as train

    config = tmp_path / "train.yaml"
    output_dir = tmp_path / "out"
    config.write_text(
        dedent(
            f"""
            backend: sentence-transformers
            training_type: splade
            output_dir: {output_dir}
            train_data: dataset-a

            sentence_transformers:
              model_name_or_path: tiny-model
              train_batch_size: 8
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    args = train.build_parser().parse_args(
        [
            "--config",
            str(config),
            "--no-distributed",
            "--set",
            "sentence_transformers.train_batch_size=4",
        ]
    )
    args._raw_argv = [
        "--config",
        str(config),
        "--no-distributed",
        "--set",
        "sentence_transformers.train_batch_size=4",
    ]
    requests = []

    def fake_load_backend_module(spec):
        return SimpleNamespace(run_training=lambda request: requests.append(request) or 0)

    monkeypatch.setattr(train, "load_backend_module", fake_load_backend_module)

    assert train.run_training(args) == 0
    assert len(requests) == 1
    resolved_config = yaml.safe_load((output_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    command_text = (output_dir / "command.txt").read_text(encoding="utf-8")
    assert resolved_config["train_data"] == "dataset-a"
    assert resolved_config["sentence_transformers"]["train_batch_size"] == 4
    assert "python -m training.train" in command_text
    assert "sentence_transformers.train_batch_size=4" in command_text


def test_run_training_can_skip_resolved_config_artifacts(monkeypatch, tmp_path):
    import training.train as train

    config = tmp_path / "train.yaml"
    output_dir = tmp_path / "out"
    config.write_text(
        dedent(
            f"""
            backend: sentence-transformers
            training_type: splade
            output_dir: {output_dir}
            train_data: dataset-a
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    args = train.build_parser().parse_args(["--config", str(config), "--no-save-resolved-config"])

    def fake_load_backend_module(spec):
        return SimpleNamespace(run_training=lambda request: 0)

    monkeypatch.setattr(train, "load_backend_module", fake_load_backend_module)

    assert train.run_training(args) == 0
    assert not (output_dir / "resolved_config.yaml").exists()
    assert not (output_dir / "command.txt").exists()


def test_run_training_rejects_explicit_checkpoint_for_grid(monkeypatch, tmp_path):
    import training.train as train

    checkpoint = tmp_path / "runs" / "model" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
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
    args = train.build_parser().parse_args(["--config", str(config), "--resume-from-checkpoint", str(checkpoint)])

    monkeypatch.setattr(train, "load_backend_module", lambda spec: SimpleNamespace(run_training=lambda request: 0))

    with pytest.raises(train.ConfigError) as exc:
        train.run_training(args)

    assert "--resume-from-checkpoint can only be used with a single expanded run" in str(exc.value)


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


def test_flagembedding_backend_maps_model_cache_dir_to_cache_dir(monkeypatch, tmp_path):
    from training.backends import flagembedding_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_jsonl(data_dir / "dataset.jsonl", [{"query": "q", "pos": ["p"], "neg": ["n"]}])
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
            "model_name_or_path": "tiny-model",
            "output_dir": str(tmp_path / "runs" / "tiny-run"),
            "run_name": "tiny-run",
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
            "train_data": str(data_dir),
            "model_cache_dir": str(tmp_path / "model-cache"),
            "knowledge_distillation": False,
        },
        config_path=str(tmp_path / "config.yaml"),
        cli_args=SimpleNamespace(run_mteb=False, run_pirb=False, remove_checkpoints=False),
    )

    assert flagembedding_backend.run_training(request) == 0
    cmd = calls["cmd"]
    assert cmd is not None
    assert "--model-cache-dir" not in cmd
    assert cmd[cmd.index("--cache-dir") + 1] == str(tmp_path / "model-cache")


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


def test_flagembedding_backend_materializes_mixed_train_data(monkeypatch, tmp_path):
    from training.backends import flagembedding_backend
    from training.backends.registry import TrainingRequest

    data_a = tmp_path / "data-a"
    data_b = tmp_path / "data-b"
    data_a.mkdir()
    data_b.mkdir()
    _write_jsonl(data_a / "dataset.jsonl", [{"query": "qa", "pos": ["pa"], "neg": ["na"]}])
    _write_jsonl(data_b / "dataset.jsonl", [{"query": "qb", "pos": ["pb"], "neg": ["nb"]}])
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
            "model_name_or_path": "tiny-model",
            "output_dir": str(tmp_path / "runs" / "tiny-run"),
            "run_name": "tiny-run",
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
            "train_data": [str(data_a), str(data_b)],
            "mixed_dataset_cache_dir": str(tmp_path / "mixed-cache"),
            "knowledge_distillation": False,
        },
        config_path=str(tmp_path / "config.yaml"),
        cli_args=SimpleNamespace(run_mteb=False, run_pirb=False, remove_checkpoints=False),
    )

    assert flagembedding_backend.run_training(request) == 0
    cmd = calls["cmd"]
    assert cmd is not None
    assert "--dataset-mix-strategy" not in cmd
    assert "--mixed-dataset-cache-dir" not in cmd
    train_data_arg = Path(cmd[cmd.index("--train-data") + 1])
    assert train_data_arg.parent == tmp_path / "mixed-cache"
    assert _filtered_queries(train_data_arg / "dataset.jsonl") == ["qa", "qb"]


def test_flagembedding_backend_rejects_proportional_batch_strategy(monkeypatch, tmp_path):
    from training.backends import flagembedding_backend
    from training.backends.registry import TrainingRequest

    data_a = tmp_path / "data-a"
    data_b = tmp_path / "data-b"
    data_a.mkdir()
    data_b.mkdir()
    _write_jsonl(data_a / "dataset.jsonl", [{"query": "qa", "pos": ["pa"], "neg": ["na"]}])
    _write_jsonl(data_b / "dataset.jsonl", [{"query": "qb", "pos": ["pb"], "neg": ["nb"]}])

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            raise AssertionError("wandb.init should not run")

    monkeypatch.setattr(flagembedding_backend, "_require_flagembedding", lambda: None)
    monkeypatch.setattr(flagembedding_backend, "_load_wandb", lambda: FakeWandb)

    request = TrainingRequest(
        backend="flagembedding",
        training_type="embedder",
        config={
            "model_name_or_path": "tiny-model",
            "output_dir": str(tmp_path / "runs" / "tiny-run"),
            "run_name": "tiny-run",
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
            "train_data": [str(data_a), str(data_b)],
            "dataset_mix_strategy": "proportional_batch_best_effort",
            "knowledge_distillation": False,
        },
        config_path=str(tmp_path / "config.yaml"),
        cli_args=SimpleNamespace(run_mteb=False, run_pirb=False, remove_checkpoints=False),
    )

    with pytest.raises(NotImplementedError, match="proportional_batch_best_effort"):
        flagembedding_backend.run_training(request)


def test_flagembedding_backend_adds_resume_checkpoint_arg(monkeypatch, tmp_path):
    from training.backends import flagembedding_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs" / "tiny-run"
    checkpoint = output_dir / "checkpoint-10"
    _write_complete_checkpoint(checkpoint)
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
            "model_name_or_path": "tiny-model",
            "output_dir": str(output_dir),
            "run_name": "tiny-run",
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
            "train_data": str(data_dir),
            "knowledge_distillation": False,
        },
        config_path=str(tmp_path / "config.yaml"),
        cli_args=SimpleNamespace(
            run_mteb=False,
            run_pirb=False,
            remove_checkpoints=False,
            resume=True,
            resume_from_checkpoint=None,
        ),
    )

    assert flagembedding_backend.run_training(request) == 0
    cmd = calls["cmd"]
    assert cmd is not None
    assert "--resume-from-checkpoint" in cmd
    assert cmd[cmd.index("--resume-from-checkpoint") + 1] == str(checkpoint)
    assert "--overwrite_output_dir" not in cmd
    assert cmd[cmd.index("--output_dir") + 1] == str(output_dir)


def test_flagembedding_backend_uses_distributed_gpu_selection(monkeypatch, tmp_path):
    from training.backends import flagembedding_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n"]}) + "\n",
        encoding="utf-8",
    )
    calls = {"cmd": None, "env": None}

    class FakeRun:
        def finish(self):
            pass

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            return FakeRun()

    def fake_subprocess_run(cmd, check, env):
        calls["cmd"] = cmd
        calls["env"] = env

    monkeypatch.setattr(flagembedding_backend, "_require_flagembedding", lambda: None)
    monkeypatch.setattr(flagembedding_backend, "_load_wandb", lambda: FakeWandb)
    monkeypatch.setattr(flagembedding_backend.subprocess, "run", fake_subprocess_run)

    request = TrainingRequest(
        backend="flagembedding",
        training_type="embedder",
        config={
            "runs_dir": str(tmp_path / "runs"),
            "distributed": {"gpus": [2, 3]},
            "architectures": ["tiny-model"],
            "hparams": [
                {
                    "learning_rate": 1e-5,
                    "num_train_epochs": 1,
                    "train_data": str(data_dir),
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
    assert cmd[cmd.index("--nproc_per_node") + 1] == "2"
    assert calls["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"


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


def test_sentence_transformers_load_training_rows_combines_multiple_sources(tmp_path):
    from training.backends.sentence_transformers_backend import _load_training_rows

    data_a = tmp_path / "data-a"
    data_b = tmp_path / "data-b"
    data_a.mkdir()
    data_b.mkdir()
    _write_jsonl(
        data_a / "dataset.jsonl",
        [{"query": "qa", "pos": ["pa"], "neg": ["na1", "na2"]}],
    )
    _write_jsonl(
        data_b / "dataset.jsonl",
        [{"query": "qb", "pos": ["pb"], "neg": ["nb1"]}],
    )

    loaded = _load_training_rows(
        {"train_data": [str(data_a), str(data_b)]},
        {},
        config_path=str(tmp_path / "config.yaml"),
    )

    assert [row["anchor"] for row in loaded.rows] == ["qa", "qb"]
    assert all("negative_2" not in row for row in loaded.rows)
    assert loaded.dataset_ids == ["0-data-a", "1-data-b"]
    assert loaded.dedupe_values[0] == {"qa", "pa", "na1"}
    assert "_dataset_id" not in loaded.rows[0]


def test_huggingface_dataset_file_url_is_downloaded_once(monkeypatch, tmp_path):
    from training import dataset_sources

    url = "https://huggingface.co/datasets/mining-negatives/fiqa_pl/blob/main/train_pl.jsonl"
    calls = []

    def fake_download(download_url, output_path):
        calls.append(download_url)
        _write_jsonl(output_path, [{"query": "q", "pos": ["p"], "neg": ["n"]}])

    monkeypatch.setattr(dataset_sources, "_download_url_to_path", fake_download)

    first = dataset_sources.materialize_huggingface_dataset_file(url, cache_dir=tmp_path / "hf-cache")
    second = dataset_sources.materialize_huggingface_dataset_file(url, cache_dir=tmp_path / "hf-cache")

    assert calls == ["https://huggingface.co/datasets/mining-negatives/fiqa_pl/resolve/main/train_pl.jsonl"]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.output_path == second.output_path
    assert first.output_path.name == "dataset.jsonl"
    assert _read_jsonl(first.output_path) == [{"query": "q", "pos": ["p"], "neg": ["n"]}]
    assert json.loads(first.metadata_path.read_text(encoding="utf-8"))["original_url"] == url


def test_sentence_transformers_load_training_rows_accepts_huggingface_file_url(monkeypatch, tmp_path):
    from training import dataset_sources
    from training.backends.sentence_transformers_backend import _load_training_rows

    url = "https://huggingface.co/datasets/mining-negatives/fiqa_pl/blob/main/train_pl.jsonl"

    def fake_download(download_url, output_path):
        _write_jsonl(output_path, [{"query": "q", "pos": ["p"], "neg": ["n1", "n2"]}])

    monkeypatch.setattr(dataset_sources, "_download_url_to_path", fake_download)

    loaded = _load_training_rows(
        {"train_data": url, "hf_dataset_cache_dir": str(tmp_path / "hf-cache")},
        {"negatives_per_query": 1},
        config_path=str(tmp_path / "config.yaml"),
    )

    assert loaded.rows == [{"anchor": "q", "positive": "p", "negative_1": "n1"}]
    assert loaded.sources[0].original == url
    assert loaded.sources[0].path.parent.parent == tmp_path / "hf-cache"


def test_sentence_transformers_dataset_loader_uses_score_labels_for_margin_mse(tmp_path):
    from training.backends.sentence_transformers_backend import load_flagembedding_jsonl_dataset

    data_file = tmp_path / "dataset.jsonl"
    data_file.write_text(
        json.dumps(
            {
                "query": "question",
                "pos": ["positive one"],
                "neg": ["negative one", "negative two"],
                "pos_scores": [27.0],
                "neg_scores": [17.75, 16.5],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_flagembedding_jsonl_dataset(
        data_file,
        negatives_per_query=2,
        query_prefix="",
        passage_prefix="",
        use_score_labels=True,
        score_normalization="none",
    )

    assert rows == [
        {
            "anchor": "question",
            "positive": "positive one",
            "negative_1": "negative one",
            "negative_2": "negative two",
            "label": [9.25, 10.5],
        }
    ]


def test_sentence_transformers_dataset_loader_can_normalize_score_labels(tmp_path):
    from training.backends.sentence_transformers_backend import load_flagembedding_jsonl_dataset

    data_file = tmp_path / "dataset.jsonl"
    data_file.write_text(
        json.dumps(
            {
                "query": "question",
                "pos": ["positive one"],
                "neg": ["negative one", "negative two"],
                "pos_scores": [100.0],
                "neg_scores": [80.0, 60.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_flagembedding_jsonl_dataset(
        data_file,
        negatives_per_query=2,
        query_prefix="",
        passage_prefix="",
        use_score_labels=True,
        score_normalization="per_query_minmax",
    )

    assert rows[0]["label"] == [0.5, 1.0]


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
    calls = {"trained": False, "saved": None, "rows": None, "model_kwargs": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            calls["rows"] = rows
            return rows

    class FakeModel:
        def __init__(self, model_name_or_path, **model_kwargs):
            self.model_name_or_path = model_name_or_path
            self.model_kwargs = model_kwargs
            calls["model_kwargs"] = model_kwargs
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
            "model_cache_dir": str(tmp_path / "model-cache"),
            "output_dir": str(output_dir),
            "sentence_transformers": {
                "model_name_or_path": "tiny-model",
                "max_steps": 1,
                "train_batch_size": 1,
                "negatives_per_query": 1,
                "max_seq_length": 64,
                "loss": "dense_only_loss",
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
    assert calls["model_kwargs"]["cache_folder"] == str(tmp_path / "model-cache")


def test_sentence_transformers_embedder_benchmarks_selected_checkpoints(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    (output_dir / "epoch-checkpoints" / "epoch-0001-step-123").mkdir(parents=True)
    (output_dir / "checkpoint-20000").mkdir(parents=True)
    calls = {"trained": False, "saved": None, "benchmarks": []}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
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
            pass

        def train(self):
            calls["trained"] = True

    def fake_run_benchmarks_for_model(model_dir, settings, metric_prefix, step, label):
        calls["benchmarks"].append(
            {
                "model_dir": model_dir,
                "metric_prefix": metric_prefix,
                "step": step,
                "label": label,
                "settings": settings,
            }
        )
        return {f"{metric_prefix}pirb_average_ndcg@10": 0.5}

    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_sentence_transformers",
        lambda: (FakeDataset, FakeModel, FakeTrainer, FakeArgs, FakeLoss, object),
    )
    monkeypatch.setattr(sentence_transformers_backend, "run_benchmarks_for_model", fake_run_benchmarks_for_model)

    request = TrainingRequest(
        backend="sentence-transformers",
        training_type="embedder",
        config={
            "train_data": str(data_dir),
            "output_dir": str(output_dir),
            "benchmark": {
                "run_pirb": True,
                "scope": "small",
                "output_dir": str(tmp_path / "bench"),
                "checkpoints": ["final", {"epoch": 1}, {"step": 20000}],
            },
            "sentence_transformers": {
                "model_name_or_path": "tiny-model",
                "max_steps": 1,
                "train_batch_size": 1,
                "negatives_per_query": 1,
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(run_mteb=False, run_pirb=False),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["trained"] is True
    assert calls["saved"] == str(output_dir / "final")
    assert [(call["label"], call["metric_prefix"], call["step"]) for call in calls["benchmarks"]] == [
        ("final", "final/", 0),
        ("epoch-0001", "epoch-0001/", 123),
        ("step-20000", "step-20000/", 20000),
    ]
    assert [Path(call["model_dir"]) for call in calls["benchmarks"]] == [
        (output_dir / "final").resolve(),
        (output_dir / "epoch-checkpoints" / "epoch-0001-step-123").resolve(),
        (output_dir / "checkpoint-20000").resolve(),
    ]


def test_sentence_transformers_embedder_resumes_from_latest_checkpoint(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    _write_complete_checkpoint(output_dir / "checkpoint-10")
    _write_complete_checkpoint(output_dir / "checkpoint-20")
    calls = {"resume": None, "saved": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
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
            pass

        def train(self, resume_from_checkpoint=None):
            calls["resume"] = resume_from_checkpoint

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
                "resume": True,
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["resume"] == str(output_dir / "checkpoint-20")
    assert calls["saved"] == str(output_dir / "final")


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


def test_sentence_transformers_matryoshka_resumes_from_checkpoint(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    checkpoint = output_dir / "checkpoint-30"
    _write_complete_checkpoint(checkpoint)
    calls = {"resume": None}

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
            pass

    class FakeArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRankingLoss:
        def __init__(self, model):
            pass

    class FakeMatryoshkaLoss:
        def __init__(self, model, loss, matryoshka_dims, **kwargs):
            pass

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            pass

        def train(self, resume_from_checkpoint=None):
            calls["resume"] = resume_from_checkpoint

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
            "resume_from_checkpoint": str(checkpoint),
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
    assert calls["resume"] == str(checkpoint)


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

    class FakeSparseMarginMSELoss:
        def __init__(self, model):
            self.model = model

    class FakeSpladeLoss:
        def __init__(self, model, loss, document_regularizer_weight, query_regularizer_weight):
            calls["regularizers"] = (document_regularizer_weight, query_regularizer_weight)

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            calls["args"] = args
            self.loss = loss

        def train(self):
            calls["trained"] = True

    class FakeWandb:
        run = object()

        @classmethod
        def finish(cls):
            calls["wandb_finished"] = True
            cls.run = None

    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_sparse_sentence_transformers",
        lambda: (
            FakeDataset,
            FakeSparseEncoder,
            FakeTrainer,
            FakeArgs,
            FakeSparseMarginMSELoss,
            FakeSparseRankingLoss,
            FakeSpladeLoss,
            FakeMLMTransformer,
            FakeSpladePooling,
        ),
    )
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    request = TrainingRequest(
        backend="sentence-transformers",
        training_type="splade",
        config={
            "train_data": str(data_dir),
            "output_dir": str(output_dir),
            "sentence_transformers": {
                "model_name_or_path": "tiny-mlm",
                "tokenizer_name_or_path": "tiny-tokenizer",
                "processor_kwargs": {"use_fast": False},
                "run_name": "tiny-grid-run",
                "max_steps": 1,
                "train_batch_size": 1,
                "batch_sampler": "no_duplicates",
                "gradient_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "report_to": ["wandb"],
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
    assert calls["modules"][0].kwargs["tokenizer_name_or_path"] == "tiny-tokenizer"
    assert calls["modules"][0].kwargs["processor_kwargs"] == {"use_fast": False}
    assert calls["modules"][1].pooling_strategy == "max"
    assert calls["args"].kwargs["run_name"] == "tiny-grid-run"
    assert calls["args"].kwargs["report_to"] == ["wandb"]
    assert calls["args"].kwargs["per_device_train_batch_size"] == 1
    assert calls["args"].kwargs["batch_sampler"] == "no_duplicates"
    assert calls["args"].kwargs["gradient_accumulation_steps"] == 4
    assert calls["args"].kwargs["gradient_checkpointing"] is True
    assert calls["wandb_finished"] is True


def test_sentence_transformers_splade_uses_proportional_batch_sampler(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_a = tmp_path / "data-a"
    data_b = tmp_path / "data-b"
    data_a.mkdir()
    data_b.mkdir()
    _write_jsonl(data_a / "dataset.jsonl", [{"query": "qa", "pos": ["pa"], "neg": ["na"]}])
    _write_jsonl(data_b / "dataset.jsonl", [{"query": "qb", "pos": ["pb"], "neg": ["nb"]}])
    output_dir = tmp_path / "out"
    calls = {"trained": False, "saved": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeMLMTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            self.model_name_or_path = model_name_or_path

    class FakeSpladePooling:
        def __init__(self, pooling_strategy):
            self.pooling_strategy = pooling_strategy

    class FakeSparseEncoder:
        def __init__(self, modules):
            self.max_seq_length = None

        def save_pretrained(self, output_path):
            calls["saved"] = output_path

    class FakeArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSparseRankingLoss:
        def __init__(self, model, scale, gather_across_devices):
            self.model = model

    class FakeSparseMarginMSELoss:
        def __init__(self, model):
            self.model = model

    class FakeSpladeLoss:
        def __init__(self, model, loss, document_regularizer_weight, query_regularizer_weight):
            self.model = model

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            calls["args"] = args
            calls["train_dataset"] = train_dataset

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
            FakeSparseMarginMSELoss,
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
            "train_data": [str(data_a), str(data_b)],
            "output_dir": str(output_dir),
            "dataset_mix_strategy": "proportional_batch_best_effort",
            "sentence_transformers": {
                "model_name_or_path": "tiny-mlm",
                "max_steps": 1,
                "train_batch_size": 2,
                "batch_sampler": "no_duplicates",
                "save_strategy": "no",
            },
        },
        config_path=str(tmp_path / "config.yaml"),
        cli_args=SimpleNamespace(),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    sampler_factory = calls["args"].kwargs["batch_sampler"]
    sampler = sampler_factory(calls["train_dataset"], batch_size=2, drop_last=False, seed=0)
    batches = list(sampler)
    assert len(batches) == 1
    assert set(batches[0]) == {0, 1}
    assert calls["trained"] is True
    assert calls["saved"] == str(output_dir / "final")


def test_sentence_transformers_splade_resumes_from_epoch_checkpoint(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    checkpoint = output_dir / "epoch-checkpoints" / "epoch-0001-step-40"
    _write_complete_checkpoint(checkpoint)
    calls = {"resume": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeMLMTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            pass

    class FakeSpladePooling:
        def __init__(self, pooling_strategy):
            pass

    class FakeSparseEncoder:
        def __init__(self, modules):
            pass

        def save_pretrained(self, output_path):
            pass

    class FakeArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSparseRankingLoss:
        def __init__(self, model, scale, gather_across_devices):
            pass

    class FakeSparseMarginMSELoss:
        def __init__(self, model):
            pass

    class FakeSpladeLoss:
        def __init__(self, model, loss, document_regularizer_weight, query_regularizer_weight):
            pass

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            pass

        def train(self, resume_from_checkpoint=None):
            calls["resume"] = resume_from_checkpoint

    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_sparse_sentence_transformers",
        lambda: (
            FakeDataset,
            FakeSparseEncoder,
            FakeTrainer,
            FakeArgs,
            FakeSparseMarginMSELoss,
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
                "resume": True,
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["resume"] == str(checkpoint)


def test_sentence_transformers_splade_can_use_margin_mse_scores(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps(
            {
                "query": "q",
                "pos": ["p"],
                "neg": ["n1", "n2"],
                "pos_scores": [10.0],
                "neg_scores": [7.0, 4.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    calls = {}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            calls["rows"] = rows
            return rows

    class FakeMLMTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            pass

    class FakeSpladePooling:
        def __init__(self, pooling_strategy):
            pass

    class FakeSparseEncoder:
        def __init__(self, modules):
            pass

        def save_pretrained(self, output_path):
            pass

    class FakeArgs:
        def __init__(self, **kwargs):
            pass

    class FakeSparseMarginMSELoss:
        def __init__(self, model):
            calls["base_loss"] = "margin"

    class FakeSparseRankingLoss:
        def __init__(self, model, scale, gather_across_devices):
            calls["base_loss"] = "mnrl"

    class FakeSpladeLoss:
        def __init__(self, model, loss, document_regularizer_weight, query_regularizer_weight):
            pass

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            pass

        def train(self):
            pass

    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_sparse_sentence_transformers",
        lambda: (
            FakeDataset,
            FakeSparseEncoder,
            FakeTrainer,
            FakeArgs,
            FakeSparseMarginMSELoss,
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
                "loss": "sparse_margin_mse",
                "score_normalization": "none",
                "negatives_per_query": 2,
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["base_loss"] == "margin"
    assert calls["rows"][0]["label"] == [3.0, 6.0]


def test_sentence_transformers_splade_adds_activation_stats_callback(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"query": "q1", "pos": ["p1"], "neg": ["n1"]}),
                json.dumps({"query": "q2", "pos": ["p2"], "neg": ["n2"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    callbacks = []

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeMLMTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            pass

    class FakeSpladePooling:
        def __init__(self, pooling_strategy):
            pass

    class FakeSparseEncoder:
        def __init__(self, modules):
            pass

        def save_pretrained(self, output_path):
            pass

    class FakeArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSparseRankingLoss:
        def __init__(self, model, scale, gather_across_devices):
            pass

    class FakeSparseMarginMSELoss:
        def __init__(self, model):
            pass

    class FakeSpladeLoss:
        def __init__(self, model, loss, document_regularizer_weight, query_regularizer_weight):
            pass

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            pass

        def add_callback(self, callback):
            callbacks.append(callback)

        def train(self):
            pass

    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_sparse_sentence_transformers",
        lambda: (
            FakeDataset,
            FakeSparseEncoder,
            FakeTrainer,
            FakeArgs,
            FakeSparseMarginMSELoss,
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
                "splade_activation_stats": {
                    "enabled": True,
                    "sample_size": 2,
                    "batch_size": 1,
                    "interval_steps": 10,
                    "include_negatives": True,
                },
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert callback.query_texts == ["q1", "q2"]
    assert callback.document_texts == ["p1", "n2"]
    assert callback.batch_size == 1
    assert callback.interval_steps == 10
    assert callback.quantization_factor == 100
    assert callback.on_epoch_begin(None, SimpleNamespace(global_step=0), "control") == "control"


def test_sentence_transformers_splade_runs_post_training_benchmark(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    calls = {"trained": False, "saved": None, "benchmark": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeMLMTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            pass

    class FakeSpladePooling:
        def __init__(self, pooling_strategy):
            pass

    class FakeSparseEncoder:
        def __init__(self, modules):
            pass

        def save_pretrained(self, output_path):
            calls["saved"] = output_path

    class FakeArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSparseRankingLoss:
        def __init__(self, model, scale, gather_across_devices):
            pass

    class FakeSparseMarginMSELoss:
        def __init__(self, model):
            pass

    class FakeSpladeLoss:
        def __init__(self, model, loss, document_regularizer_weight, query_regularizer_weight):
            pass

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss):
            pass

        def train(self):
            calls["trained"] = True

    def fake_run_benchmarks_for_model(model_dir, settings, metric_prefix, step, label):
        calls["benchmark"] = {
            "model_dir": model_dir,
            "settings": settings,
            "metric_prefix": metric_prefix,
            "step": step,
            "label": label,
        }
        return {"final/pirb_average_ndcg@10": 0.5}

    monkeypatch.setattr(
        sentence_transformers_backend,
        "_load_sparse_sentence_transformers",
        lambda: (
            FakeDataset,
            FakeSparseEncoder,
            FakeTrainer,
            FakeArgs,
            FakeSparseMarginMSELoss,
            FakeSparseRankingLoss,
            FakeSpladeLoss,
            FakeMLMTransformer,
            FakeSpladePooling,
        ),
    )
    monkeypatch.setattr(sentence_transformers_backend, "run_benchmarks_for_model", fake_run_benchmarks_for_model)

    request = TrainingRequest(
        backend="sentence-transformers",
        training_type="splade",
        config={
            "train_data": str(data_dir),
            "output_dir": str(output_dir),
            "benchmark": {
                "run_pirb": True,
                "scope": "small",
                "max_seq_length": 384,
                "output_dir": str(tmp_path / "bench"),
                "query_instruction": "Pytanie: ",
            },
            "sentence_transformers": {
                "model_name_or_path": "tiny-mlm",
                "max_steps": 1,
                "train_batch_size": 1,
                "negatives_per_query": 1,
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(run_mteb=False, run_pirb=False),
    )

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["trained"] is True
    assert calls["saved"] == str(output_dir / "final")
    assert calls["benchmark"]["model_dir"] == str((output_dir / "final").resolve())
    assert calls["benchmark"]["metric_prefix"] == "final/"
    assert calls["benchmark"]["step"] == 0
    assert calls["benchmark"]["label"] == "final"
    settings = calls["benchmark"]["settings"]
    assert settings.run_pirb is True
    assert settings.pirb_scope == "small"
    assert settings.pirb_max_seq_length == 384
    assert settings.query_instruction_for_retrieval == "Pytanie: "
    assert settings.output_dir == tmp_path / "bench"


def test_sentence_transformers_post_training_benchmarks_selected_checkpoints(monkeypatch, tmp_path):
    from training.backends import sentence_transformers_backend
    from training.backends.registry import TrainingRequest

    output_dir = tmp_path / "out"
    (output_dir / "final").mkdir(parents=True)
    (output_dir / "epoch-checkpoints" / "epoch-0001-step-123").mkdir(parents=True)
    (output_dir / "epoch-checkpoints" / "epoch-0002-step-456").mkdir(parents=True)
    (output_dir / "checkpoint-20000").mkdir(parents=True)
    calls = []
    barriers = []

    def fake_run_benchmarks_for_model(model_dir, settings, metric_prefix, step, label):
        calls.append(
            {
                "model_dir": model_dir,
                "settings": settings,
                "metric_prefix": metric_prefix,
                "step": step,
                "label": label,
            }
        )
        return {f"{metric_prefix}pirb_average_ndcg@10": 0.5}

    monkeypatch.setattr(sentence_transformers_backend, "is_main_process", lambda: True)
    monkeypatch.setattr(sentence_transformers_backend, "barrier_if_distributed", lambda: barriers.append(True))
    monkeypatch.setattr(sentence_transformers_backend, "run_benchmarks_for_model", fake_run_benchmarks_for_model)

    request = TrainingRequest(
        backend="sentence-transformers",
        training_type="splade",
        config={
            "benchmark": {
                "run_pirb": True,
                "scope": "small",
                "output_dir": str(tmp_path / "bench"),
                "checkpoints": ["final", {"epoch": 1}, {"epoch": 2}, {"step": 20000}],
            }
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(run_mteb=False, run_pirb=False),
    )

    sentence_transformers_backend._run_sentence_transformers_post_training_benchmarks(output_dir, request.config, {}, request)

    assert [(call["label"], call["metric_prefix"], call["step"]) for call in calls] == [
        ("final", "final/", 0),
        ("epoch-0001", "epoch-0001/", 123),
        ("epoch-0002", "epoch-0002/", 456),
        ("step-20000", "step-20000/", 20000),
    ]
    assert [Path(call["model_dir"]) for call in calls] == [
        (output_dir / "final").resolve(),
        (output_dir / "epoch-checkpoints" / "epoch-0001-step-123").resolve(),
        (output_dir / "epoch-checkpoints" / "epoch-0002-step-456").resolve(),
        (output_dir / "checkpoint-20000").resolve(),
    ]
    assert barriers == [True]


def test_sentence_transformers_splade_maps_model_cache_dir_to_hf_kwargs():
    from training.backends import sentence_transformers_backend

    calls = {}

    class FakeMLMTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            calls["model_name_or_path"] = model_name_or_path
            calls["kwargs"] = kwargs

    class FakeSpladePooling:
        def __init__(self, **kwargs):
            calls["pooling_kwargs"] = kwargs

    class FakeSparseEncoder:
        def __init__(self, modules):
            self.modules = modules

    model = sentence_transformers_backend._build_splade_model(
        FakeSparseEncoder,
        FakeMLMTransformer,
        FakeSpladePooling,
        "tiny-mlm",
        {
            "model_cache_dir": "cache/models",
            "tokenizer_name_or_path": "tiny-tokenizer",
        },
    )

    assert isinstance(model, FakeSparseEncoder)
    assert calls["model_name_or_path"] == "tiny-mlm"
    assert calls["kwargs"]["model_kwargs"]["cache_dir"] == "cache/models"
    assert calls["kwargs"]["processor_kwargs"]["cache_dir"] == "cache/models"
    assert calls["kwargs"]["config_kwargs"]["cache_dir"] == "cache/models"


def test_sentence_transformers_splade_processor_fallback_uses_tokenizer(monkeypatch):
    from training.backends import sentence_transformers_backend

    tokenizer_calls = []

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(path, *args, **kwargs):
            raise ValueError("Unrecognized processing class in tokenizer-only-checkpoint")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, *args, **kwargs):
            tokenizer_calls.append((path, kwargs))
            return "loaded-tokenizer"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=FakeAutoProcessor, AutoTokenizer=FakeAutoTokenizer),
    )

    class FakeMLMTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            from transformers import AutoProcessor

            self.model_name_or_path = model_name_or_path
            self.processor = AutoProcessor.from_pretrained(
                kwargs["tokenizer_name_or_path"],
                **kwargs["processor_kwargs"],
            )

    transformer = sentence_transformers_backend._build_mlm_transformer(
        FakeMLMTransformer,
        "model-checkpoint",
        {
            "tokenizer_name_or_path": "tokenizer-only-checkpoint",
            "processor_kwargs": {"use_fast": False},
        },
    )

    assert transformer.model_name_or_path == "model-checkpoint"
    assert transformer.processor == "loaded-tokenizer"
    assert tokenizer_calls == [("tokenizer-only-checkpoint", {"use_fast": False})]


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
    calls = {"trained": False, "saved": None, "rows": None, "collator": False, "model_kwargs": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            calls["rows"] = rows
            return rows

    class FakeColBERT:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls["model_kwargs"] = kwargs

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
            "model_cache_dir": str(tmp_path / "model-cache"),
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
    assert calls["model_kwargs"]["cache_folder"] == str(tmp_path / "model-cache")


def test_pylate_colbert_resumes_from_checkpoint(monkeypatch, tmp_path):
    from training.backends import pylate_backend
    from training.backends.registry import TrainingRequest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.jsonl").write_text(
        json.dumps({"query": "q", "pos": ["p"], "neg": ["n1"]}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    checkpoint = output_dir / "checkpoint-50"
    _write_complete_checkpoint(checkpoint)
    calls = {"resume": None}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeColBERT:
        def __init__(self, **kwargs):
            pass

        def tokenize(self, texts, **kwargs):
            return {"input_ids": texts}

        def save_pretrained(self, output_path):
            pass

    class FakeLoss:
        def __init__(self, model, gather_across_devices, temperature):
            pass

    class FakeCollator:
        def __init__(self, tokenize_fn):
            pass

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, loss, data_collator):
            pass

        def train(self, resume_from_checkpoint=None):
            calls["resume"] = resume_from_checkpoint

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
                "resume_from_checkpoint": str(checkpoint),
            },
        },
        config_path="config.yaml",
        cli_args=SimpleNamespace(),
    )

    assert pylate_backend.run_training(request) == 0
    assert calls["resume"] == str(checkpoint)


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


def test_run_experiments_train_mode_forwards_resume(monkeypatch):
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
            "--resume-from-checkpoint",
            "runs/model/checkpoint-10",
        ]
    )

    assert result == 0
    cmd, _ = calls[0]
    assert "--resume-from-checkpoint" in cmd
    assert cmd[cmd.index("--resume-from-checkpoint") + 1] == "runs/model/checkpoint-10"


def test_benchmark_target_accepts_wsl_path_on_windows(monkeypatch):
    import run_experiments

    monkeypatch.setattr(run_experiments.os, "name", "nt")

    target = run_experiments.parse_benchmark_target(
        "/mnt/c/work/embedder-training-v2/runs/model/final::",
        default_qi="",
    )

    assert target.model_or_path == r"C:\work\embedder-training-v2\runs\model\final"


def test_run_pirb_marks_sparse_encoder_as_splade(monkeypatch, tmp_path):
    import convert_utils

    model_dir = tmp_path / "splade-model"
    model_dir.mkdir()
    (model_dir / "config_sentence_transformers.json").write_text(
        json.dumps({"model_type": "SparseEncoder"}),
        encoding="utf-8",
    )

    calls = {}

    def fake_run(cmd, check, cwd):
        calls["cmd"] = cmd
        calls["check"] = check
        calls["cwd"] = cwd
        results_path = Path(cmd[cmd.index("--results_json") + 1])
        results_path.write_text(json.dumps({"results": [{"average_ndcg@10": 1.0}]}), encoding="utf-8")

    monkeypatch.setattr(convert_utils.subprocess, "run", fake_run)

    metrics = convert_utils.run_pirb(str(model_dir), query_instruction_for_retrieval="Pytanie: ", scope="tiny")

    models_config = Path(calls["cmd"][calls["cmd"].index("--models_config") + 1])
    cfg = json.loads(models_config.read_text(encoding="utf-8"))
    assert calls["check"] is True
    assert cfg[0]["type"] == "splade"
    assert cfg[0]["q_prefix"] == "Pytanie: "
    assert metrics == {"pirb_average_ndcg@10": 1.0}


def test_run_pirb_keeps_dense_config_without_sparse_marker(monkeypatch, tmp_path):
    import convert_utils

    model_dir = tmp_path / "dense-model"
    model_dir.mkdir()
    (model_dir / "modules.json").write_text(json.dumps([]), encoding="utf-8")

    calls = {}

    def fake_run(cmd, check, cwd):
        calls["cmd"] = cmd
        results_path = Path(cmd[cmd.index("--results_json") + 1])
        results_path.write_text(json.dumps({"results": [{"average_ndcg@10": 2.0}]}), encoding="utf-8")

    monkeypatch.setattr(convert_utils.subprocess, "run", fake_run)

    convert_utils.run_pirb(str(model_dir), query_instruction_for_retrieval="", scope="tiny")

    models_config = Path(calls["cmd"][calls["cmd"].index("--models_config") + 1])
    cfg = json.loads(models_config.read_text(encoding="utf-8"))
    assert "type" not in cfg[0]
