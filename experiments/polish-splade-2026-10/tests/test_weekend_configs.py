from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.backends.sentence_transformers_backend import _backend_config  # noqa: E402
from training.benchmarks import resolve_benchmark_settings  # noqa: E402
from training.config_grid import expand_config_grid  # noqa: E402
from training.multi_dataset import normalize_train_data_specs  # noqa: E402
from training.train import load_config  # noqa: E402


GRID_CONFIG = REPO_ROOT / "configs" / "weekend" / "ft_rehearsal_grid.yaml"
OVERNIGHT_CONFIG = REPO_ROOT / "configs" / "weekend" / "ft_rehearsal_overnight.yaml"


def _expand(config_path):
    config = load_config(str(config_path))
    return config, expand_config_grid(config, backend="sentence-transformers", training_type="splade")


def _settings(variant):
    return resolve_benchmark_settings(variant, _backend_config(variant), SimpleNamespace())


@pytest.mark.parametrize("config_path", [GRID_CONFIG, OVERNIGHT_CONFIG])
def test_weekend_configs_target_the_splade_architecture(config_path):
    config, variants = _expand(config_path)
    assert config["backend"] == "sentence-transformers"
    assert config["training_type"] == "splade"
    assert {variant["grid_architecture"] for variant in variants} == {"sdadas/polish-splade"}


def test_grid_runs_the_learning_rates_in_the_configured_order():
    _, variants = _expand(GRID_CONFIG)
    assert [variant["learning_rate"] for variant in variants] == ["1e-5", "2e-5", "5e-6"]


def test_grid_entries_differ_only_in_the_learning_rate():
    _, variants = _expand(GRID_CONFIG)
    without_lr = [
        {key: value for key, value in variant["grid_hparams"].items() if key != "learning_rate"}
        for variant in variants
    ]
    assert without_lr[0] == without_lr[1] == without_lr[2]
    assert without_lr[0] == {
        "num_train_epochs": 1,
        "document_regularizer_weight": 0.001,
        "query_regularizer_weight": 0.00005,
    }


def test_every_grid_run_gets_its_own_directory_naming_the_learning_rate():
    _, variants = _expand(GRID_CONFIG)
    output_dirs = [variant["output_dir"] for variant in variants]

    assert len(set(output_dirs)) == 3
    for variant, learning_rate in zip(variants, ("1e-5", "2e-5", "5e-6")):
        assert f"lr-{learning_rate}" in variant["run_name"]
        assert variant["run_name"] in variant["output_dir"]


def test_every_grid_run_gets_its_own_benchmark_metrics_directory():
    _, variants = _expand(GRID_CONFIG)
    benchmark_dirs = [_settings(variant).output_dir for variant in variants]

    assert len(set(benchmark_dirs)) == 3
    for variant, benchmark_dir in zip(variants, benchmark_dirs):
        assert benchmark_dir is not None
        assert benchmark_dir.name == variant["run_name"]


@pytest.mark.parametrize("config_path", [GRID_CONFIG, OVERNIGHT_CONFIG])
def test_weekend_configs_benchmark_only_the_final_checkpoint_on_tiny(config_path):
    _, variants = _expand(config_path)
    for variant in variants:
        settings = _settings(variant)
        assert settings.run_pirb is True
        assert settings.pirb_scope == "tiny"
        assert settings.checkpoints == ("final",)
        assert settings.pirb_task_selection is None


@pytest.mark.parametrize("config_path", [GRID_CONFIG, OVERNIGHT_CONFIG])
def test_weekend_configs_share_the_training_settings(config_path):
    _, variants = _expand(config_path)
    for variant in variants:
        backend_config = _backend_config(variant)
        assert backend_config["train_batch_size"] == 4
        assert backend_config["gradient_accumulation_steps"] == 8
        assert backend_config["negatives_per_query"] == 10
        assert backend_config["max_seq_length"] == 512
        assert backend_config["fp16"] is True
        assert backend_config["gradient_checkpointing"] is True
        assert backend_config["save_at_fractions"] == [0.8]
        assert backend_config["warmup_ratio"] == 0.1
        assert backend_config["lr_scheduler_type"] == "linear"
        assert backend_config["num_train_epochs"] == 1
        assert backend_config["document_regularizer_weight"] == 0.001
        assert backend_config["query_regularizer_weight"] == 0.00005
        assert backend_config["splade_activation_stats"]["sample_size"] == 128
        assert backend_config["splade_activation_stats"]["quantization_factor"] == 100


@pytest.mark.parametrize("config_path", [GRID_CONFIG, OVERNIGHT_CONFIG])
def test_weekend_configs_use_a_sampler_the_mix_strategy_accepts(config_path):
    from training.backends.sentence_transformers_backend import _normalize_batch_sampler

    _, variants = _expand(config_path)
    for variant in variants:
        backend_config = _backend_config(variant)
        assert backend_config["dataset_mix_strategy"] == "proportional_batch_best_effort"
        assert _normalize_batch_sampler(backend_config["batch_sampler"]) == "no_duplicates"


@pytest.mark.parametrize("config_path", [GRID_CONFIG, OVERNIGHT_CONFIG])
def test_weekend_configs_reuse_the_optimized_dataset_filter(config_path):
    config, _ = _expand(config_path)
    optimized = load_config(str(REPO_ROOT / "configs" / "splade_hf_dataset_comparison_optimized.yaml"))
    assert config["dataset_filter"] == optimized["dataset_filter"]
    assert (REPO_ROOT / config["dataset_filter"]).is_file()


def test_grid_mix_subsamples_the_rehearsal_sources():
    config, _ = _expand(GRID_CONFIG)
    entries = normalize_train_data_specs(config["train_data"])

    assert [entry.sampling.max_samples if entry.sampling else None for entry in entries] == [None, 30000, 20000]


def test_overnight_mix_only_subsamples_msmarco():
    config, _ = _expand(OVERNIGHT_CONFIG)
    entries = normalize_train_data_specs(config["train_data"])

    assert [entry.sampling.max_samples if entry.sampling else None for entry in entries] == [None, None, 100000]


@pytest.mark.parametrize("config_path", [GRID_CONFIG, OVERNIGHT_CONFIG])
def test_weekend_dataset_paths_are_marked_as_placeholders(config_path):
    assert "TODO" in config_path.read_text(encoding="utf-8")


def test_overnight_learning_rate_is_marked_as_a_placeholder():
    assert "TODO: replace with the best learning rate" in OVERNIGHT_CONFIG.read_text(encoding="utf-8")


def test_run_experiments_can_drive_a_sentence_transformers_grid(monkeypatch):
    import run_experiments

    recorded = {}
    monkeypatch.setattr(
        run_experiments.subprocess, "run", lambda cmd, check: recorded.setdefault("cmd", cmd)
    )
    args = run_experiments.build_parser().parse_args(
        [
            "--backend",
            "sentence-transformers",
            "--training-type",
            "splade",
            "--grid-configuration-file",
            str(GRID_CONFIG),
            "--run_pirb",
        ]
    )
    run_experiments.run_training_mode(args)

    cmd = recorded["cmd"]
    assert cmd[cmd.index("--backend") + 1] == "sentence-transformers"
    assert cmd[cmd.index("--training-type") + 1] == "splade"
    assert cmd[cmd.index("--config") + 1] == str(GRID_CONFIG)


def test_run_experiments_keeps_its_flagembedding_default(monkeypatch):
    import run_experiments

    recorded = {}
    monkeypatch.setattr(
        run_experiments.subprocess, "run", lambda cmd, check: recorded.setdefault("cmd", cmd)
    )
    args = run_experiments.build_parser().parse_args([])
    run_experiments.run_training_mode(args)

    cmd = recorded["cmd"]
    assert cmd[cmd.index("--backend") + 1] == "flagembedding"
    assert cmd[cmd.index("--training-type") + 1] == "embedder"
