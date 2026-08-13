"""Tests for PIRB CPU-budget sizing and thread/batch-size plumbing."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _run_pirb_config(monkeypatch, tmp_path, **kwargs):
    """Call run_pirb with a stubbed subprocess and return the written model config."""
    import convert_utils

    output_dir = tmp_path / "pirb"
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def fake_run(cmd, **run_kwargs):
        results_json = Path(cmd[cmd.index("--results_json") + 1])
        results_json.write_text(json.dumps({"results": [{"ndcg_tasks": 1.0}]}), encoding="utf-8")
        fake_run.env = run_kwargs.get("env")
        return SimpleNamespace(returncode=0)

    fake_run.env = None
    monkeypatch.setattr(convert_utils.subprocess, "run", fake_run)
    convert_utils.run_pirb(
        str(model_dir),
        query_instruction_for_retrieval="",
        output_dir=str(output_dir),
        **kwargs,
    )
    config = json.loads((output_dir / "models_config.json").read_text(encoding="utf-8"))
    return config[0], fake_run.env


def test_cpu_budget_prefers_slurm_allocation(monkeypatch):
    from convert_utils import cpu_budget_per_worker, detect_cpu_budget

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(64)))

    assert detect_cpu_budget() == 8
    assert cpu_budget_per_worker(4) == 2
    assert cpu_budget_per_worker(1) == 8


def test_cpu_budget_falls_back_to_affinity(monkeypatch):
    from convert_utils import cpu_budget_per_worker

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(12)))

    assert cpu_budget_per_worker(4) == 3


@pytest.mark.parametrize("slurm_value", ["", "0", "not-a-number"])
def test_cpu_budget_ignores_invalid_slurm_values(monkeypatch, slurm_value):
    from convert_utils import cpu_budget_per_worker

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", slurm_value)
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(6)))

    assert cpu_budget_per_worker(2) == 3


def test_cpu_budget_never_drops_below_one(monkeypatch):
    from convert_utils import cpu_budget_per_worker

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")

    assert cpu_budget_per_worker(8) == 1


def test_cpu_budget_is_unknown_when_detection_fails(monkeypatch):
    from convert_utils import cpu_budget_per_worker

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)

    def unavailable(_pid):
        raise OSError("no affinity mask")

    monkeypatch.setattr("os.sched_getaffinity", unavailable)
    monkeypatch.setattr("os.cpu_count", lambda: None)

    assert cpu_budget_per_worker(4) is None


def test_run_pirb_derives_threads_from_worker_count(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    cfg_entry, _ = _run_pirb_config(monkeypatch, tmp_path, parallel_workers=4)

    assert cfg_entry["threads"] == 2


def test_run_pirb_honours_explicit_threads_and_batch_size(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    cfg_entry, _ = _run_pirb_config(
        monkeypatch,
        tmp_path,
        threads=5,
        batch_size=64,
        parallel_workers=4,
    )

    assert cfg_entry["threads"] == 5
    assert cfg_entry["batch_size"] == 64


def test_run_pirb_keeps_backend_defaults_when_budget_is_unknown(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)

    def unavailable(_pid):
        raise OSError("no affinity mask")

    monkeypatch.setattr("os.sched_getaffinity", unavailable)
    monkeypatch.setattr("os.cpu_count", lambda: None)

    cfg_entry, _ = _run_pirb_config(monkeypatch, tmp_path)

    assert "threads" not in cfg_entry
    assert "batch_size" not in cfg_entry


def test_benchmark_settings_read_pirb_threads_and_batch_size_from_config():
    from training.benchmarks import resolve_benchmark_settings

    settings = resolve_benchmark_settings(
        {"benchmark": {"run_pirb": True, "pirb_threads": 3, "pirb_batch_size": 96}},
        {},
        SimpleNamespace(),
    )

    assert settings.pirb_threads == 3
    assert settings.pirb_batch_size == 96
    assert settings.pirb_parallel_workers == 1


def test_benchmark_settings_cli_overrides_pirb_threads_and_batch_size():
    from training.benchmarks import resolve_benchmark_settings

    settings = resolve_benchmark_settings(
        {"benchmark": {"run_pirb": True, "pirb_threads": 3, "pirb_batch_size": 96}},
        {},
        SimpleNamespace(pirb_threads=7, pirb_batch_size=16),
    )

    assert settings.pirb_threads == 7
    assert settings.pirb_batch_size == 16


def test_benchmark_batch_size_does_not_leak_into_pirb_batch_size():
    from training.benchmarks import resolve_benchmark_settings

    settings = resolve_benchmark_settings(
        {"benchmark": {"run_pirb": True, "batch_size": 128}},
        {},
        SimpleNamespace(),
    )

    assert settings.batch_size == 128
    assert settings.pirb_batch_size is None


def test_benchmark_settings_default_to_backend_values():
    from training.benchmarks import resolve_benchmark_settings

    settings = resolve_benchmark_settings({"benchmark": {"run_pirb": True}}, {}, SimpleNamespace())

    assert settings.pirb_threads is None
    assert settings.pirb_batch_size is None


@pytest.mark.parametrize("key", ["pirb_threads", "pirb_batch_size"])
def test_benchmark_settings_reject_non_positive_values(key):
    from training.benchmarks import resolve_benchmark_settings

    with pytest.raises(ValueError, match=key):
        resolve_benchmark_settings({"benchmark": {"run_pirb": True, key: 0}}, {}, SimpleNamespace())


def test_run_benchmarks_for_model_forwards_pirb_resources(monkeypatch, tmp_path):
    import convert_utils
    from training.benchmarks import BenchmarkSettings, run_benchmarks_for_model

    captured = {}

    def fake_run_pirb(model_dir, **kwargs):
        captured.update(kwargs)
        return {"pirb_ndcg_tasks": 1.0}

    monkeypatch.setattr(convert_utils, "run_pirb", fake_run_pirb)

    settings = BenchmarkSettings(
        run_mteb=False,
        run_pirb=True,
        benchmark_name="NanoBEIR",
        pirb_scope="tiny",
        batch_size=64,
        pirb_max_seq_length=256,
        query_instruction_for_retrieval="",
        output_dir=tmp_path,
        log_to_wandb=False,
        pirb_threads=3,
        pirb_batch_size=64,
        pirb_parallel_workers=4,
    )
    run_benchmarks_for_model(str(tmp_path / "model"), settings, label="final")

    assert captured["threads"] == 3
    assert captured["batch_size"] == 64
    assert captured["parallel_workers"] == 4


def test_parallel_runners_split_the_cpu_budget_across_workers(monkeypatch, tmp_path):
    from training.benchmarks import BenchmarkSettings, BenchmarkTarget, run_benchmarks_for_targets
    import training.benchmarks as benchmarks

    monkeypatch.setattr(benchmarks, "visible_cuda_devices", lambda: ["0", "1"])
    captured = []

    def fake_runner(model_dir, settings, **kwargs):
        captured.append(settings.pirb_parallel_workers)
        return {}

    settings = BenchmarkSettings(
        run_mteb=False,
        run_pirb=True,
        benchmark_name="NanoBEIR",
        pirb_scope="tiny",
        batch_size=64,
        pirb_max_seq_length=256,
        query_instruction_for_retrieval="",
        log_to_wandb=False,
        parallel_pirb_tasks=False,
    )
    targets = [
        BenchmarkTarget(label="epoch-0001", path=tmp_path, step=1),
        BenchmarkTarget(label="final", path=tmp_path, step=2),
    ]
    run_benchmarks_for_targets(targets, settings, runner=fake_runner, prepare_pirb=lambda: None)

    assert captured == [2, 2]
