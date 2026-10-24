import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_model  # noqa: E402
from training.benchmarks import (  # noqa: E402
    BenchmarkSettings,
    BenchmarkTarget,
    build_pirb_results,
    pirb_results_path,
    read_pirb_task_metrics,
    run_benchmarks_for_targets,
    write_pirb_results_for_targets,
)


def _runlog_entry(task_id, model, metrics, group_id="MaupQA"):
    return json.dumps(
        {
            "task_id": task_id,
            "group_id": group_id,
            "model": f"splade_{model}",
            "metrics": metrics,
            "metadata": {"name": model, "fp16": True},
            "timestamp": "01/01/2026,00:00:00",
        },
        ensure_ascii=False,
    )


def _write_runlog(path, entries):
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return path


def _settings(tmp_path, **overrides):
    defaults = dict(
        run_mteb=False,
        run_pirb=True,
        benchmark_name="NanoBEIR",
        pirb_scope="tiny",
        batch_size=64,
        pirb_max_seq_length=512,
        query_instruction_for_retrieval="",
        pirb_results_dir=tmp_path / "results",
        run_name="example-run",
    )
    defaults.update(overrides)
    return BenchmarkSettings(**defaults)


def test_maupqa_preset_expands_to_every_configured_subset(tmp_path):
    config = tmp_path / "benchmark.json"
    config.write_text(
        json.dumps(
            [
                {"subset": "mkqa", "type": "maupqa"},
                {"task_id": "scifact-pl", "type": "beir"},
                {"subset": "poquad", "type": "maupqa"},
            ]
        ),
        encoding="utf-8",
    )

    assert benchmark_model.resolve_tasks("maupqa", benchmark_config=config) == "maupqa-mkqa,maupqa-poquad"


def test_quick6_preset_lists_the_six_tasks():
    assert benchmark_model.resolve_tasks("quick6") == ",".join(benchmark_model.QUICK6_TASK_IDS)
    assert len(benchmark_model.QUICK6_TASK_IDS) == 6


def test_tiny_preset_keeps_the_pirb_scope():
    assert benchmark_model.resolve_tasks("tiny") == "tiny"


def test_explicit_task_lists_are_normalized():
    assert benchmark_model.resolve_tasks(" scifact-pl , arguana-pl ") == "scifact-pl,arguana-pl"


def test_empty_task_selection_is_rejected():
    with pytest.raises(benchmark_model.BenchmarkModelError):
        benchmark_model.resolve_tasks("  ")


def test_local_models_are_benchmarked_through_an_absolute_path(tmp_path):
    model_dir = tmp_path / "final"
    model_dir.mkdir()
    target = benchmark_model.build_target(str(model_dir), "run")
    assert target.model_ref == str(model_dir.resolve())


def test_hugging_face_models_keep_their_id():
    target = benchmark_model.build_target("sdadas/polish-splade", "run")
    assert target.model_ref == "sdadas/polish-splade"


def test_read_pirb_task_metrics_selects_the_requested_model(tmp_path):
    runlog = _write_runlog(
        tmp_path / "runlog.txt",
        [
            _runlog_entry(
                "maupqa-mkqa",
                "/models/other",
                {"Accuracy@1": 1.0, "Recall@100": 2.0, "MRR@10": 3.0, "NDCG@10": 4.0},
            ),
            _runlog_entry(
                "maupqa-mkqa",
                "/models/wanted",
                {
                    "Accuracy@1": 10.0,
                    "Accuracy@1-5": [10.0, 11.0, 12.0, 13.0, 14.0],
                    "Recall@100": 20.0,
                    "MRR@10": 30.0,
                    "NDCG@10": 40.0,
                },
            ),
            _runlog_entry(
                "scifact-pl",
                "/models/wanted",
                {"Accuracy@1": 1.5, "Recall@100": 2.5, "MRR@10": 3.5, "NDCG@10": 4.5},
                group_id="BEIR-PL",
            ),
        ],
    )

    metrics = read_pirb_task_metrics("/models/wanted", runlog_path=runlog)

    assert sorted(metrics) == ["maupqa-mkqa", "scifact-pl"]
    assert metrics["maupqa-mkqa"] == {
        "NDCG@10": 40.0,
        "MRR@10": 30.0,
        "Recall@100": 20.0,
        "Accuracy@1": 10.0,
    }


def test_read_pirb_task_metrics_ignores_entries_before_the_offset(tmp_path):
    runlog = tmp_path / "runlog.txt"
    old_entry = _runlog_entry("maupqa-mkqa", "/models/wanted", {"NDCG@10": 1.0})
    _write_runlog(runlog, [old_entry])
    offset = runlog.stat().st_size
    with runlog.open("a", encoding="utf-8") as fh:
        fh.write(_runlog_entry("scifact-pl", "/models/wanted", {"NDCG@10": 2.0}) + "\n")

    metrics = read_pirb_task_metrics("/models/wanted", offset=offset, runlog_path=runlog)

    assert sorted(metrics) == ["scifact-pl"]


def test_read_pirb_task_metrics_skips_unreadable_lines(tmp_path):
    runlog = tmp_path / "runlog.txt"
    runlog.write_text(
        "not json\n\n" + _runlog_entry("scifact-pl", "/models/wanted", {"NDCG@10": 2.0}) + "\n",
        encoding="utf-8",
    )

    assert sorted(read_pirb_task_metrics("/models/wanted", runlog_path=runlog)) == ["scifact-pl"]


def test_standalone_results_file_lands_next_to_the_run_name(tmp_path):
    settings = _settings(tmp_path, pirb_results_group_by_label=False)
    assert pirb_results_path(settings, "final") == tmp_path / "results" / "example-run" / "pirb_results.json"


def test_post_training_results_files_are_separated_per_checkpoint(tmp_path):
    settings = _settings(tmp_path)
    assert pirb_results_path(settings, "epoch-0001") == (
        tmp_path / "results" / "example-run" / "epoch-0001" / "pirb_results.json"
    )


def test_disabled_results_dir_skips_the_summary(tmp_path):
    assert pirb_results_path(_settings(tmp_path, pirb_results_dir=None), "final") is None


def test_build_pirb_results_has_the_documented_shape():
    payload = build_pirb_results(
        run_name="example-run",
        model="/models/wanted",
        tasks_preset="quick6",
        config={"pirb_scope": "tiny"},
        results={"scifact-pl": {"NDCG@10": 4.5}},
        created_at="2026-01-01T00:00:00+01:00",
        commit="abc123",
    )

    assert payload["run_name"] == "example-run"
    assert payload["model"] == "/models/wanted"
    assert payload["tasks_preset"] == "quick6"
    assert payload["created_at"] == "2026-01-01T00:00:00+01:00"
    assert payload["git_commit"] == "abc123"
    assert payload["maupqa_label_fix"] is False
    assert payload["config"] == {"pirb_scope": "tiny"}
    assert payload["results"] == {"scifact-pl": {"NDCG@10": 4.5}}


def test_write_pirb_results_for_targets_writes_one_file_per_target(tmp_path):
    runlog = _write_runlog(
        tmp_path / "runlog.txt",
        [
            _runlog_entry("scifact-pl", "/runs/final", {"NDCG@10": 4.5, "MRR@10": 3.5}),
            _runlog_entry("scifact-pl", "/runs/epoch-1", {"NDCG@10": 2.5, "MRR@10": 1.5}),
        ],
    )
    settings = _settings(tmp_path)
    targets = [
        BenchmarkTarget(label="final", path=tmp_path, model="/runs/final"),
        BenchmarkTarget(label="epoch-0001", path=tmp_path, model="/runs/epoch-1"),
    ]

    written = write_pirb_results_for_targets(targets, settings, runlog_offset=0, runlog_path=runlog)

    assert sorted(written) == ["epoch-0001", "final"]
    final = json.loads(written["final"].read_text(encoding="utf-8"))
    assert final["results"]["scifact-pl"]["NDCG@10"] == 4.5
    assert final["config"]["checkpoint_label"] == "final"
    assert final["tasks_preset"] == "tiny"
    epoch = json.loads(written["epoch-0001"].read_text(encoding="utf-8"))
    assert epoch["results"]["scifact-pl"]["NDCG@10"] == 2.5


def test_run_benchmarks_for_targets_writes_results_after_the_run(tmp_path, monkeypatch):
    import training.benchmarks as benchmarks

    runlog = tmp_path / "runlog.txt"
    runlog.write_text("", encoding="utf-8")
    monkeypatch.setattr(benchmarks, "pirb_runlog_path", lambda: runlog)

    def fake_runner(model_dir, settings, *, metric_prefix="", step=None, label="final", **_):
        with runlog.open("a", encoding="utf-8") as fh:
            fh.write(_runlog_entry("scifact-pl", model_dir, {"NDCG@10": 7.0}) + "\n")
        return {f"{metric_prefix}pirb_average_ndcg@10": 7.0}

    settings = _settings(tmp_path, pirb_results_group_by_label=False, parallel_pirb_tasks=False)
    target = BenchmarkTarget(label="final", path=tmp_path, model="/runs/final")

    metrics = run_benchmarks_for_targets([target], settings, runner=fake_runner)

    assert metrics["final"]["final/pirb_average_ndcg@10"] == 7.0
    results_file = tmp_path / "results" / "example-run" / "pirb_results.json"
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    assert payload["model"] == "/runs/final"
    assert payload["results"] == {"scifact-pl": {"NDCG@10": 7.0}}


def test_benchmark_model_builds_settings_that_write_a_single_summary(tmp_path):
    args = benchmark_model.build_parser().parse_args(
        [
            "--model",
            "sdadas/polish-splade",
            "--tasks",
            "quick6",
            "--run-name",
            "soup-a",
            "--results-dir",
            str(tmp_path / "results"),
            "--max-seq-length",
            "256",
            "--no-fp16",
        ]
    )
    settings = benchmark_model.build_settings(args, scope="scifact-pl", run_name="soup-a")

    assert settings.run_pirb is True
    assert settings.run_mteb is False
    assert settings.pirb_fp16 is False
    assert settings.pirb_max_seq_length == 256
    assert settings.tasks_preset == "quick6"
    assert settings.pirb_results_group_by_label is False
    assert pirb_results_path(settings, "soup-a") == tmp_path / "results" / "soup-a" / "pirb_results.json"
