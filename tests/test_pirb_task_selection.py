import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.benchmarks import (  # noqa: E402
    QUICK6_TASK_IDS,
    BenchmarkSettings,
    BenchmarkTarget,
    BenchmarkTaskSelectionError,
    PirbTaskGroup,
    benchmark_metrics_payload,
    filter_pirb_task_groups,
    pirb_benchmark_config_path,
    pirb_task_groups_from_config,
    pirb_task_presets,
    resolve_benchmark_settings,
    resolve_pirb_task_selection,
    run_benchmarks_for_targets,
)


BENCHMARK_CONFIG = [
    {"subset": "mkqa", "type": "maupqa"},
    {"subset": "poquad", "type": "maupqa"},
    {"split": "test-A", "domain": "allegro-faq", "type": "poleval"},
    {"task_id": "scifact-pl", "type": "beir"},
    {"type": "mfaq"},
    {"type": "gpt-exams"},
]


@pytest.fixture
def benchmark_config(tmp_path):
    path = tmp_path / "pirb.json"
    path.write_text(json.dumps(BENCHMARK_CONFIG), encoding="utf-8")
    return path


def _settings(**overrides):
    defaults = dict(
        run_mteb=False,
        run_pirb=True,
        benchmark_name="NanoBEIR",
        pirb_scope="tiny",
        batch_size=64,
        pirb_max_seq_length=512,
        query_instruction_for_retrieval="",
    )
    defaults.update(overrides)
    return BenchmarkSettings(**defaults)


def test_task_ids_and_groups_are_read_from_the_benchmark_config(benchmark_config):
    assert pirb_task_groups_from_config(benchmark_config) == {
        "maupqa-mkqa": "MaupQA",
        "maupqa-poquad": "MaupQA",
        "poleval-2022-test-A-allegro-faq": "PolEval-2022",
        "scifact-pl": "BEIR-PL",
        "mfaq": "Other",
        "gpt-exams": "Other",
    }


def test_the_shipped_benchmark_config_is_readable():
    task_groups = pirb_task_groups_from_config()
    assert task_groups["maupqa-mkqa"] == "MaupQA"
    assert task_groups["poleval-2022-test-A-legal-questions"] == "PolEval-2022"
    assert task_groups["arguana-pl"] == "BEIR-PL"
    # Every quick6 task must exist in the benchmark that is actually run.
    assert set(QUICK6_TASK_IDS) <= set(task_groups)
    assert pirb_benchmark_config_path().is_file()


def test_maupqa_preset_covers_the_whole_group(benchmark_config):
    presets = pirb_task_presets(benchmark_config)
    assert presets["maupqa"] == ("maupqa-mkqa", "maupqa-poquad")


def test_quick6_preset_lists_six_tasks(benchmark_config):
    assert pirb_task_presets(benchmark_config)["quick6"] == QUICK6_TASK_IDS
    assert len(QUICK6_TASK_IDS) == 6


def test_preset_names_resolve_to_task_ids(benchmark_config):
    assert resolve_pirb_task_selection("maupqa", benchmark_config=benchmark_config) == (
        "maupqa-mkqa",
        "maupqa-poquad",
    )


def test_task_lists_resolve_in_order(benchmark_config):
    assert resolve_pirb_task_selection(
        ["scifact-pl", "maupqa-mkqa"], benchmark_config=benchmark_config
    ) == ("scifact-pl", "maupqa-mkqa")


def test_comma_separated_strings_are_accepted(benchmark_config):
    assert resolve_pirb_task_selection(
        " scifact-pl , mfaq ", benchmark_config=benchmark_config
    ) == ("scifact-pl", "mfaq")


def test_duplicate_tasks_are_collapsed(benchmark_config):
    assert resolve_pirb_task_selection(
        ["mfaq", "mfaq", "scifact-pl"], benchmark_config=benchmark_config
    ) == ("mfaq", "scifact-pl")


@pytest.mark.parametrize("value", [None, "", "   ", []])
def test_an_empty_selection_means_the_whole_scope(value, benchmark_config):
    assert resolve_pirb_task_selection(value, benchmark_config=benchmark_config) is None


def test_unknown_tasks_are_rejected_with_the_available_presets(benchmark_config):
    with pytest.raises(BenchmarkTaskSelectionError, match="Unknown PIRB task") as excinfo:
        resolve_pirb_task_selection(["scifact-pl", "nope-pl"], benchmark_config=benchmark_config)
    assert "nope-pl" in str(excinfo.value)
    assert "quick6" in str(excinfo.value)


def test_non_list_selections_are_rejected(benchmark_config):
    with pytest.raises(BenchmarkTaskSelectionError, match="preset name or a list"):
        resolve_pirb_task_selection({"tasks": ["scifact-pl"]}, benchmark_config=benchmark_config)


def test_filtering_keeps_only_the_selected_tasks():
    groups = [
        PirbTaskGroup("maupqa-mkqa", ("maupqa-mkqa",), 10),
        PirbTaskGroup("poleval-2022-allegro-faq", ("poleval-2022-test-A-allegro-faq", "poleval-2022-test-B-allegro-faq"), 20),
        PirbTaskGroup("scifact-pl", ("scifact-pl",), 30),
    ]

    filtered = filter_pirb_task_groups(groups, ("maupqa-mkqa", "poleval-2022-test-A-allegro-faq"))

    assert [group.cache_name for group in filtered] == ["maupqa-mkqa", "poleval-2022-allegro-faq"]
    assert filtered[1].task_ids == ("poleval-2022-test-A-allegro-faq",)
    # The cache group is kept whole otherwise, so its size still drives chunking.
    assert filtered[1].size_bytes == 20


def test_filtering_without_a_selection_keeps_every_group():
    groups = [PirbTaskGroup("scifact-pl", ("scifact-pl",), 30)]
    assert filter_pirb_task_groups(groups, None) == groups
    assert filter_pirb_task_groups(groups, ()) == groups


def test_settings_read_the_task_selection_from_the_cli(monkeypatch, benchmark_config):
    import training.benchmarks as benchmarks

    monkeypatch.setattr(benchmarks, "pirb_benchmark_config_path", lambda: benchmark_config)
    settings = resolve_benchmark_settings({}, {}, SimpleNamespace(run_pirb=True, pirb_tasks="maupqa"))

    assert settings.pirb_task_selection == ("maupqa-mkqa", "maupqa-poquad")
    assert settings.pirb_task_selection_label == "maupqa"


def test_settings_read_the_task_selection_from_yaml(monkeypatch, benchmark_config):
    import training.benchmarks as benchmarks

    monkeypatch.setattr(benchmarks, "pirb_benchmark_config_path", lambda: benchmark_config)
    config = {"benchmark": {"run_pirb": True, "scope": "tiny", "tasks": ["scifact-pl", "mfaq"]}}
    settings = resolve_benchmark_settings(config, {}, SimpleNamespace())

    assert settings.pirb_task_selection == ("scifact-pl", "mfaq")
    assert settings.pirb_task_selection_label == ["scifact-pl", "mfaq"]


def test_settings_without_a_selection_keep_the_scope(monkeypatch, benchmark_config):
    import training.benchmarks as benchmarks

    monkeypatch.setattr(benchmarks, "pirb_benchmark_config_path", lambda: benchmark_config)
    settings = resolve_benchmark_settings({"benchmark": {"run_pirb": True}}, {}, SimpleNamespace())

    assert settings.pirb_task_selection is None
    assert settings.pirb_task_selection_label is None


def test_the_selection_replaces_the_scope_of_every_run(tmp_path):
    recorded = []

    def fake_runner(model_dir, settings, *, metric_prefix="", step=None, label="final", **_):
        recorded.append(settings.pirb_scope)
        return {f"{metric_prefix}pirb_datasets": 2}

    settings = _settings(
        pirb_scope="tiny",
        parallel_pirb_tasks=False,
        pirb_task_selection=("scifact-pl", "mfaq"),
        pirb_task_selection_label="my-list",
    )
    target = BenchmarkTarget(label="final", path=tmp_path)

    run_benchmarks_for_targets([target], settings, runner=fake_runner)

    assert recorded == ["scifact-pl,mfaq"]


def test_loaded_task_groups_are_narrowed_to_the_selection(tmp_path, monkeypatch):
    import training.benchmarks as benchmarks

    monkeypatch.setattr(benchmarks, "visible_cuda_devices", lambda: ["0", "1"])
    scopes = []

    def loader(scope):
        return [
            {"cache_name": "scifact-pl", "task_ids": ["scifact-pl"], "size_bytes": 30},
            {"cache_name": "mfaq", "task_ids": ["mfaq"], "size_bytes": 20},
            {"cache_name": "maupqa-mkqa", "task_ids": ["maupqa-mkqa"], "size_bytes": 10},
        ]

    def fake_runner(model_dir, settings, *, metric_prefix="", step=None, label="final", **_):
        scopes.append(settings.pirb_scope)
        return {
            f"{metric_prefix}pirb_datasets": len(settings.pirb_scope.split(",")),
            f"{metric_prefix}pirb_ndcg_tasks": 1.0,
            f"{metric_prefix}pirb_average_ndcg@10": 1.0,
        }

    settings = _settings(pirb_scope="tiny", pirb_task_selection=("scifact-pl", "mfaq"))
    target = BenchmarkTarget(label="final", path=tmp_path)

    metrics = run_benchmarks_for_targets(
        [target], settings, runner=fake_runner, load_pirb_task_groups=loader
    )

    # maupqa-mkqa was loaded but is not selected, so no chunk may run it.
    assert sorted(",".join(scopes).split(",")) == ["mfaq", "scifact-pl"]
    assert metrics["final"]["final/pirb_datasets"] == 2


def test_metrics_payload_records_the_selection_and_metadata():
    settings = _settings(
        pirb_scope="scifact-pl,mfaq",
        pirb_task_selection=("scifact-pl", "mfaq"),
        pirb_task_selection_label="quick6",
    )
    payload = benchmark_metrics_payload({"final/pirb_scifact-pl": 61.2}, model="/runs/final", settings=settings)

    assert payload["model"] == "/runs/final"
    assert payload["scope"] == "scifact-pl,mfaq"
    assert payload["pirb_task_selection"] == "quick6"
    assert payload["created_at"]
    assert "git_commit" in payload
    assert payload["final/pirb_scifact-pl"] == 61.2


def test_metrics_payload_records_an_explicit_task_list():
    settings = _settings(pirb_task_selection=("scifact-pl",), pirb_task_selection_label=["scifact-pl"])
    payload = benchmark_metrics_payload({}, model="/runs/final", settings=settings)
    assert payload["pirb_task_selection"] == ["scifact-pl"]


def test_metrics_payload_omits_the_selection_when_the_whole_scope_runs():
    payload = benchmark_metrics_payload({}, model="/runs/final", settings=_settings())
    assert "pirb_task_selection" not in payload


def test_metrics_file_contains_the_metadata(tmp_path, monkeypatch):
    from training.benchmarks import run_benchmarks_for_model

    monkeypatch.setitem(
        sys.modules,
        "convert_utils",
        SimpleNamespace(
            run_mteb=lambda *a, **k: {},
            run_pirb=lambda model, **kwargs: {"pirb_scifact-pl": 61.2},
        ),
    )
    settings = _settings(
        pirb_scope="scifact-pl",
        output_dir=tmp_path / "benchmarks",
        log_to_wandb=False,
        pirb_task_selection=("scifact-pl",),
        pirb_task_selection_label="my-list",
    )

    run_benchmarks_for_model("/runs/final", settings, metric_prefix="final/", label="final")

    payload = json.loads((tmp_path / "benchmarks" / "final" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["model"] == "/runs/final"
    assert payload["scope"] == "scifact-pl"
    assert payload["pirb_task_selection"] == "my-list"
    assert payload["final/pirb_scifact-pl"] == 61.2
    assert payload["created_at"]


def test_train_cli_accepts_the_task_selection():
    from training.train import build_parser

    args = build_parser().parse_args(["--run-pirb", "--pirb-tasks", "quick6"])
    assert args.pirb_tasks == "quick6"


def test_train_cli_task_selection_survives_the_distributed_relaunch():
    from training.distributed import argv_from_args

    argv = argv_from_args(SimpleNamespace(config="configs/grid.yaml", pirb_tasks="quick6"))
    assert argv[argv.index("--pirb-tasks") + 1] == "quick6"


def test_run_experiments_passes_the_selection_and_output_dir_to_training(monkeypatch):
    import run_experiments

    recorded = {}
    monkeypatch.setattr(
        run_experiments.subprocess, "run", lambda cmd, check: recorded.setdefault("cmd", cmd)
    )
    args = run_experiments.build_parser().parse_args(
        ["--run_pirb", "--pirb-tasks", "quick6", "--benchmark-output-dir", "out/bench"]
    )
    run_experiments.run_training_mode(args)

    cmd = recorded["cmd"]
    assert cmd[cmd.index("--pirb-tasks") + 1] == "quick6"
    assert cmd[cmd.index("--benchmark-output-dir") + 1] == "out/bench"


def test_run_experiments_benchmark_mode_resolves_presets(monkeypatch, benchmark_config):
    import run_experiments
    import training.benchmarks as benchmarks

    monkeypatch.setattr(benchmarks, "pirb_benchmark_config_path", lambda: benchmark_config)
    args = run_experiments.build_parser().parse_args(["--mode", "benchmark", "--run_pirb", "--pirb-tasks", "maupqa"])

    settings = run_experiments.benchmark_settings_from_args(args)

    assert settings.pirb_task_selection == ("maupqa-mkqa", "maupqa-poquad")
    assert settings.pirb_task_selection_label == "maupqa"


def test_run_experiments_benchmark_mode_writes_metrics_json(monkeypatch, tmp_path, benchmark_config):
    import run_experiments
    import training.benchmarks as benchmarks

    monkeypatch.setattr(benchmarks, "pirb_benchmark_config_path", lambda: benchmark_config)
    monkeypatch.setitem(
        sys.modules,
        "convert_utils",
        SimpleNamespace(
            run_mteb=lambda *a, **k: {},
            run_pirb=lambda model, **kwargs: {
                "pirb_scifact-pl": 61.2,
                "pirb_datasets": 1,
                "pirb_average_ndcg@10": 61.2,
            },
            ensure_sentence_transformer=lambda value: value,
        ),
    )
    monkeypatch.setattr(run_experiments, "_load_wandb", lambda: SimpleNamespace(log=lambda *a, **k: None))

    args = run_experiments.build_parser().parse_args(
        [
            "--mode",
            "benchmark",
            "--run_pirb",
            "--pirb-tasks",
            "scifact-pl",
            "--benchmark-output-dir",
            str(tmp_path / "bench"),
        ]
    )
    run_experiments.log_benchmarks_for_st_model(
        "/models/example",
        epoch_idx=0,
        query_instruction="",
        args=args,
        tasks=None,
        metrics_path=tmp_path / "bench" / "final" / "metrics.json",
    )

    payload = json.loads((tmp_path / "bench" / "final" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["model"] == "/models/example"
    assert payload["scope"] == "scifact-pl"
    assert payload["pirb_task_selection"] == "scifact-pl"
    assert payload["epoch0/pirb_scifact-pl"] == 61.2
    assert payload["created_at"]


def test_run_experiments_metrics_root_separates_several_targets(tmp_path):
    import run_experiments

    args = run_experiments.build_parser().parse_args(
        ["--mode", "benchmark", "--benchmark-output-dir", str(tmp_path / "bench")]
    )

    single = run_experiments.benchmark_output_root(args, "sdadas/polish-splade", target_count=1)
    several = run_experiments.benchmark_output_root(args, "sdadas/polish-splade", target_count=2)

    assert single == tmp_path / "bench"
    assert several == tmp_path / "bench" / "sdadas_polish-splade"


def test_run_experiments_metrics_root_is_disabled_without_an_output_dir():
    import run_experiments

    args = run_experiments.build_parser().parse_args(["--mode", "benchmark"])
    assert run_experiments.benchmark_output_root(args, "model", target_count=1) is None
