import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compare_results  # noqa: E402


def _write_metrics(path, scores, *, label="final", model="/runs/example/final", extra=None):
    """Write a metrics.json in the layout the benchmark produces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{label}/" if label else ""
    payload = {f"{prefix}pirb_{task_id}": value for task_id, value in scores.items()}
    payload[f"{prefix}pirb_ndcg_tasks"] = sum(scores.values())
    payload[f"{prefix}pirb_average_ndcg@10"] = sum(scores.values()) / len(scores)
    payload[f"{prefix}pirb_datasets"] = len(scores)
    payload[f"{prefix}pirb_model"] = model
    payload.update(
        {
            "model": model,
            "scope": "tiny",
            "created_at": "2026-01-01T00:00:00+01:00",
            "git_commit": "abc123",
        }
    )
    payload.update(extra or {})
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_leaderboard(path, models):
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "scifact-pl", "groupId": "BEIR-PL"},
                    {"id": "arguana-pl", "groupId": "BEIR-PL"},
                    {"id": "maupqa-mkqa", "groupId": "MaupQA"},
                ],
                "results": [
                    {"id": model_id, "results": {task_id: {"NDCG@10": value} for task_id, value in scores.items()}}
                    for model_id, scores in models.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(argv):
    return compare_results.build_comparison(compare_results.build_parser().parse_args(argv))


def test_pirb_task_id_accepts_prefixed_metric_keys():
    assert compare_results.pirb_task_id("final/pirb_maupqa-mkqa") == "maupqa-mkqa"
    assert compare_results.pirb_task_id("pirb_scifact-pl") == "scifact-pl"


def test_pirb_task_id_rejects_aggregates_and_metadata():
    for key in (
        "final/pirb_average_ndcg@10",
        "final/pirb_datasets",
        "final/pirb_ndcg_tasks",
        "final/pirb_model",
        "final/pirb_task_selection",
        "final/mean_ndcg_at_10",
        "final/NanoSciFact/cosine/ndcg_at_10",
        "model",
        "git_commit",
    ):
        assert compare_results.pirb_task_id(key) is None, key


def test_pirb_task_scores_skips_non_numeric_values():
    metrics = {
        "final/pirb_scifact-pl": 40.0,
        "final/pirb_model": "/runs/example/final",
        "final/pirb_datasets": 2,
        "final/pirb_task_selection": ["scifact-pl"],
        "created_at": "2026-01-01T00:00:00+01:00",
    }
    assert compare_results.pirb_task_scores(metrics) == {"scifact-pl": 40.0}


def test_task_groups_are_derived_without_a_leaderboard():
    assert compare_results.task_group("maupqa-mkqa") == "MaupQA"
    assert compare_results.task_group("poleval-2022-test-A-allegro-faq") == "PolEval-2022"
    assert compare_results.task_group("scifact-pl") == "BEIR-PL"
    assert compare_results.task_group("onet") == "Web Datasets"
    assert compare_results.task_group("gpt-exams") == "Other"


def test_leaderboard_groups_win_over_the_derived_ones():
    assert compare_results.task_group("msmarco-pl", {"msmarco-pl": "Other"}) == "Other"


def test_shared_tasks_reports_the_dropped_ones():
    columns = [
        compare_results.Column("a", "a", "a", {"t1": 1.0, "t2": 2.0}),
        compare_results.Column("b", "b", "b", {"t2": 3.0, "t3": 4.0}),
    ]
    assert compare_results.shared_tasks(columns) == (["t2"], ["t1", "t3"])


def test_columns_are_labelled_after_the_parent_directory(tmp_path):
    _write_metrics(tmp_path / "lr-1e-5" / "metrics.json", {"scifact-pl": 40.0})
    _write_metrics(tmp_path / "lr-2e-5" / "metrics.json", {"scifact-pl": 50.0})

    rendered, _ = _run([str(tmp_path / "lr-1e-5"), str(tmp_path / "lr-2e-5"), "--format", "csv"])

    assert rendered.splitlines()[0].split(",")[4:] == ["lr-1e-5", "lr-2e-5"]


def test_explicit_labels_replace_the_directory_names(tmp_path):
    _write_metrics(tmp_path / "a" / "final" / "metrics.json", {"scifact-pl": 40.0})
    _write_metrics(tmp_path / "b" / "final" / "metrics.json", {"scifact-pl": 50.0})

    rendered, _ = _run(
        [
            str(tmp_path / "a"),
            str(tmp_path / "b"),
            "--labels",
            "soup-a",
            "soup-b",
            "--format",
            "csv",
        ]
    )

    assert rendered.splitlines()[0].split(",")[4:] == ["soup-a", "soup-b"]


def test_label_count_must_match_the_metrics_files(tmp_path):
    _write_metrics(tmp_path / "a" / "metrics.json", {"scifact-pl": 40.0})
    with pytest.raises(compare_results.CompareResultsError, match="--labels has 2 values"):
        _run([str(tmp_path / "a"), "--labels", "one", "two"])


def test_repeated_directory_names_get_distinct_columns(tmp_path):
    _write_metrics(tmp_path / "run-a" / "final" / "metrics.json", {"scifact-pl": 40.0})
    _write_metrics(tmp_path / "run-b" / "final" / "metrics.json", {"scifact-pl": 50.0})

    rendered, _ = _run([str(tmp_path / "run-a"), str(tmp_path / "run-b"), "--format", "csv"])

    assert rendered.splitlines()[0].split(",")[4:] == ["final", "run-b/final"]


def test_mean_and_delta_use_only_the_intersection(tmp_path):
    _write_metrics(tmp_path / "a" / "metrics.json", {"scifact-pl": 40.0, "arguana-pl": 20.0})
    _write_metrics(tmp_path / "b" / "metrics.json", {"scifact-pl": 50.0, "arguana-pl": 30.0, "maupqa-mkqa": 90.0})

    rendered, task_count = _run([str(tmp_path / "a"), str(tmp_path / "b"), "--format", "csv"])

    assert task_count == 2
    rows = {f"{line.split(',')[0]}:{line.split(',')[1]}": line.split(",") for line in rendered.strip().splitlines()[1:]}
    # maupqa-mkqa is scored by only one column, so it is not part of any mean.
    assert "task:maupqa-mkqa" not in rows
    assert rows["mean:mean"][4:] == ["30.00", "40.00"]
    assert rows["delta:delta vs a"][4:] == ["+0.00", "+10.00"]


def test_group_means_are_reported_per_pirb_group(tmp_path):
    _write_metrics(tmp_path / "a" / "metrics.json", {"scifact-pl": 40.0, "maupqa-mkqa": 10.0})
    _write_metrics(tmp_path / "b" / "metrics.json", {"scifact-pl": 60.0, "maupqa-mkqa": 20.0})

    rendered, _ = _run([str(tmp_path / "a"), str(tmp_path / "b"), "--format", "csv"])

    lines = [line.split(",") for line in rendered.strip().splitlines()]
    group_rows = {line[2]: line[3:] for line in lines if line[0] == "group_mean"}
    assert group_rows["MaupQA"] == ["1", "10.00", "20.00"]
    assert group_rows["BEIR-PL"] == ["1", "40.00", "60.00"]


def test_leaderboard_models_become_the_first_reference_columns(tmp_path):
    leaderboard = _write_leaderboard(
        tmp_path / "data.json",
        {
            "./models/polish-splade": {"scifact-pl": 50.0, "arguana-pl": 40.0},
            "./models/mmlw/mmlw-retrieval-roberta-base": {"scifact-pl": 70.0, "arguana-pl": 40.0},
        },
    )
    _write_metrics(tmp_path / "run-a" / "metrics.json", {"scifact-pl": 60.0, "arguana-pl": 40.0})

    rendered, _ = _run([str(tmp_path / "run-a"), "--leaderboard", str(leaderboard)])

    header = [line for line in rendered.splitlines() if line.startswith("| row |")][0]
    assert header.index("./models/polish-splade") < header.index("run-a")
    assert "Reference column: `./models/polish-splade`" in rendered
    delta_row = [line for line in rendered.splitlines() if line.startswith("| delta vs")][0]
    assert delta_row.endswith("| +0.00 | +10.00 | +5.00 |")


def test_missing_leaderboard_model_is_skipped_with_a_warning(tmp_path, capsys):
    leaderboard = _write_leaderboard(tmp_path / "data.json", {"./models/polish-splade": {"scifact-pl": 55.0}})
    _write_metrics(tmp_path / "run-a" / "metrics.json", {"scifact-pl": 60.0})

    rendered, _ = _run([str(tmp_path / "run-a"), "--leaderboard", str(leaderboard)])

    assert "no model './models/mmlw/mmlw-retrieval-roberta-base'" in capsys.readouterr().err
    assert "./models/polish-splade" in rendered


def test_leaderboard_without_any_requested_model_is_an_error(tmp_path):
    leaderboard = _write_leaderboard(tmp_path / "data.json", {"bm25": {"scifact-pl": 55.0}})
    _write_metrics(tmp_path / "run-a" / "metrics.json", {"scifact-pl": 60.0})

    with pytest.raises(compare_results.CompareResultsError, match="none of the requested models"):
        _run([str(tmp_path / "run-a"), "--leaderboard", str(leaderboard)])


def test_skipped_tasks_are_warned_about_and_listed(tmp_path, capsys):
    _write_metrics(tmp_path / "a" / "metrics.json", {"scifact-pl": 40.0})
    _write_metrics(tmp_path / "b" / "metrics.json", {"scifact-pl": 50.0, "maupqa-mkqa": 90.0})

    rendered, _ = _run([str(tmp_path / "a"), str(tmp_path / "b")])

    assert "maupqa-mkqa" in capsys.readouterr().err
    assert "Skipped tasks (missing in at least one column): maupqa-mkqa" in rendered


def test_columns_without_a_shared_task_are_rejected(tmp_path):
    _write_metrics(tmp_path / "a" / "metrics.json", {"scifact-pl": 40.0})
    _write_metrics(tmp_path / "b" / "metrics.json", {"maupqa-mkqa": 90.0})

    with pytest.raises(compare_results.CompareResultsError, match="share no task"):
        _run([str(tmp_path / "a"), str(tmp_path / "b")])


def test_metrics_file_without_task_scores_is_rejected(tmp_path):
    path = tmp_path / "a" / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"final/mean_ndcg_at_10": 0.5}), encoding="utf-8")

    with pytest.raises(compare_results.CompareResultsError, match="no per-task"):
        _run([str(path)])


def test_explicit_reference_column_is_honored(tmp_path):
    _write_metrics(tmp_path / "a" / "metrics.json", {"scifact-pl": 40.0})
    _write_metrics(tmp_path / "b" / "metrics.json", {"scifact-pl": 50.0})

    rendered, _ = _run([str(tmp_path / "a"), str(tmp_path / "b"), "--reference", "b", "--format", "csv"])

    delta = [line.split(",") for line in rendered.splitlines() if line.startswith("delta")][0]
    assert delta[4:] == ["-10.00", "+0.00"]


def test_unknown_reference_column_is_rejected(tmp_path):
    _write_metrics(tmp_path / "a" / "metrics.json", {"scifact-pl": 40.0})
    with pytest.raises(compare_results.CompareResultsError, match="Unknown --reference"):
        _run([str(tmp_path / "a"), "--reference", "nope"])


def test_glob_patterns_are_expanded(tmp_path):
    _write_metrics(tmp_path / "runs" / "a" / "metrics.json", {"scifact-pl": 40.0})
    _write_metrics(tmp_path / "runs" / "b" / "metrics.json", {"scifact-pl": 50.0})

    paths = compare_results.expand_metrics_paths([str(tmp_path / "runs" / "*" / "metrics.json")])

    assert [path.parent.name for path in paths] == ["a", "b"]


def test_unmatched_pattern_is_rejected(tmp_path):
    with pytest.raises(compare_results.CompareResultsError, match="No metrics files matched"):
        compare_results.expand_metrics_paths([str(tmp_path / "missing" / "*.json")])


def test_output_file_receives_the_table(tmp_path):
    _write_metrics(tmp_path / "a" / "metrics.json", {"scifact-pl": 40.0})
    output = tmp_path / "out" / "comparison.md"

    assert compare_results.main([str(tmp_path / "a"), "--output", str(output)]) == 0
    assert "# PIRB comparison (NDCG@10)" in output.read_text(encoding="utf-8")
