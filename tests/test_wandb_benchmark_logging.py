"""Tests for logging benchmark metrics when training is skipped.

The benchmark logger only writes into an already open W&B run, and on the path
that skips a finished training no trainer opens one - which silently dropped every
metric of a re-run benchmark.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeWandb:
    """Just enough of the wandb module to observe how it is driven."""

    def __init__(self, *, fail_init: bool = False, active: bool = False):
        self.run = SimpleNamespace(id="already-open") if active else None
        self.init_calls = []
        self.finish_calls = 0
        self.fail_init = fail_init

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise RuntimeError("no credentials")
        self.run = SimpleNamespace(id=kwargs.get("id"))
        return self.run

    def finish(self):
        self.finish_calls += 1
        self.run = None


@pytest.fixture
def fake_wandb(monkeypatch):
    def install(**kwargs):
        module = FakeWandb(**kwargs)
        monkeypatch.setitem(sys.modules, "wandb", module)
        return module

    return install


def test_benchmarks_attach_to_the_run_recorded_for_the_training(fake_wandb, tmp_path):
    from training.wandb_tracking import wandb_run_for_benchmarks

    wandb = fake_wandb()
    with wandb_run_for_benchmarks(tmp_path, report_to=["wandb"], configured_run_id="run-1234"):
        assert wandb.run is not None, "the benchmark must run with an open W&B run"

    assert wandb.init_calls == [{"id": "run-1234", "resume": "allow"}]
    assert wandb.finish_calls == 1


def test_the_run_id_comes_from_the_training_manifest(fake_wandb, tmp_path):
    from training.wandb_tracking import wandb_run_for_benchmarks

    (tmp_path / "training_manifest.json").write_text(
        json.dumps({"schema_version": 1, "wandb_run_id": "from-manifest"}), encoding="utf-8")
    wandb = fake_wandb()

    with wandb_run_for_benchmarks(tmp_path, report_to=["wandb"]):
        pass

    assert wandb.init_calls == [{"id": "from-manifest", "resume": "allow"}]


def test_an_open_run_is_left_alone(fake_wandb, tmp_path):
    """During training the trainer owns the run; a second one would split the data."""
    from training.wandb_tracking import wandb_run_for_benchmarks

    wandb = fake_wandb(active=True)

    with wandb_run_for_benchmarks(tmp_path, report_to=["wandb"], configured_run_id="run-1234"):
        pass

    assert wandb.init_calls == []
    assert wandb.finish_calls == 0


def test_nothing_happens_without_wandb_reporting(fake_wandb, tmp_path):
    from training.wandb_tracking import wandb_run_for_benchmarks

    wandb = fake_wandb()

    with wandb_run_for_benchmarks(tmp_path, report_to=[], configured_run_id="run-1234"):
        pass

    assert wandb.init_calls == []


def test_a_failing_init_does_not_fail_the_benchmark(fake_wandb, tmp_path, capsys):
    from training.wandb_tracking import wandb_run_for_benchmarks

    fake_wandb(fail_init=True)
    entered = []

    with wandb_run_for_benchmarks(tmp_path, report_to=["wandb"], configured_run_id="run-1234"):
        entered.append(True)

    assert entered == [True], "hours of benchmarking must not be lost to a W&B problem"
    assert "Could not attach to W&B run" in capsys.readouterr().out


def test_a_damaged_manifest_does_not_fail_the_benchmark(fake_wandb, tmp_path, capsys):
    from training.wandb_tracking import wandb_run_for_benchmarks

    (tmp_path / "training_manifest.json").write_text("not json at all", encoding="utf-8")
    wandb = fake_wandb()
    entered = []

    with wandb_run_for_benchmarks(tmp_path, report_to=["wandb"]):
        entered.append(True)

    assert entered == [True]
    assert wandb.init_calls == []
    assert "will not reach W&B" in capsys.readouterr().out


def test_a_missing_manifest_warns_instead_of_guessing(fake_wandb, tmp_path, capsys):
    from training.wandb_tracking import wandb_run_for_benchmarks

    wandb = fake_wandb()

    with wandb_run_for_benchmarks(tmp_path, report_to=["wandb"]):
        pass

    assert wandb.init_calls == []
    assert "no training manifest" in capsys.readouterr().out


def test_skipped_training_wraps_its_benchmarks_in_a_wandb_run(monkeypatch, tmp_path):
    """The regression: a re-run benchmark logged nothing because of this gap."""
    import training.backends.sentence_transformers_backend as backend

    (tmp_path / "final").mkdir(parents=True)
    order = []

    class Tracker:
        def __init__(self, output_dir, *, report_to, configured_run_id=None):
            order.append(("open", report_to, configured_run_id))

        def __enter__(self):
            return None

        def __exit__(self, *exc):
            order.append(("close",))
            return False

    monkeypatch.setattr(backend, "should_skip_training_for_final", lambda *a, **k: True)
    monkeypatch.setattr(backend, "wandb_run_for_benchmarks", Tracker)
    monkeypatch.setattr(backend, "_run_sentence_transformers_post_training_benchmarks",
                        lambda *a, **k: order.append(("benchmarks",)))

    request = SimpleNamespace(
        config={"sentence_transformers": {"output_dir": str(tmp_path), "report_to": ["wandb"]}},
        cli_args=SimpleNamespace(resume_if_available=True),
    )

    assert backend._skip_completed_training(request, default_output_dir=str(tmp_path)) is True
    assert order == [("open", ["wandb"], None), ("benchmarks",), ("close",)], \
        "the benchmarks have to run inside the W&B run, not beside it"
