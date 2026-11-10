from pathlib import Path
import runpy
import sys
import os

import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_pirb.py"))


@pytest.mark.parametrize("configured", [True, False])
def test_training_allocator_is_removed_before_benchmark_and_cli_is_preserved(monkeypatch, tmp_path, configured):
    events = []
    keys = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
    for key in keys:
        if configured:
            monkeypatch.setenv(key, "expandable_segments:True")
        else:
            monkeypatch.delenv(key, raising=False)
    target = tmp_path / "run_benchmark.py"
    before_argv, before_path = sys.argv, sys.path[:]
    def execute(path, run_name):
        events.append("benchmark")
        assert path == str(target) and run_name == "__main__"
        assert sys.argv == [str(target), "--scope", "full", "--overwrite"]
        assert sys.path[0] == str(tmp_path)
        assert all(key not in os.environ for key in keys)
    monkeypatch.setattr(MODULE["main"].__globals__["runpy"], "run_path", execute)
    MODULE["main"]([str(target), "--scope", "full", "--overwrite"])
    assert events == ["benchmark"]
    assert sys.argv is before_argv and sys.path == before_path
    for key in keys:
        assert os.environ.get(key) == ("expandable_segments:True" if configured else None)
