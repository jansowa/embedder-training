"""A failed evaluation must not be mistaken for a valid zero score."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name == 'nt' or shutil.which('bash') is None, reason='Bash queue runs on Linux')
@pytest.mark.parametrize('mean,tasks,expected_rc', [(None, 0, 1), (0.0, 10, 0)])
def test_regime_distinguishes_missing_scores_from_valid_zero(tmp_path, mean, tasks, expected_rc):
    library = Path(__file__).resolve().parents[1] / 'scripts/queue_lib.sh'
    result = tmp_path / 'experiments/polish-splade-2026-10/results-regimes/quantized/sample/tiny10.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({'mean_ndcg@10': mean, 'n_tasks': tasks}))
    (tmp_path / 'models/sample').mkdir(parents=True)
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/pirb_regime_eval.sh').write_text('#!/bin/bash\nexit 1\n')
    shell = '''source "$1"
PY="$2"
need_space() { return 0; }
wait_gpu_free() { return 0; }
wait_ram() { return 0; }
regime quantized sample models/sample tiny10
'''
    process = subprocess.run(['bash', '-c', shell, 'test', str(library), sys.executable],
                             cwd=tmp_path, text=True, capture_output=True)
    assert process.returncode == expected_rc, process.stdout + process.stderr
    if tasks == 0:
        assert 'FAILED' in process.stdout
        assert 'already present' not in process.stdout
    else:
        assert 'already present' in process.stdout
