"""Pure helpers of scripts/splade_density_reference.py (no torch, no sentence-transformers)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import splade_density_reference as ref  # noqa: E402


def test_script_import_is_light():
    assert "torch" not in vars(ref)
    assert "sentence_transformers" not in vars(ref)


def test_sample_texts_splits_evenly_and_is_seeded(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text(
        "\n".join(json.dumps({"query": f"qa{i}", "pos": [f"pa{i}"], "neg": [f"na{i}", f"pa{i}"]}) for i in range(10)) + "\n",
        encoding="utf-8",
    )
    second.write_text(json.dumps({"query": "qb", "pos": "pb", "neg": ["nb"]}) + "\n", encoding="utf-8")

    documents, queries, per_file = ref.sample_texts([first, second], 5, 42)
    assert len([d for d in documents if d.startswith(("pa", "na"))]) == 3
    assert sorted(d for d in documents if d.endswith("b")) == ["nb", "pb"]
    assert len(queries) == 4  # 3 from the first file, the only query of the second
    assert per_file[str(first)] == {"documents_available": 20, "queries_available": 10, "documents": 3, "queries": 3}
    assert ref.sample_texts([first, second], 5, 42) == (documents, queries, per_file)


def test_summary_and_fit():
    assert ref.summarize([1, 2, 3, 4, 5]) == {"n": 5, "mean": 3.0, "median": 3.0, "p10": 1.4, "p90": 4.6}
    slope, intercept = ref.linear_fit([100, 200, 300], [250, 450, 650])
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(50.0)
    assert ref.linear_fit([5, 5], [7, 9]) == (0.0, 8.0)


def test_help_does_not_need_torch(capsys):
    with pytest.raises(SystemExit) as excinfo:
        ref.main(["--help"])
    assert excinfo.value.code == 0
    assert "--threshold-at" in capsys.readouterr().out
