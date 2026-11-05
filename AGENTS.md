# Repository Guidelines

## Project Structure & Module Organization

This repository trains and evaluates embedding models through a backend-selectable Python CLI. Core code lives in `training/`: `train.py` is the main entry point, `config_grid.py` expands YAML grids, `checkpoints.py` handles resume logic, and `backends/` contains `flagembedding`, `sentence-transformers`, and `pylate` integrations. Dataset filtering helpers are in `training/dataset_filters/`. The second module, `distill/`, compresses a SPLADE model into a smaller student with the same tokenizer (student surgery, representation distillation on unlabeled text, fidelity checks; see `docs/distillation.md`); it shares only the sparse-gradient decoder with `training/`, and the ranking stage of a student still runs through `training.train`.

Generic configuration files live in `configs/` (smoke configs, dataset filter profiles, PIRB task sets); prefer adding new smoke YAMLs there. An experiment series keeps its own configs, queues, results and notes under `experiments/<series>/` (currently `experiments/polish-splade-2026-10/`); put new experiment YAMLs there, not in `configs/`. Tests are under `tests/`, docs under `docs/`, generic utility scripts (queue helpers, evaluation regimes, result tables, guards) under `scripts/`, and vendored benchmark code under `third_party/pirb/`. Dataset building and annotation tools belong to the separate data kitchen repository, not here. Treat `runs/`, `models/`, `datasets/`, `logs/`, `cache/`, `wandb/`, `.venv*`, and `dataset-*` directories as generated artifacts, not source.

## Build, Test, and Development Commands

Create the recommended environment:

```bash
uv venv --python 3.10.4 .venv
uv pip sync requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128 --index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

Install only one backend when working narrowly:

```bash
pip install -r requirements/requirements-sentence-transformers.txt
pip install -r requirements/requirements-flagembedding.txt
pip install -r requirements/requirements-pylate.txt
```

Run the CLI with a YAML config:

```bash
python -m training.train --backend sentence-transformers --training-type splade --config configs/smoke_sentence_transformers_splade.yaml
```

Run tests:

```bash
pytest -q
```

## Coding Style & Naming Conventions

Use Python 3.10-compatible code, 4-space indentation, type hints where they clarify interfaces, and `pathlib.Path` for filesystem work. Keep backend-specific behavior inside the relevant module in `training/backends/`; shared CLI and grid logic should remain backend-neutral. Use snake_case for functions, variables, and YAML keys. Name smoke configs with the existing `smoke_<backend>_<training_type>.yaml` pattern.

## Testing Guidelines

Tests use `pytest`. Add unit tests in `tests/test_*.py`, and prefer lightweight fixtures or `tmp_path` over real training runs. Keep optional backend imports lazy; existing tests assert that importing `training.train` does not import heavy backend packages. For smoke training commands, disable external logging with `WANDB_MODE=disabled` unless the run intentionally reports to W&B.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects such as `Add batch sampler without duplicates` and `Fix bugs related to herbert and wandb run names`. Follow that style: describe the behavior change, not the implementation mechanics. Pull requests should include a concise summary, the configs or backends touched, tests run, and any dataset/model/checkpoint assumptions. Link related issues when available, and avoid committing generated outputs from `runs/`, `cache/`, or `wandb/`.

## Security & Configuration Tips

Do not commit secrets, W&B credentials, private model tokens, or large local datasets. Prefer environment variables for credentials and keep machine-specific paths out of shared YAML unless they are documented examples.
