# Embedding trainer and evaluator

A backend-selectable training CLI for retrieval models (dense, Matryoshka,
SPLADE, ColBERT) on top of FlagEmbedding, SentenceTransformers and PyLate, with
post-training evaluation on PIRB (the Polish Information Retrieval Benchmark)
and MTEB, and shell helpers for running long series of measurements serially on
one GPU.

| path | contents |
|---|---|
| `training/` | the CLI (`python -m training.train`), config grids, resume logic, backends, dataset filters, benchmark runners |
| `run_experiments.py` | legacy wrapper: training through `training.train`, and benchmark-only runs |
| `configs/` | smoke configs, dataset filter profiles (`dataset_filters/`), PIRB task sets (`pirb_tasks/`) |
| `scripts/` | queue helpers (`queue_lib.sh`), evaluation regimes, result tables, weight soups, guards ([scripts/README.md](scripts/README.md)) |
| `docs/` | backends and recipes, HPC benchmarks, how to write a queue |
| `experiments/` | one directory per experiment series: its configs, queues, results and notes ([experiments/README.md](experiments/README.md)) |
| `third_party/pirb/` | vendored PIRB fork (git submodule) |
| `tests/` | unit tests (`pytest -q`) |

Training data is built elsewhere: dataset cuts, LLM annotation and the bridges to
the hard-negative miner live in the separate data kitchen repository
(`retrieval-data-kitchen`). Generated artifacts stay out of git: `runs/`,
`models/`, `datasets/`, `logs/`, `cache/`, `wandb/`.

## Environment

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is missing
uv venv --python 3.10.4 .venv
source .venv/bin/activate
uv pip sync requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match
```

The committed `requirements.lock` is resolved for Python 3.10 and the CUDA 12.8
PyTorch wheels (`torch==2.7.1+cu128`). Those wheels bundle their own CUDA
runtime, so what has to be new enough is the driver, not the system toolkit;
they run on any driver of the CUDA 12.x family (the reference 3060 Ti box runs a
12.4 driver) and support Blackwell (`sm_120`) cards.

Regenerating the lock, for example for another wheel variant
(see https://pytorch.org/get-started/locally/ for the index URLs):

```shell
uv pip compile requirements.in \
    -o requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-url https://pypi.org/simple \
    --override overrides.txt \
    --index-strategy unsafe-best-match \
    --python-version 3.10.4
```

`--python-version` matters: resolving with a newer interpreter drops the
conditional dependencies Python 3.10 still needs (`async-timeout`,
`exceptiongroup`, `tomli`).

Single-backend installs (`requirements/requirements-<backend>.txt`) are
described in [docs/training-backends.md](docs/training-backends.md).

### Java for PIRB

PIRB retrieval runs on Anserini through pyserini, and pyserini 0.39 ships a fat
jar built for Java 21. An older system JDK fails at JVM startup with a
`jnius.JavaException` (`UnsupportedClassVersionError`) while loading the
Anserini classes. `third_party/pirb/utils/system.py` reads `JAVA_HOME` and
otherwise guesses the JDK from `which java`, so export the right one before a
benchmark or the full test suite (the queue helpers do it for you):

```shell
export JAVA_HOME=/path/to/jdk-21.0.5+11
```

Without a JDK 21, install one next to the project without touching the system
default:

```shell
mkdir -p ~/opt && cd ~/opt
curl -fsSL -o jdk21.tar.gz \
    "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.5%2B11/OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz"
tar xzf jdk21.tar.gz && rm jdk21.tar.gz
export JAVA_HOME=$HOME/opt/jdk-21.0.5+11
```

`scripts/pirb_resume_e2e.py` downloads the same build into its own `--root`, so
it stays independent of this setup.

### Tests

```shell
pytest -q
```

Tests under `third_party/pirb/` start a real JVM and need `JAVA_HOME` from the
section above; `pytest -q tests experiments` runs without it. Importing
`training.train` must not import the heavy backend packages (a test checks it).

## Training

```shell
python -m training.train \
  --backend sentence-transformers \
  --training-type splade \
  --config configs/smoke_sentence_transformers_splade.yaml
```

Main parameters:
- `--backend`: one of `flagembedding`, `sentence-transformers`, `pylate`.
- `--training-type`: recipe for the selected backend, for example `embedder`, `splade`, `matryoshka`, `colbert`.
- `--config`: YAML config path. CLI values override `backend` and `training_type` from YAML.
- `--run-mteb` / `--run-pirb`: optional post-training evaluation for supported backends.
- `--remove-checkpoints`: remove `checkpoint-*` directories after a successful FlagEmbedding run.
- `--gpus 0,1`: use specific GPU ids. By default the implemented backends use all visible GPUs.
- `--num-gpus 2`: use the first N currently visible GPUs.
- `--no-distributed`: force single-process training.
- `--resume`: resume each run from its latest `checkpoint-*` or preserved epoch checkpoint.
- `--resume-from-checkpoint PATH`: resume a single run from a specific checkpoint directory.

A config lists `architectures` and `hparams`; the grid is their product, and each
entry trains into its own run directory under `runs_dir`. Backends, recipes
(dense, Matryoshka, SPLADE with GradCache and distillation, ColBERT), lazy
imports, declarative dataset filters (`dataset_filter`) and smoke tests are
documented in [docs/training-backends.md](docs/training-backends.md).

Multi-GPU:

```shell
# Only GPU 2 and GPU 3.
python -m training.train --backend sentence-transformers --training-type splade \
  --config configs/smoke_sentence_transformers_splade.yaml --gpus 2,3

# The first two visible GPUs.
python -m training.train --backend pylate --training-type colbert \
  --config configs/smoke_pylate_colbert.yaml --num-gpus 2
```

The equivalent YAML block:

```yaml
distributed:
  enabled: auto
  gpus: [2, 3]
```

Use `gpus: [2]` for one specific GPU, and `num_gpus: 2` for the first two visible GPUs.

For long SentenceTransformers/SPLADE epochs, use `save_strategy: steps` with
`keep_epoch_checkpoints: true`: rotating step checkpoints make resume frequent,
while full epoch checkpoints are preserved under `epoch-checkpoints/`.

Minimal FlagEmbedding smoke test in its own environment:

```shell
uv venv --python 3.10.4 .venv-flagembedding
source .venv-flagembedding/bin/activate
uv pip install -r requirements/requirements-flagembedding.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match

WANDB_MODE=disabled python -m training.train \
  --backend flagembedding \
  --training-type embedder \
  --config configs/smoke_flagembedding_embedder.yaml
```

## Evaluation

After training, `benchmark` in the config evaluates selected checkpoints
(final only when `checkpoints` is omitted):

```yaml
benchmark:
  run_pirb: true
  scope: small
  checkpoints:
    - final
    - epoch: 1
    - step: 20000
```

Benchmark only, without training (PIRB scopes: `tiny` (default), `small`, `all`;
or an explicit `--pirb-tasks` list):

```shell
python run_experiments.py \
  --mode benchmark \
  --run_pirb \
  --pirb_scope tiny \
  --pirb-max-seq-length 512 \
  --remove-pirb-indexes \
  --benchmark-target "/path/to/model::query_instruction: " \
  --benchmark-output-dir results/my-model
```

MTEB (default benchmark NanoBEIR): `python run_experiments.py --run_mteb`.

PIRB numbers depend on how query weights reach Anserini. `scripts/pirb_regime_eval.sh`
evaluates a SPLADE model on a task set from `configs/pirb_tasks/` in one of four
regimes (`quantized` matches the public PIRB leaderboard; `truncated` is what
`run_experiments.py` reports; `old-pyserini`, `new-backend`); see the header of
the script. Result tables over many models: `scripts/queue_summary.py`,
`scripts/regime_task_table.py`, `scripts/seed_noise_table.py`,
`scripts/sparsity_table.py`, `scripts/leaderboard_compare.py`. They read
`$RESULTS_ROOT`, `$REGIMES_ROOT`, `$HOLDOUT_ROOT` (defaults under the current
experiment series, `scripts/result_roots.py`).

Benchmarks on a Slurm cluster (CPU budget of parallel PIRB workers, an sbatch
fragment): [docs/hpc-benchmarks.md](docs/hpc-benchmarks.md).

## Queues

Series of trainings and benchmarks run unattended as bash queues built from
`scripts/queue_lib.sh`: serial, each step guarded (idle GPU, free RAM and disk,
timeout), skipped when its result exists, failures logged and never fatal. How
to write one, the settings and every helper: [docs/queues.md](docs/queues.md).

## Experiments

`experiments/polish-splade-2026-10/` holds the October 2026 fine-tuning series
of `sdadas/polish-splade` (configs, the queues v2 to v25, all results, plans and
findings); [experiments/README.md](experiments/README.md) explains what was
studied and how to read the summaries.
