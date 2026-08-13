# Preparing environment:
## CUDA 12.4 / 12.6:
1. Install uv:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh 
exec $SHELL 
```
2. Create venv with Python 3.10.4:
```shell
uv venv --python 3.10.4 .venv
```
3. Activate venv:
```shell
source .venv/bin/activate
```
4. Synchronize packages:
```shell
uv pip sync requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match
```
## Another CUDA version:
Before synchronizing packages, you must generate the lock file. Check https://pytorch.org/get-started/locally/ for the specific URL and use it during the generation. Example for CUDA 12.8:
```shell
uv pip compile requirements.in \
    -o requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-url https://pypi.org/simple \
    --override overrides.txt \
    --index-strategy unsafe-best-match
```

# Execution instructions:
1. Prepare configuration in configs/
2. Choose benchmark from list:
https://github.com/embeddings-benchmark/mteb/blob/main/docs/benchmarks.md
3. Execute the backend-selectable training CLI:
```shell
python -m training.train \
  --backend flagembedding \
  --training-type embedder \
  --config configs/grid.yaml
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

Running the benchmarks on a Slurm cluster, including how the CPU budget is split
between parallel PIRB workers and a recommended sbatch fragment, is documented in
`docs/hpc-benchmarks.md`.

Available backends, installation variants, lazy-import behavior, SentenceTransformers dense/Matryoshka/SPLADE training, PyLate ColBERT training, extension-point training types, and smoke tests are documented in `docs/training-backends.md`.
Declarative pre-training dataset filters can be configured with `dataset_filter`; the profile format, cache behavior, and filtered smoke test are also documented there.

Multi-GPU examples:
```shell
# Use all visible GPUs.
python -m training.train \
  --backend sentence-transformers \
  --training-type splade \
  --config configs/polish_splade_dataset_small_multiple_lr.yaml

# Use only GPU 2 and GPU 3.
python -m training.train \
  --backend sentence-transformers \
  --training-type splade \
  --config configs/polish_splade_dataset_small_multiple_lr.yaml \
  --gpus 2,3

# Use the first two visible GPUs.
python -m training.train \
  --backend pylate \
  --training-type colbert \
  --config configs/smoke_pylate_colbert.yaml \
  --num-gpus 2
```

The equivalent YAML block is:
```yaml
distributed:
  enabled: auto
  gpus: [2, 3]
```
Use `gpus: [2]` for one specific GPU, and `num_gpus: 2` for the first two visible GPUs.

For long SentenceTransformers/SPLADE epochs, use `save_strategy: steps` with `keep_epoch_checkpoints: true`: rotating step checkpoints make resume frequent, while full epoch checkpoints are preserved under `epoch-checkpoints/`.

Post-training benchmarks can evaluate selected saved checkpoints for supported backends. When `benchmark.checkpoints` is omitted, the backend keeps its existing final-only behavior.

```yaml
benchmark:
  run_pirb: true
  scope: small
  checkpoints:
    - final
    - epoch: 1
    - epoch: 2
    - step: 20000
```

Minimal FlagEmbedding smoke test:
```shell
uv venv --python 3.10.4 .venv-flagembedding
source .venv-flagembedding/bin/activate
uv pip install -r requirements/requirements-flagembedding.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match

WANDB_MODE=disabled python -m training.train \
  --backend flagembedding \
  --training-type embedder \
  --config configs/smoke_flagembedding_embedder.yaml
```

4. The legacy `run_experiments.py` wrapper still works and delegates training to `python -m training.train`. Example with PIRB benchmark, scope 'small' and removing checkpoints:
```shell
python run_experiments.py --run_pirb --remove_checkpoints --pirb_scope "small"
```
Example with MTEB (default benchmark - NanoBEIR):
```shell
python run_experiments.py --run_mteb
```
5. To run benchmark only (without training) run:
```shell
python run_experiments.py \
  --mode benchmark \
  --run_pirb \
  --pirb_scope "small" \
  --benchmark-target "/path/to/model::query_instruction: "
```
