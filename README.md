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
3. Execute `run_experiments` scripts with parameters. Example with PIRB benchmark, scope 'small' and removing checkpoints:
```shell
python run_experiments.py --run_pirb --remove_checkpoints --pirb_scope "small"
```
Example with MTEB (default benchmark - NanoBEIR):
```shell
python run_experiments.py --run_mteb
```
4. To run benchmark only (without training) run:
```shell
python run_experiments.py \
  --mode benchmark \
  --run_pirb \
  --pirb_scope "small" \
  --benchmark-target "/ścieżka/do/modelu::query_instruction: "
```