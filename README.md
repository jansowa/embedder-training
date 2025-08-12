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
    --index-url https://pypi.org/simple
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
3. Execute:
```shell
python run_experiments --benchmark_name BENCHMARK_NAME --grid-configuration-file GRID_CONFIGURATION_FILE
```
By default, the script will select the configuration from `configs/grid.yaml` and run the NanoBEIR benchmark.