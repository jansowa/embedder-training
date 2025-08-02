Preparing environment:
TODO :)

Execution instructions:
1. Prepare configuration in configs/
2. Choose benchmark from list:
https://github.com/embeddings-benchmark/mteb/blob/main/docs/benchmarks.md
3. Execute:
```shell
python run_experiments --benchmark_name BENCHMARK_NAME --grid-configuration-file GRID_CONFIGURATION_FILE
```
By default, script will choose config from configs/grid.yaml and run NanoBEIR benchmark.