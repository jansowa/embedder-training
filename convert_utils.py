import gc
import torch
from sentence_transformers import models, SentenceTransformer
import mteb
import subprocess
import tempfile
import json
import sys
from pathlib import Path


def convert_to_sentence_transformer(input_dir: str, output_dir: str) -> None:
    """
    Converts Transformer model into SentenceTransformer model with cls pooling
    """
    transformer = models.Transformer(input_dir)

    pooling = models.Pooling(
        word_embedding_dimension=transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[transformer, pooling])
    model.save(output_dir)

    del model, transformer, pooling
    torch.cuda.empty_cache()
    gc.collect()


def flatten(results: list) -> dict[str, float]:
    flat_results = dict()
    ndcg_at_10 = []
    ndcg_at_5 = []
    ndcg_at_3 = []
    for result in results:
        for score_key, score_value in result.scores.items():
            for metric_name, metric_value in score_value[0].items():
                if isinstance(metric_value, str) or isinstance(metric_value, list):
                    continue
                metric_value = float(metric_value)
                flat_results[f"{result.task_name}/{score_key}/{metric_name}"] = metric_value
                if metric_name.endswith("ndcg_at_10"):
                    ndcg_at_10.append(metric_value)
                if metric_name.endswith("ndcg_at_5"):
                    ndcg_at_5.append(metric_value)
                if metric_name.endswith("ndcg_at_3"):
                    ndcg_at_3.append(metric_value)

    if ndcg_at_10:
        flat_results["mean_ndcg_at_10"] = sum(ndcg_at_10) / len(ndcg_at_10)
    if ndcg_at_5:
        flat_results["mean_ndcg_at_5"] = sum(ndcg_at_5) / len(ndcg_at_5)
    if ndcg_at_3:
        flat_results["mean_ndcg_at_3"] = sum(ndcg_at_3) / len(ndcg_at_3)

    return flat_results

def run_mteb(st_dir: str, tasks, batch_size: int=64):
    model = mteb.get_model(st_dir)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=None,
                             encode_kwargs={"batch_size": batch_size})
    flat_results = flatten(results)
    return flat_results



def run_pirb(st_dir: str, query_instruction_for_retrieval: str, max_seq_length: int=512, scope: str = "tiny") -> dict[str, float]:
    # Example result: TODO
    pirb_run_benchmark_path = "third_party/pirb/run_benchmark.py"

    tmpdir = Path(tempfile.mkdtemp(prefix="pirb_"))
    models_cfg = tmpdir / "models_config.json"
    results_json = tmpdir / "results.json"

    cfg = [{
        "name": st_dir,
        "bf16": True,
        "max_seq_length": max_seq_length,
        "q_prefix_name": query_instruction_for_retrieval,
    }]
    models_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    script_path = Path(pirb_run_benchmark_path).resolve()
    pirb_root = script_path.parent

    cmd = [
        sys.executable,
        script_path.name,
        "--models_config", str(models_cfg),
        "--results_json", str(results_json),
        "--scope", scope,
        "--benchmark_config", "config/benchmarks/pirb-without-private.json"
    ]

    subprocess.run(cmd, check=True, cwd=pirb_root)

    data = json.loads(results_json.read_text(encoding="utf-8"))
    metrics = data["results"][0]
    metrics = {f"pirb_{key}": value for key, value in metrics.items()}
    return metrics