import gc
import subprocess
import tempfile
import json
import sys
from pathlib import Path


def _load_sentence_transformers():
    try:
        from sentence_transformers import models, SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SentenceTransformer conversion requires sentence-transformers. "
            "Install it with: pip install -r requirements/requirements-sentence-transformers.txt"
        ) from exc
    return models, SentenceTransformer


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fix_dynamic_config(model_dir: str) -> None:
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        return

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    auto_map = cfg.get("auto_map")
    if not isinstance(auto_map, dict):
        return

    base_repo = None
    auto_model_val = auto_map.get("AutoModel")
    if isinstance(auto_model_val, str) and "--" in auto_model_val:
        base_repo = auto_model_val.split("--", 1)[0]

    if base_repo is None:
        name_or_path = cfg.get("_name_or_path")
        if isinstance(name_or_path, str) and "/" in name_or_path:
            base_repo = name_or_path

    if base_repo is None:
        return

    cfg["_name_or_path"] = base_repo

    for key, val in list(auto_map.items()):
        if not isinstance(val, str):
            continue
        suffix = val.split("--", 1)[-1]
        auto_map[key] = f"{base_repo}--{suffix}"

    cfg["auto_map"] = auto_map

    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")



def convert_to_sentence_transformer(input_dir: str, output_dir: str, pooling_method:str = "mean") -> None:
    """
    Converts Transformer model into SentenceTransformer model with cls pooling
    """
    models, SentenceTransformer = _load_sentence_transformers()
    print("Converting model to SentenceTransformer format")
    transformer = models.Transformer(
        input_dir,
        config_args={"trust_remote_code": True},
        model_args={"trust_remote_code": True},
        tokenizer_args={"trust_remote_code": True},
    )

    pooling = models.Pooling(
        word_embedding_dimension=transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=pooling_method=="cls",
        pooling_mode_mean_tokens=pooling_method=="mean",
        pooling_mode_max_tokens=pooling_method=="max",
    )

    model = SentenceTransformer(modules=[transformer, pooling])
    model.save(output_dir)

    # Fix config.json after saving the SentenceTransformer wrapper.
    _fix_dynamic_config(output_dir)

    del model, transformer, pooling
    _empty_cuda_cache()
    gc.collect()


def is_sentence_transformer_dir(model_dir: str) -> bool:
    """
    Heuristically check whether a directory already contains a SentenceTransformer model.
    """
    p = Path(model_dir)
    return (p / "modules.json").exists() or (p / "config_sentence_transformers.json").exists()


def ensure_sentence_transformer(
    model_name_or_path: str,
    cache_dir: str = "./cache/sentence-transformers",
) -> str:
    """
    Return a model path or name that can be passed directly to MTEB as a
    SentenceTransformer.

    - If the input is a local SentenceTransformer directory, return it.
    - If the input is a local non-SentenceTransformer directory, create
      <name>-st next to it, convert there, and return that path.
    - If the input is a Hugging Face model name:
        * if SentenceTransformer(model_name_or_path) works, return the name;
        * otherwise convert to cache_dir/<safe_name>-st and return that path.
    """
    _, SentenceTransformer = _load_sentence_transformers()
    p = Path(model_name_or_path)
    if p.exists():
        print("Local model path exists")
        # Local resources.
        if is_sentence_transformer_dir(str(p)):
            print("Directory contains a SentenceTransformer model")
            return str(p.resolve())

        # Reuse a neighboring -st conversion when it already exists.
        out_dir = p.with_name(p.name + "-st")
        if out_dir.exists() and is_sentence_transformer_dir(str(out_dir)):
            print("Returning existing SentenceTransformer conversion")
            return str(out_dir.resolve())

        # Convert from a local directory.
        convert_to_sentence_transformer(str(p), str(out_dir))
        return str(out_dir.resolve())

    # Hugging Face model name.
    try:
        model = SentenceTransformer(model_name_or_path)
    except Exception:
        # This is not a ready-to-use SentenceTransformer model on Hugging Face.
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        safe_name = model_name_or_path.replace("/", "_").replace(".", "_")
        out_dir = cache_root / f"{safe_name}-st"

        if out_dir.exists() and is_sentence_transformer_dir(str(out_dir)):
            return str(out_dir.resolve())

        convert_to_sentence_transformer(model_name_or_path, str(out_dir))
        return str(out_dir.resolve())
    else:
        # It loaded as SentenceTransformer, so no conversion is needed.
        del model
        gc.collect()
        _empty_cuda_cache()
        return model_name_or_path


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
    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MTEB evaluation requires mteb. Install it with the matching backend requirements file."
        ) from exc
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
        "q_prefix": query_instruction_for_retrieval,
        "rm": True,
        "trust_remote_code": True
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
