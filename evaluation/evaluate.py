from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import os

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import SentenceTransformer


_DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
)


def _ensure_dataset(name: str, root: str = "./beir_data") -> str:
    """Pobiera i rozpakowuje zbiór, jeśli trzeba; zwraca ścieżkę folderu."""
    target = Path(root) / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  → Pobieram i rozpakowuję {name} …")
        util.download_and_unzip(_DATASET_URL.format(name), str(target.parent))
    return str(target)


def evaluate_sentence_transformer(
    model_path: str,
    dataset_names: List[str] | None = None,
    batch_size: int = 16,
    k_values: List[int] = (10,),
    embedding_cache_dir: str = "./beir_embeddings",
    dataset_dir: str = "./beir_data",
) -> Dict[str, Dict[str, float]]:
    """
    Ewaluacja Sentence-Transformers w BEIR.
    Zwraca słownik {dataset: {metric@k: score}}.
    """
    if dataset_names is None:
        dataset_names = ["scifact", "nfcorpus", "fiqa"]

    beir_model = models.SentenceBERT(model_path)
    results: Dict[str, Dict[str, float]] = {}

    for name in dataset_names:
        print(f"\n=== Evaluating {name.upper()} ===")
        data_path = _ensure_dataset(name, dataset_dir)
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        # folder na cache embeddingów tego zbioru
        emb_dir = Path(embedding_cache_dir) / name
        emb_dir.mkdir(parents=True, exist_ok=True)

        retriever = DenseRetrievalExactSearch(beir_model, batch_size=batch_size)
        evaluator = EvaluateRetrieval(
            retriever, k_values=list(k_values), score_function="cos_sim"
        )

        # 1) zakoduj + zapisz embeddingi (o ile nie istnieją)
        # 2) wyszukaj na ich podstawie przy pomocy FAISS
        run = evaluator.encode_and_retrieve(
            corpus,
            queries,
            encode_output_path=str(emb_dir),
            overwrite=False,         # nie nadpisuj cache
            score_function="cos_sim" # przekazujemy do search_from_files
        )

        # 3) policz metryki
        ndcg, _map, recall, precision = evaluator.evaluate(
            qrels, run, k_values=list(k_values)
        )

        # zachowaj tylko wartości @k z ostatniego k
        k_suffix = f"@{k_values[-1]}"
        results[name] = {
            m: v
            for d in (ndcg, _map, recall, precision)
            for m, v in d.items()
            if m.endswith(k_suffix)
        }

    return results
# print(evaluate_sentence_transformer("answerdotai/ModernBERT-base", dataset_names=["nfcorpus"]))
print(evaluate_sentence_transformer("../ModernBERT-base-st", dataset_names=["nfcorpus"]))