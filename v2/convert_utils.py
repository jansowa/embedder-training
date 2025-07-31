"""
Utils:  (1) konwersja checkpointu FlagEmbedding → SentenceTransformer
        (2) uruchomienie ewaluacji MTEB i zwrot spłaszczonych metryk.

"""
from pathlib import Path
import gc
import torch
import wandb
from sentence_transformers import models, SentenceTransformer
import mteb


def convert_to_sentence_transformer(input_dir: str, output_dir: str) -> None:
    """
    Zamienia checkpoint Hugging Face (Transformer + Tokenizer) w katalogu
    `input_dir` na SentenceTransformer z poolingiem CLS i zapisuje do `output_dir`.
    """
    transformer = models.Transformer(input_dir)  # wraper AutoModel+AutoTokenizer

    pooling = models.Pooling(
        word_embedding_dimension=transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[transformer, pooling])
    model.save(output_dir)

    # sprzątanie pamięci
    del model, transformer, pooling
    torch.cuda.empty_cache()
    gc.collect()


from collections.abc import Mapping

def run_mteb(st_dir: str, tasks):
    """
    Uruchamia MTEB i zwraca płaskie {task/metric: value},
    niezależnie od tego, czy MTEB zwraca listę TaskResult
    (od v1.1) czy słownik (stare wersje).
    """
    model = mteb.get_model(st_dir)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=None,
                             encode_kwargs={"batch_size": 64})

    flat = {}

    # 1️⃣ Nowy format → lista TaskResult
    if isinstance(results, list):
        for res in results:
            # TaskResult.dataset_name = "ag_news"  /  .task_name w niektórych forkach
            task = getattr(res, "dataset_name",
                    getattr(res, "task_name", "unknown_task"))

            # TaskResult.scores -> dict metric→value
            scores = getattr(res, "scores",
                     getattr(res, "score", None))

            if isinstance(scores, Mapping):
                for metric, value in scores.items():
                    flat[f"{task}/{metric}"] = value
            else:  # pojedyncza liczba
                flat[f"{task}/score"] = scores
    else:
        raise TypeError("Nieoczekiwany typ wyniku zwrócony przez MTEB")
    return flat