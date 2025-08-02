import gc
import torch
from sentence_transformers import models, SentenceTransformer
import mteb


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


from collections.abc import Mapping

def run_mteb(st_dir: str, tasks, batch_size: int=64):
    model = mteb.get_model(st_dir)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=None,
                             encode_kwargs={"batch_size": batch_size})

    flat = {}

    if isinstance(results, list):
        for res in results:
            task = getattr(res, "dataset_name",
                    getattr(res, "task_name", "unknown_task"))

            scores = getattr(res, "scores",
                     getattr(res, "score", None))

            if isinstance(scores, Mapping):
                for metric, value in scores.items():
                    flat[f"{task}/{metric}"] = value
            else:
                flat[f"{task}/score"] = scores
    else:
        raise TypeError("Nieoczekiwany typ wyniku zwrócony przez MTEB")
    return flat