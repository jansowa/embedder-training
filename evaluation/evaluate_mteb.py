import mteb
import wandb
from sentence_transformers import SentenceTransformer

wandb_key = None

if wandb_key:
    wandb.login(key=wandb_key)
    wandb.init(
        project="mteb-eval",      # nazwa projektu w W&B
        name="ModernBERT-base",   # dowolna nazwa tej sesji
        config={                  # konfiguracja, którą łatwo będzie potem przeglądać
            "model_name": "../ModernBERT-base-st",
            "batch_size": 2,
            "benchmark": "NanoBEIR",
        }
    )

# Define the sentence-transformers model name
# model_name = "answerdotai/ModernBERT-base"
model_name = "../ModernBERT-base-st"

model = mteb.get_model(model_name) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)
tasks = mteb.get_benchmarks(names=["NanoBEIR"])
# task_name = "NanoClimateFeverRetrieval"
# task_name = "ArguAna-PL"
# tasks = mteb.get_tasks(tasks=[task_name])

evaluation = mteb.MTEB(tasks=tasks)
run_statistics = evaluation.run(
    model,
    output_folder="results/nano",
    encode_kwargs={"batch_size": 2},
    verbosity=2
)
print(f"{run_statistics[0].scores=}")
print(f"{type(run_statistics)=}")

if wandb_key:
    for stat in run_statistics:
        task = stat.task_name.replace("/", "_")
        # stat.scores to dict, np. {"train": [...], "validation": [...], ...}
        for split, metrics_list in stat.scores.items():
            for i, metrics in enumerate(metrics_list):
                # Możesz też pominąć numerację, jeśli zawsze jest 1 element na split
                prefix = f"{task}/{split}"
                # WandB pozwala logować wiele wartości jednocześnie
                wandb.log({ f"{prefix}/{k}": v for k, v in metrics.items() })

    # 6. Zamknij run
    wandb.finish()