import mteb
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "answerdotai/ModernBERT-base"
# model_name = "../ModernBERT-base-st"

model = mteb.get_model(model_name) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)
# tasks = mteb.get_benchmarks(names=["NanoBEIR"])
# print(f"{tasks=}")
# task_name = "NanoClimateFeverRetrieval"
task_name = "ArguAna-PL"
tasks = mteb.get_tasks(tasks=[task_name])

evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(
    model,
    output_folder="results/nano",
    # eval_splits=["test"],           # tylko test ⇒ jeszcze szybciej
    encode_kwargs={"batch_size": 16},
    verbosity=2
)