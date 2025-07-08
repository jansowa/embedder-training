import mteb

tasks = mteb.get_tasks(task_types=["Retrieval"], languages=["pol"])

# Wylistuj rozmiary (liczba dokumentów)
for t in tasks:
    print(f"{t.metadata}")
    # name = t.metadata.name
    # n_docs = t.metadata.corpus_size or "?"
    # print(f"{name:40s} | {n_docs} dokumentów")