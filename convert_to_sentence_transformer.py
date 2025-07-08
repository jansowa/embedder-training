def convert_to_sentence_transformer(input_dir: str, output_dir: str) -> None:
    """
    Converts a Hugging Face Transformers model stored in `input_dir`
    into a SentenceTransformer that uses CLS pooling and saves it
    to `output_dir`. All CPU / GPU memory is released before exit.
    """
    # ----- Imports -----
    from sentence_transformers import models, SentenceTransformer
    import torch, gc

    # ----- 1. Load base model + tokenizer as a Sentence-Transformers module -----
    transformer = models.Transformer(input_dir)  # wraps AutoModel + AutoTokenizer

    # ----- 2. Add CLS-based pooling -----
    pooling = models.Pooling(
        word_embedding_dimension=transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,   # use [CLS] vector
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False
    )

    # ----- 3. (Optional) L2-normalization of sentence embeddings -----

    # ----- 4. Assemble full SentenceTransformer and save it -----
    model = SentenceTransformer(modules=[transformer, pooling])
    model.save(output_dir)

    # ----- 5. Free memory (CPU & GPU) -----
    del model, transformer, pooling
    torch.cuda.empty_cache()   # clear GPU cache if CUDA was used
    gc.collect()               # force Python garbage collection

convert_to_sentence_transformer("./ModernBERT-base", "./ModernBERT-base-st")