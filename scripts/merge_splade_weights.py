#!/usr/bin/env python
"""Merge several SPLADE checkpoints into one weight-averaged model.

This implements the "model soup" / WiSE-FT recipe: every tensor of the output is
the weighted average of the matching tensors of the input models. The models must
therefore share an architecture, a tensor layout, and a tokenizer.

The output directory is written as a Hugging Face masked-LM model plus a full
SparseEncoder wrapper, so it can be passed to the PIRB benchmark exactly like any
other SPLADE model directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


DEFAULT_QUANTIZATION_FACTOR = 100
DEFAULT_MAX_SEQ_LENGTH = 512
WEIGHT_SUM_TOLERANCE = 1e-6

# Short Polish sentences covering a few domains, so the sanity check exercises
# different parts of the vocabulary without depending on any dataset.
SANITY_CHECK_SENTENCES = (
    "Jaka jest stolica Polski?",
    "Umowa najmu lokalu mieszkalnego wymaga formy pisemnej.",
    "Objawy grypy to gorączka, kaszel i bóle mięśni.",
    "Jak obliczyć pole trójkąta prostokątnego?",
    "Zwrot towaru zakupionego przez internet jest możliwy w ciągu 14 dni.",
)


class MergeError(ValueError):
    """Raised when the requested models cannot be merged."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="MODEL",
        help=(
            "Two or more models to merge. Each value is a Hugging Face model id or a local directory. "
            "A SparseEncoder directory works as well because its masked-LM model lives in the directory root."
        ),
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Merge weight per model. The values must sum to 1. Defaults to a uniform average.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Shortcut for exactly two models: alpha is the weight of the second model "
            "and 1 - alpha the weight of the first one."
        ),
    )
    parser.add_argument("--output", required=True, type=Path, help="Directory for the merged model.")
    parser.add_argument(
        "--pooling-strategy",
        choices=("max", "sum"),
        default="max",
        help="SpladePooling strategy stored in the SparseEncoder wrapper.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Maximum sequence length stored in the SparseEncoder wrapper and used by the sanity check.",
    )
    parser.add_argument(
        "--quantization-factor",
        type=int,
        default=DEFAULT_QUANTIZATION_FACTOR,
        help="Quantization factor used to count active dimensions, matching scripts/compare_splade_sparsity.py.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device used by the sanity check. Merging itself always runs on the CPU.",
    )
    parser.add_argument(
        "--sanity-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode the built-in sentences with every input model and the merged model.",
    )
    args = parser.parse_args(argv)
    if args.max_seq_length <= 0:
        parser.error("--max-seq-length must be positive")
    if args.quantization_factor <= 0:
        parser.error("--quantization-factor must be positive")
    return args


def resolve_weights(
    models: Sequence[str],
    weights: Sequence[float] | None,
    alpha: float | None,
) -> list[float]:
    """Turn ``--weights``/``--alpha`` into one weight per model."""
    if len(models) < 2:
        raise MergeError("--models requires at least two models to merge.")
    if weights is not None and alpha is not None:
        raise MergeError("Use either --weights or --alpha, not both.")

    if alpha is not None:
        if len(models) != 2:
            raise MergeError(f"--alpha is only defined for exactly two models, got {len(models)}.")
        resolved = [1.0 - float(alpha), float(alpha)]
    elif weights is not None:
        if len(weights) != len(models):
            raise MergeError(f"--weights has {len(weights)} values but {len(models)} models were given.")
        resolved = [float(weight) for weight in weights]
    else:
        resolved = [1.0 / len(models)] * len(models)

    total = sum(resolved)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        formatted = ", ".join(f"{weight:g}" for weight in resolved)
        raise MergeError(f"Merge weights must sum to 1, got {total:g} for [{formatted}].")
    return resolved


def _load_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise MergeError("Merging SPLADE weights requires torch.") from exc
    return torch


def _load_transformers():
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise MergeError("Merging SPLADE weights requires transformers.") from exc
    return AutoModelForMaskedLM, AutoTokenizer


def describe_state_dict_differences(
    reference_name: str,
    reference: dict[str, Any],
    other_name: str,
    other: dict[str, Any],
) -> list[str]:
    """Describe every tensor name or shape mismatch between two state dicts."""
    differences: list[str] = []
    for name in sorted(set(reference) - set(other)):
        differences.append(f"tensor '{name}' is missing in {other_name}")
    for name in sorted(set(other) - set(reference)):
        differences.append(f"tensor '{name}' is missing in {reference_name}")
    for name in sorted(set(reference) & set(other)):
        reference_shape = tuple(reference[name].shape)
        other_shape = tuple(other[name].shape)
        if reference_shape != other_shape:
            differences.append(
                f"tensor '{name}' has shape {reference_shape} in {reference_name} "
                f"but {other_shape} in {other_name}"
            )
    return differences


def _special_tokens(tokenizer: Any) -> dict[str, Any]:
    return {
        str(name): str(value)
        for name, value in sorted(getattr(tokenizer, "special_tokens_map", {}).items())
        if isinstance(value, str)
    }


def describe_tokenizer_differences(
    reference_name: str,
    reference: Any,
    other_name: str,
    other: Any,
) -> list[str]:
    """Describe vocabulary size and special token mismatches between two tokenizers."""
    differences: list[str] = []
    for label, value_of in (
        ("vocab size", lambda tokenizer: int(getattr(tokenizer, "vocab_size", -1))),
        ("tokenizer length", len),
    ):
        reference_value = value_of(reference)
        other_value = value_of(other)
        if reference_value != other_value:
            differences.append(
                f"{label} is {reference_value} in {reference_name} but {other_value} in {other_name}"
            )

    reference_specials = _special_tokens(reference)
    other_specials = _special_tokens(other)
    for name in sorted(set(reference_specials) | set(other_specials)):
        reference_token = reference_specials.get(name)
        other_token = other_specials.get(name)
        if reference_token != other_token:
            differences.append(
                f"special token '{name}' is {reference_token!r} in {reference_name} "
                f"but {other_token!r} in {other_name}"
            )
        if reference_token is not None and reference_token == other_token:
            reference_id = reference.convert_tokens_to_ids(reference_token)
            other_id = other.convert_tokens_to_ids(other_token)
            if reference_id != other_id:
                differences.append(
                    f"special token '{name}' has id {reference_id} in {reference_name} "
                    f"but {other_id} in {other_name}"
                )
    return differences


def average_state_dicts(state_dicts: Sequence[dict[str, Any]], weights: Sequence[float]) -> dict[str, Any]:
    """Average matching tensors in float32 and restore the reference dtype.

    Integer and boolean buffers cannot be averaged meaningfully, so they must be
    identical across the inputs and are copied from the first model.
    """
    torch = _load_torch()
    if len(state_dicts) != len(weights):
        raise MergeError("Each state dict needs exactly one merge weight.")

    reference = state_dicts[0]
    merged: dict[str, Any] = {}
    for name, reference_tensor in reference.items():
        tensors = [state_dict[name] for state_dict in state_dicts]
        if not reference_tensor.is_floating_point():
            for index, tensor in enumerate(tensors[1:], start=1):
                if not torch.equal(tensor, reference_tensor):
                    raise MergeError(
                        f"Non-float tensor '{name}' differs between model 1 and model {index + 1}; "
                        "it cannot be averaged."
                    )
            merged[name] = reference_tensor.clone()
            continue

        accumulator = torch.zeros(reference_tensor.shape, dtype=torch.float32)
        for tensor, weight in zip(tensors, weights):
            accumulator.add_(tensor.detach().to(dtype=torch.float32), alpha=float(weight))
        merged[name] = accumulator.to(dtype=reference_tensor.dtype)
    return merged


def load_models(model_paths: Sequence[str]) -> tuple[list[Any], list[Any]]:
    """Load every masked-LM model and tokenizer and verify that they can be merged."""
    AutoModelForMaskedLM, AutoTokenizer = _load_transformers()
    models = []
    tokenizers = []
    for model_path in model_paths:
        print(f"[INFO] Loading {model_path}", flush=True)
        models.append(AutoModelForMaskedLM.from_pretrained(model_path))
        tokenizers.append(AutoTokenizer.from_pretrained(model_path))

    differences: list[str] = []
    reference_model, reference_tokenizer = models[0], tokenizers[0]
    reference_name = model_paths[0]
    for model_path, model, tokenizer in zip(model_paths[1:], models[1:], tokenizers[1:]):
        differences.extend(
            describe_state_dict_differences(
                reference_name,
                reference_model.state_dict(),
                model_path,
                model.state_dict(),
            )
        )
        differences.extend(
            describe_tokenizer_differences(reference_name, reference_tokenizer, model_path, tokenizer)
        )
    if differences:
        formatted = "\n".join(f"  - {difference}" for difference in differences)
        raise MergeError(f"The selected models are not compatible:\n{formatted}")
    return models, tokenizers


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ModuleNotFoundError:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def sparse_encoder_files(
    *,
    pooling_strategy: str = "max",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    versions: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Build the SparseEncoder wrapper files that sit next to the masked-LM model."""
    versions = versions or {}
    return {
        "modules.json": [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.sparse_encoder.models.MLMTransformer"},
            {
                "idx": 1,
                "name": "1",
                "path": "1_SpladePooling",
                "type": "sentence_transformers.sparse_encoder.models.SpladePooling",
            },
        ],
        "config_sentence_transformers.json": {
            "model_type": "SparseEncoder",
            "__version__": {
                "sentence_transformers": versions.get("sentence_transformers"),
                "transformers": versions.get("transformers"),
                "pytorch": versions.get("torch"),
            },
            "prompts": {"query": "", "document": ""},
            "default_prompt_name": None,
            "similarity_fn_name": "dot",
        },
        "sentence_bert_config.json": {"max_seq_length": int(max_seq_length), "do_lower_case": False},
        "1_SpladePooling/config.json": {
            "pooling_strategy": str(pooling_strategy),
            "activation_function": "relu",
            "word_embedding_dimension": None,
        },
    }


def write_sparse_encoder_directory(
    output_dir: Path,
    *,
    pooling_strategy: str = "max",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> None:
    """Write the SparseEncoder wrapper around an already saved masked-LM model."""
    files = sparse_encoder_files(
        pooling_strategy=pooling_strategy,
        max_seq_length=max_seq_length,
        versions={name: _module_version(name) for name in ("sentence_transformers", "transformers", "torch")},
    )
    for relative_path, payload in files.items():
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_merged_model(
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    merged_state_dict: dict[str, Any],
    *,
    pooling_strategy: str = "max",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> None:
    """Save the merged weights as a masked-LM model plus a SparseEncoder wrapper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(merged_state_dict)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_sparse_encoder_directory(
        output_dir,
        pooling_strategy=pooling_strategy,
        max_seq_length=max_seq_length,
    )


class SpladeEncoder:
    """Encode texts the way PIRB does: relu -> log1p -> max over the sequence."""

    def __init__(self, model: Any, tokenizer: Any, *, device: str, max_seq_length: int):
        self.torch = _load_torch()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_length = int(max_seq_length)

    def encode(self, texts: Sequence[str]) -> Any:
        torch = self.torch
        encoded = self.tokenizer(
            list(texts),
            padding="longest",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.max_seq_length,
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**encoded)["logits"]
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            vectors = torch.max(torch.log1p(torch.relu(logits)) * attention_mask, dim=1).values
        return vectors.detach().float().cpu()


def active_dimension_counts(vectors: Any, quantization_factor: int) -> list[int]:
    """Count the dimensions that survive PIRB's integer quantization."""
    torch = _load_torch()
    quantized = torch.round(vectors * float(quantization_factor))
    return [int(value) for value in (quantized > 0).sum(dim=-1).tolist()]


def cosine_similarities(left: Any, right: Any) -> list[float]:
    torch = _load_torch()
    return [float(value) for value in torch.nn.functional.cosine_similarity(left, right, dim=-1).tolist()]


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def run_sanity_check(
    model_paths: Sequence[str],
    models: Sequence[Any],
    tokenizers: Sequence[Any],
    merged_model: Any,
    merged_tokenizer: Any,
    *,
    device: str,
    max_seq_length: int,
    quantization_factor: int,
    sentences: Sequence[str] = SANITY_CHECK_SENTENCES,
) -> None:
    """Report active dimensions and the cosine between every input and the merge."""
    merged_vectors = SpladeEncoder(
        merged_model, merged_tokenizer, device=device, max_seq_length=max_seq_length
    ).encode(sentences)

    sentence_headers = [f"s{index}" for index in range(1, len(sentences) + 1)]
    active_rows: list[list[str]] = []
    cosine_rows: list[list[str]] = []
    for model_path, model, tokenizer in zip(model_paths, models, tokenizers):
        vectors = SpladeEncoder(model, tokenizer, device=device, max_seq_length=max_seq_length).encode(sentences)
        counts = active_dimension_counts(vectors, quantization_factor)
        active_rows.append([model_path, *[str(count) for count in counts], f"{_mean(counts):.1f}"])
        cosines = cosine_similarities(vectors, merged_vectors)
        cosine_rows.append([model_path, *[f"{value:.4f}" for value in cosines], f"{_mean(cosines):.4f}"])

    merged_counts = active_dimension_counts(merged_vectors, quantization_factor)
    active_rows.append(["<merged>", *[str(count) for count in merged_counts], f"{_mean(merged_counts):.1f}"])

    print()
    print(f"Sanity check on {len(sentences)} built-in sentences (quantization factor {quantization_factor}):")
    print()
    print("Active dimensions:")
    _print_table(["model", *sentence_headers, "mean"], active_rows)
    print()
    print("Cosine similarity against the merged model:")
    _print_table(["model", *sentence_headers, "mean"], cosine_rows)


def _resolve_device(requested: str) -> str:
    torch = _load_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise MergeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        weights = resolve_weights(args.models, args.weights, args.alpha)
        models, tokenizers = load_models(args.models)
        for model_path, weight in zip(args.models, weights):
            print(f"[INFO] Merge weight {weight:g} for {model_path}", flush=True)
        merged_state_dict = average_state_dicts([model.state_dict() for model in models], weights)
        save_merged_model(
            args.output,
            models[0],
            tokenizers[0],
            merged_state_dict,
            pooling_strategy=args.pooling_strategy,
            max_seq_length=args.max_seq_length,
        )
        print(f"[INFO] Saved merged SPLADE model to: {args.output}", flush=True)

        if args.sanity_check:
            # The first model now holds the merged weights, so it is reloaded from
            # disk to keep the per-model comparison honest.
            AutoModelForMaskedLM, AutoTokenizer = _load_transformers()
            device = _resolve_device(args.device)
            run_sanity_check(
                args.models,
                [AutoModelForMaskedLM.from_pretrained(path) for path in args.models],
                tokenizers,
                AutoModelForMaskedLM.from_pretrained(str(args.output)),
                AutoTokenizer.from_pretrained(str(args.output)),
                device=device,
                max_seq_length=args.max_seq_length,
                quantization_factor=args.quantization_factor,
            )
    except MergeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
