"""8-bit GPU inference of a SPLADE ``SparseEncoder`` with torchao kernels (int8 / fp8 tensor cores).

Nothing is exported: the quantization is applied in memory to the ``AutoModelForMaskedLM`` inside the
Sentence Transformers model (``model[0].auto_model``) right after loading, so ``model.encode_query`` /
``model.encode_document`` keep working unchanged. The schemes are dynamic (no calibration), so the result is a
pure function of the published fp16/fp32 weights; what a release can ship is this recipe, not a checkpoint.

Schemes (the ``scheme`` part of a variant):

  fp32, fp16,  no quantization, the model in that dtype (the baselines; fp16 is what PIRB runs on the GPU,
  bf16         fp32 the precision reference)
  int8-dyn     int8 weights per output channel + int8 activations per token, scales computed at run time;
               ``torch._int_mm`` on INT8 tensor cores: Ampere (RTX 30xx, A100) and newer
  fp8-dyn      fp8 e4m3 weights + fp8 activations, dynamic, per row (default) or per tensor;
               ``torch._scaled_mm`` on FP8 tensor cores: compute capability >= 8.9 (Ada, Hopper, Blackwell)
  int8-wo      int8 weights only, activations stay in the base dtype: saves memory, not compute (a control)

Options after a colon, comma-separated (``int8-dyn:nohead,eager``):

  nohead           keep the MLM head (transform dense + the vocabulary decoder) in the base dtype. The decoder
                   (768 -> 50 001 on the 6-layer RoBERTa) is ~47 % of the Linear MACs, so it is where the
                   speed is, and also where the precision matters: its logits decide which terms exist at all
  eager            no ``torch.compile`` (dynamic quantization without compile adds unfused quantize kernels
                   in front of every matmul and is usually slower than the baseline)
  compile          ``torch.compile`` mode "default" (the default unless --compile none)
  max-autotune     ``torch.compile`` mode "max-autotune" (long warm-up, sometimes faster GEMMs)
  fp16, bf16       base dtype of a quantized variant (default bf16, what torchao's int8/fp8 paths are tuned for;
                   int8-dyn in fp16 can overflow in the rescaled int32 accumulator)
  row, tensor      fp8 scale granularity (row: per-token activations and per-channel weights; tensor: one scale
                   each, the safer kernel on GPUs where the row-wise ``_scaled_mm`` is missing)
  fasthead[N]      release.fast_head: the decoder returns the masked token maximum, relu/log1p after it; N = tokens
                   per decoder slice (``fasthead`` = 0 = the whole sequence in one call, bitwise the stock vectors)
  poolchunk<N>     stock path with ``SpladePooling.chunk_size = N`` (no code on the user's side; the logits stay)
  body             compile only the transformer body (``auto_model.base_model``), the MLM head runs eager.
                   Forced with fasthead: its slicing loop and the per-call mask must stay out of the graph

Tied weights: RoBERTa ties ``lm_head.decoder.weight`` to the word embeddings. Quantizing the decoder replaces
its weight with a new (quantized) parameter and unties it; the embeddings keep the original. That is harmless
for inference, but a quantized model must never be saved over a checkpoint.

torchao is not in requirements.lock (it pins against a torch version); install a release matching the torch of
the environment (the torchao README has the table), ideally in a separate venv on the card that runs it.
See docs/gpu-quant.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMES = ("fp32", "fp16", "bf16", "int8-dyn", "fp8-dyn", "int8-wo")
QUANTIZED = ("int8-dyn", "fp8-dyn", "int8-wo")
COMPILE_MODES = ("none", "default", "max-autotune", "reduce-overhead")
FP8_MIN_CAPABILITY = (8, 9)
INT8_MIN_CAPABILITY = (8, 0)

# First components of the MLM head in the masked-LM classes we train (RoBERTa / XLM-R: lm_head.dense,
# lm_head.decoder; BERT: cls.predictions.transform.dense, cls.predictions.decoder; ALBERT: predictions.*;
# ModernBERT: head.dense, decoder).
HEAD_PREFIXES = ("lm_head", "cls", "predictions", "head", "decoder")


class GpuQuantError(ValueError):
    """Invalid variant, unsupported GPU or missing torchao."""


@dataclass(frozen=True)
class Variant:
    scheme: str
    head: bool = True  # quantize the MLM head too
    compile_mode: str = "default"  # "none" = eager
    base_dtype: str = "bf16"  # dtype the model is loaded in (for the fp32 / fp16 / bf16 schemes: the scheme itself)
    fp8_granularity: str = "row"
    fast_head: int | None = None  # None: stock head; 0: fast head, whole sequence; N: N-token decoder slices
    pool_chunk: int | None = None  # SpladePooling.chunk_size on the stock path
    compile_scope: str = "model"  # "model": auto_model.forward; "body": auto_model.base_model.forward

    @property
    def quantized(self) -> bool:
        return self.scheme in QUANTIZED

    @property
    def label(self) -> str:
        parts = []
        if self.fast_head is not None:
            parts.append("fasthead" if self.fast_head == 0 else f"fasthead{self.fast_head}")
        if self.pool_chunk is not None:
            parts.append(f"poolchunk{self.pool_chunk}")
        if self.compile_mode != "none" and self.compile_scope == "body":
            parts.append("body")
        if self.quantized:
            if not self.head:
                parts.append("nohead")
            if self.base_dtype != "bf16":
                parts.append(self.base_dtype)
            if self.scheme == "fp8-dyn" and self.fp8_granularity != "row":
                parts.append(self.fp8_granularity)
        parts.append("eager" if self.compile_mode == "none" else ("compile" if self.compile_mode == "default" else self.compile_mode))
        return f"{self.scheme}:{','.join(parts)}"


def parse_variant(spec: str, default_compile: str = "default") -> Variant:
    """``"int8-dyn:nohead,eager"`` -> Variant. See the module docstring for the grammar."""
    scheme, _, opts = spec.strip().partition(":")
    scheme = scheme.strip()
    if scheme not in SCHEMES:
        raise GpuQuantError(f"unknown scheme {scheme!r} in {spec!r}; known: {', '.join(SCHEMES)}")
    if default_compile not in COMPILE_MODES:
        raise GpuQuantError(f"unknown compile mode {default_compile!r}; known: {', '.join(COMPILE_MODES)}")
    head, compile_mode, granularity = True, default_compile, "row"
    fast_head: int | None = None
    pool_chunk: int | None = None
    scope = "model"
    base = scheme if scheme in ("fp32", "fp16", "bf16") else "bf16"
    for opt in filter(None, (o.strip() for o in opts.split(","))):
        if opt == "nohead":
            head = False
        elif opt == "eager":
            compile_mode = "none"
        elif opt == "compile":
            compile_mode = "default"
        elif opt in ("max-autotune", "reduce-overhead"):
            compile_mode = opt
        elif opt in ("fp16", "bf16"):
            base = opt
        elif opt in ("row", "tensor"):
            granularity = opt
        elif opt.startswith("fasthead"):
            rest = opt[len("fasthead"):]
            if rest and not rest.isdigit():
                raise GpuQuantError(f"bad option {opt!r} in {spec!r} (fasthead or fasthead<N>)")
            fast_head = int(rest) if rest else 0
        elif opt.startswith("poolchunk") and opt[len("poolchunk"):].isdigit() and int(opt[len("poolchunk"):]) > 0:
            pool_chunk = int(opt[len("poolchunk"):])
        elif opt == "body":
            scope = "body"
        else:
            raise GpuQuantError(f"unknown option {opt!r} in {spec!r}")
    if scheme in ("fp32", "fp16", "bf16"):
        if not head or base != scheme or granularity != "row":
            raise GpuQuantError(f"{spec!r}: nohead / dtype / granularity options apply to quantized schemes only")
    elif granularity != "row" and scheme != "fp8-dyn":
        raise GpuQuantError(f"{spec!r}: row / tensor apply to fp8-dyn only")
    if fast_head is not None and pool_chunk is not None:
        raise GpuQuantError(f"{spec!r}: poolchunk applies to the stock head only")
    if fast_head is not None:
        scope = "body"
    if compile_mode == "none":
        scope = "model"
    return Variant(scheme=scheme, head=head, compile_mode=compile_mode, base_dtype=base, fp8_granularity=granularity,
                   fast_head=fast_head, pool_chunk=pool_chunk, compile_scope=scope)


def is_head(fqn: str) -> bool:
    """Whether a module name (as ``named_modules`` of the masked-LM model gives it) belongs to the MLM head."""
    return fqn.split(".", 1)[0] in HEAD_PREFIXES


def linear_filter(include_head: bool):
    """``filter_fn(module, fqn)`` for ``torchao.quantization.quantize_``: every ``nn.Linear``, minus the head."""
    from torch import nn

    def keep(module: Any, fqn: str) -> bool:
        return isinstance(module, nn.Linear) and (include_head or not is_head(fqn))

    return keep


def _torchao_config(variant: Variant) -> Any:
    """The torchao config object; the class API (torchao >= 0.10) with a fallback to the older function API."""
    try:
        import torchao.quantization as q
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise GpuQuantError("torchao is not installed (see docs/gpu-quant.md)") from exc

    def pick(new: str, old: str):
        return getattr(q, new, None) or getattr(q, old)

    if variant.scheme == "int8-dyn":
        return pick("Int8DynamicActivationInt8WeightConfig", "int8_dynamic_activation_int8_weight")()
    if variant.scheme == "int8-wo":
        return pick("Int8WeightOnlyConfig", "int8_weight_only")()
    if variant.scheme == "fp8-dyn":
        try:
            from torchao.quantization import PerRow, PerTensor
        except ImportError:  # pragma: no cover - older layout
            from torchao.quantization.granularity import PerRow, PerTensor
        granularity = PerRow() if variant.fp8_granularity == "row" else PerTensor()
        return pick("Float8DynamicActivationFloat8WeightConfig", "float8_dynamic_activation_float8_weight")(granularity=granularity)
    raise GpuQuantError(f"{variant.scheme} is not a quantized scheme")


def check_device(variant: Variant, device: str = "cuda") -> dict[str, Any]:
    """GPU name and capability; raises when the card cannot run the scheme's kernels."""
    import torch

    if torch.device(device).type == "cpu":  # smoke tests of the benchmark plumbing only
        if variant.quantized:
            raise GpuQuantError(f"{variant.label}: the CPU path takes unquantized variants only")
        return {"gpu": "cpu", "capability": None, "torch": torch.__version__, "torchao": None}
    if not torch.cuda.is_available():
        raise GpuQuantError("CUDA is not available")
    index = torch.device(device).index or 0
    cap = torch.cuda.get_device_capability(index)
    info = {"gpu": torch.cuda.get_device_name(index), "capability": f"{cap[0]}.{cap[1]}", "torch": torch.__version__}
    try:
        import torchao

        info["torchao"] = torchao.__version__
    except ImportError:
        info["torchao"] = None
    need = FP8_MIN_CAPABILITY if variant.scheme == "fp8-dyn" else INT8_MIN_CAPABILITY if variant.scheme == "int8-dyn" else None
    if need and cap < need:
        raise GpuQuantError(f"{variant.scheme} needs compute capability >= {need[0]}.{need[1]}; {info['gpu']} is {info['capability']}")
    if variant.base_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise GpuQuantError(f"{info['gpu']} has no bf16 support")
    return info


def quantize_masked_lm(auto_model: Any, variant: Variant) -> dict[str, Any]:
    """Quantize the Linear layers of a masked-LM model in place; returns what was quantized."""
    if not variant.quantized:
        return {"linear_quantized": 0, "linear_kept": 0}
    from torch import nn
    from torchao.quantization import quantize_

    keep = linear_filter(variant.head)
    names = [n for n, m in auto_model.named_modules() if isinstance(m, nn.Linear)]
    chosen = [n for n, m in auto_model.named_modules() if keep(m, n)]
    quantize_(auto_model, _torchao_config(variant), filter_fn=keep)
    return {"linear_quantized": len(chosen), "linear_kept": len(names) - len(chosen),
            "kept": [n for n in names if n not in set(chosen)]}


def load_sparse_encoder(model_path: str, variant: Variant, device: str = "cuda", max_seq_length: int | None = None):
    """``SparseEncoder`` on the GPU in the variant's dtype, quantized and compiled. Returns (model, info)."""
    import torch
    from sentence_transformers import SparseEncoder

    info = check_device(variant, device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[variant.base_dtype]
    model = SparseEncoder(model_path, device=device, model_kwargs={"torch_dtype": dtype})
    if max_seq_length:
        model.max_seq_length = max_seq_length
    model.eval()
    auto_model = model[0].auto_model
    # Loaders that drop the vocabulary bias (a checkpoint with lm_head.bias but no lm_head.decoder.bias, loaded in
    # fp32 by transformers 4.48) silently change every vector; the mean makes that visible across variants.
    bias = getattr(auto_model.get_output_embeddings(), "bias", None)
    info["head_bias_mean"] = float(bias.float().mean()) if bias is not None else None
    info.update(quantize_masked_lm(auto_model, variant))
    if variant.pool_chunk is not None:
        model[1].chunk_size = variant.pool_chunk
    if variant.fast_head is not None:
        from release.fast_head import enable_fast_head

        enable_fast_head(model, chunk_tokens=variant.fast_head)
    if variant.compile_mode != "none":
        # The bound forward is compiled, not the module: MLMTransformer reads auto_model.config and friends, and
        # the module class (and its state_dict keys) stays what it was. Shapes vary with the padded length, so
        # dynamic shapes are on from the start instead of one recompile per new length.
        target = auto_model.base_model if variant.compile_scope == "body" else auto_model
        target.forward = torch.compile(target.forward, mode=variant.compile_mode, dynamic=True)
    info["variant"] = variant.label
    info["dtype"] = variant.base_dtype
    return model, info
