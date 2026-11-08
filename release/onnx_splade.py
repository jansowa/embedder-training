#!/usr/bin/env python
"""Export a SPLADE checkpoint to ONNX: the fp32 graph and three int8 forms for CPU inference.

Variants (each one a self-contained directory: ``model.onnx``, tokenizer, ``config.json``, ``onnx_splade.json``):

  fp32          the exported graph (reference for every comparison)
  dynamic-int8  weights int8 (per channel), activations quantized on the fly per batch; no calibration
  static-int8   QDQ graph, weights and activations int8 with activation ranges from calibration texts
  mixed-int8    static int8 except chosen blocks kept in fp32: by default the MLM head (vocabulary
                projection, whose output goes through relu / log1p / max and decides which terms survive),
                optionally plus the ``--mixed-auto-k`` most sensitive blocks found by a per-block probe;
                ``--mixed-method dynamic`` builds it on the dynamic graph instead (no calibration: on CPUs
                without VNNI the static ranges lose to per-batch ones, see results/onnx/fix_r2v12b.md)

Only MatMul / Gemm with constant weights are quantized: the attention products (Q K^T, softmax V) stay
fp32 (``--quantize-attention-matmuls`` to change that for static / mixed). ``--quantize-embeddings`` adds the embedding Gather (weight-only int8, shrinks the
file by ~a third on a 50 k vocabulary).

Built for a busy machine: everything runs on the CPU (CUDA hidden before torch is imported, the GPU is
left to the training queue), ``--threads`` caps torch and onnxruntime, the calibrator gets
``--calib-chunk`` batches at a time (MinMax reduces every tensor to two scalars inside the graph; the
histogram methods hold full tensors, the [batch, seq, vocab] logits included), and the activation ranges
are cached on disk so static and mixed (and every sensitivity trial) share one calibration pass.

Usage:
  CUDA_VISIBLE_DEVICES= .venv/bin/python -m release.onnx_splade --model models/arm-dense-kl-n5kj-ens400 \
      --out models/onnx/n5kj --variants fp32,dynamic,static,mixed --calib-texts datasets/kd/kd_v1.jsonl \
      --threads 4
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import logging
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from release.fidelity import compare_vectors
from release.onnx_encoder import META_FILE, MODEL_FILE, OnnxSplade, read_meta
from release.texts import parse_quotas, sample_texts

LOG = logging.getLogger("release.onnx_splade")

VARIANT_NAMES = {"fp32": "fp32", "dynamic": "dynamic-int8", "static": "static-int8", "mixed": "mixed-int8"}
WEIGHT_OPS = ("MatMul", "Gemm")
FORMAT_VERSION = 1

_HEAD = re.compile(r"(?:^|/)(?:lm_head|cls|predictions|head|decoder|generator_lm_head)(?:/|$)")
_LAYER = re.compile(r"(?:^|/)(?:layer|layers|albert_layers)\.(\d+)(?:/|$)")
_EMB = re.compile(r"(?:^|/)(?:embeddings|embedding_hidden_mapping_in)(?:/|$)")


# --------------------------------------------------------------------------------------------- blocks


def node_block(name: str) -> str:
    """Coarse block of an exported node from its scope path: ``head``, ``layer.N``, ``embeddings`` or ``other``.

    TorchScript-exported names keep the module path (``/roberta/encoder/layer.3/attention/self/query/MatMul``,
    ``/lm_head/decoder/MatMul``, ``/cls/predictions/transform/dense/MatMul``, ``/albert/encoder/
    albert_layer_groups.0/albert_layers.0/...``, ModernBERT ``/model/layers.7/...``, ``/head/dense``, ``/decoder``).
    """
    if _HEAD.search(name):
        return "head"
    layers = _LAYER.findall(name)
    if layers:
        return f"layer.{layers[-1]}"
    if _EMB.search(name):
        return "embeddings"
    return "other"


def quantizable_nodes(model_path: str | Path, quantize_embeddings: bool = False) -> list[dict[str, str]]:
    """Nodes the quantizer would touch: MatMul with a constant B, Gemm, and (optionally) Gather on a constant table."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    constants = {init.name for init in model.graph.initializer}
    constants |= {out for node in model.graph.node if node.op_type == "Constant" for out in node.output}
    nodes = []
    for node in model.graph.node:
        if node.op_type == "MatMul" and len(node.input) > 1 and node.input[1] in constants:
            pass
        elif node.op_type == "Gemm":
            pass
        elif quantize_embeddings and node.op_type == "Gather" and node.input and node.input[0] in constants:
            pass
        else:
            continue
        nodes.append({"name": node.name, "op_type": node.op_type, "block": node_block(node.name)})
    return nodes


def activation_matmul_nodes(model: Any) -> list[str]:
    """MatMuls whose B is not a constant (attention scores and context): activation x activation products."""
    constants = {init.name for init in model.graph.initializer}
    constants |= {out for node in model.graph.node if node.op_type == "Constant" for out in node.output}
    return [n.name for n in model.graph.node if n.op_type == "MatMul" and len(n.input) > 1 and n.input[1] not in constants]


def block_order(blocks: Iterable[str]) -> list[str]:
    def key(block: str) -> tuple[int, int]:
        if block == "embeddings":
            return (0, 0)
        if block.startswith("layer."):
            return (1, int(block.split(".")[1]))
        if block == "head":
            return (2, 0)
        return (3, 0)

    return sorted(set(blocks), key=key)


def nodes_in_blocks(nodes: Sequence[dict[str, str]], blocks: Iterable[str]) -> list[str]:
    wanted = set(blocks)
    return [n["name"] for n in nodes if n["block"] in wanted]


# --------------------------------------------------------------------------------------------- export


def default_max_length(model_dir: Path, fallback: int = 512) -> int:
    cfg = model_dir / "sentence_bert_config.json"
    if cfg.is_file():
        try:
            value = json.loads(cfg.read_text()).get("max_seq_length")
            if value:
                return int(value)
        except (ValueError, OSError):
            pass
    return fallback


def export_fp32(
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    graph: str = "splade",
    opset: int = 17,
    max_length: int | None = None,
    check_texts: Sequence[str] = (),
) -> dict[str, Any]:
    """Export ``AutoModelForMaskedLM`` (+ SPLADE pooling for ``graph="splade"``) to ``out_dir/model.onnx`` in fp32."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    try:
        mlm = AutoModelForMaskedLM.from_pretrained(str(model_dir), torch_dtype=torch.float32, attn_implementation="eager")
    except (TypeError, ValueError):
        mlm = AutoModelForMaskedLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    mlm.eval()
    logits_dim = int(mlm.get_output_embeddings().weight.shape[0])
    vocab_dim = len(tokenizer.get_vocab())
    output_dim = min(logits_dim, vocab_dim) if graph == "splade" else logits_dim

    class SpladeGraph(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module, dim: int, pooled: bool):
            super().__init__()
            self.inner, self.dim, self.pooled = inner, dim, pooled
            self.trim = dim < logits_dim  # padded MLM heads (pl-ModernBERT: 50 008 logits, 50 002 tokens)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            logits = self.inner(input_ids=input_ids, attention_mask=attention_mask).logits
            if not self.pooled:
                return logits
            if self.trim:
                logits = logits[..., : self.dim]
            weights = torch.log1p(torch.relu(logits)) * attention_mask.unsqueeze(-1).to(logits.dtype)
            return weights.amax(dim=1)

    pooled = graph == "splade"
    wrapper = SpladeGraph(mlm, output_dim, pooled).eval()
    output_name = "sparse_vector" if pooled else "logits"
    dummy = tokenizer(
        ["Ala ma kota.", "Dłuższy przykładowy tekst, żeby wejście miało dopełnienie do najdłuższego wiersza."],
        padding=True, return_tensors="pt",
    )
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        output_name: {0: "batch"} if pooled else {0: "batch", 1: "sequence"},
    }
    path = out_dir / MODEL_FILE
    start = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    export_seconds = time.time() - start
    tokenizer.save_pretrained(str(out_dir))
    shutil.copy2(model_dir / "config.json", out_dir / "config.json")

    meta = {
        "format": "onnx-splade",
        "format_version": FORMAT_VERSION,
        "variant": "fp32",
        "graph": graph,
        "model_file": MODEL_FILE,
        "inputs": ["input_ids", "attention_mask"],
        "output": output_name,
        "output_dim": output_dim,
        "logits_dim": logits_dim,
        "max_length": int(max_length or default_max_length(model_dir)),
        "pooling": "max_t log1p(relu(logits)) * attention_mask",
        "source_model": str(model_dir.resolve()),
        "opset": opset,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "export_seconds": round(export_seconds, 1),
    }
    write_meta(out_dir, meta)

    if check_texts:  # the graph against the torch model it came from
        enc =tokenizer(list(check_texts), padding=True, truncation=True, max_length=meta["max_length"], return_tensors="pt")
        with torch.no_grad():
            ref = wrapper(enc["input_ids"], enc["attention_mask"])
            if not pooled:
                ref = torch.log1p(torch.relu(ref)) * enc["attention_mask"].unsqueeze(-1)
                ref = ref.amax(dim=1)
        ref = ref.float().numpy()
        del enc
        got = OnnxSplade(out_dir, threads=torch.get_num_threads()).encode(list(check_texts), batch_size=8)
        stats = compare_vectors(ref[:, :output_dim], got)
        meta["torch_check"] = {
            "n": len(check_texts),
            "max_abs_diff": float(np.abs(ref[:, :output_dim] - got).max()),
            "cosine_min": float(min(stats["cosine"]["p10"], stats["cosine"]["mean"])),
            "active_jaccard_mean": stats["active_jaccard"]["mean"],
        }
        write_meta(out_dir, meta)
    del wrapper, mlm
    gc.collect()
    return meta


def write_meta(out_dir: Path, meta: dict[str, Any]) -> None:
    (Path(out_dir) / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_variant_files(src_dir: Path, dst_dir: Path) -> None:
    """Tokenizer and config next to a quantized graph (everything but the graph and the metadata)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file() and item.name not in (MODEL_FILE, META_FILE) and not item.name.endswith(".onnx.data"):
            shutil.copy2(item, dst_dir / item.name)


def preprocess(src: Path, dst: Path) -> Path:
    """onnxruntime's pre-quantization pass: symbolic shape inference + graph optimisation (folds the transposed
    tied decoder weight into a constant so its MatMul is quantizable)."""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    try:
        quant_pre_process(str(src), str(dst), skip_optimization=False)
    except Exception as exc:  # symbolic shape inference fails on some graphs; ONNX shape inference is enough
        LOG.warning("pre-processing with symbolic shapes failed (%s); retrying without", exc)
        quant_pre_process(str(src), str(dst), skip_optimization=False, skip_symbolic_shape=True)
    return dst


# --------------------------------------------------------------------------------------------- quantization


def _quant_type(name: str):
    from onnxruntime.quantization import QuantType

    return {"s8": QuantType.QInt8, "int8": QuantType.QInt8, "u8": QuantType.QUInt8, "uint8": QuantType.QUInt8}[name.lower()]


def quantize_dynamic_int8(
    src: Path,
    dst: Path,
    *,
    per_channel: bool = True,
    reduce_range: bool = False,
    weight_type: str = "s8",
    nodes_to_exclude: Sequence[str] = (),
    quantize_embeddings: bool = False,
) -> dict[str, Any]:
    from onnxruntime.quantization import quantize_dynamic

    ops = list(WEIGHT_OPS) + (["Gather"] if quantize_embeddings else [])
    quantize_dynamic(
        str(src),
        str(dst),
        op_types_to_quantize=ops,
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=_quant_type(weight_type),
        nodes_to_exclude=list(nodes_to_exclude),
        extra_options={"MatMulConstBOnly": True},
    )
    return {
        "method": "dynamic",
        "op_types": ops,
        "per_channel": per_channel,
        "reduce_range": reduce_range,
        "weight_type": weight_type,
        "activations": "uint8, per batch (dynamic)",
        "nodes_excluded": len(nodes_to_exclude),
    }


def _calibrator_class(method: str):
    """An onnxruntime calibrator whose session runs with a capped thread count (the stock one takes every core).

    Memory: MinMax adds ReduceMin / ReduceMax nodes to the graph, so a batch returns two scalars per tensor and
    the [batch, seq, vocab] logits never leave the session. The histogram methods (percentile, entropy) return
    every calibrated tensor in full, logits included (~200 MB per batch of 4 x 256 tokens on a 50 k vocabulary),
    and keep them until ``collect_data`` returns; ``calibrate`` therefore feeds them a few batches at a time."""
    from onnxruntime.quantization.calibrate import EntropyCalibrater, MinMaxCalibrater, PercentileCalibrater

    base = {"minmax": MinMaxCalibrater, "percentile": PercentileCalibrater, "entropy": EntropyCalibrater}[method]

    class ThreadCappedCalibrater(base):  # type: ignore[misc, valid-type]
        threads: int | None = None

        def create_inference_session(self):
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if self.threads:
                opts.intra_op_num_threads = int(self.threads)
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            self.infer_session = ort.InferenceSession(
                self.augmented_model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )

    return ThreadCappedCalibrater


class _ListReader:
    """CalibrationDataReader over a list of feed dicts."""

    def __init__(self, batches: Sequence[dict[str, np.ndarray]]):
        self._iter = iter(batches)

    def get_next(self):
        return next(self._iter, None)


def calibration_batches(tokenizer: Any, texts: Sequence[str], batch_size: int, max_length: int) -> list[dict[str, np.ndarray]]:
    """Feeds for the calibrator; texts sorted by length so a batch carries little padding (padding positions
    still flow through the encoder and would widen the recorded ranges with values no real token produces)."""
    ordered = sorted(texts, key=len)
    batches = []
    for start in range(0, len(ordered), batch_size):
        enc = tokenizer(
            list(ordered[start:start + batch_size]), padding="longest", truncation=True,
            max_length=max_length, return_tensors="np",
        )
        batches.append({"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})
    return batches


def calibrate(
    model_path: Path,
    batches: Sequence[dict[str, np.ndarray]],
    work_dir: Path,
    *,
    method: str = "minmax",
    percentile: float = 99.999,
    chunk: int = 4,
    threads: int | None = None,
    quantize_embeddings: bool = False,
):
    """Activation ranges (onnxruntime ``TensorsData``) of the quantizable nodes' inputs over ``batches``."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cls = _calibrator_class(method)
    cls.threads = threads
    ops = list(WEIGHT_OPS) + (["Gather"] if quantize_embeddings else [])
    augmented = work_dir / "augmented.onnx"
    kwargs: dict[str, Any] = {"symmetric": False}
    # Not max_intermediate_outputs: in onnxruntime 1.23 MinMaxCalibrater drops (clears without merging) the
    # outputs when that limit is hit. Calling collect_data per chunk merges the ranges after every call.
    if method == "percentile":
        kwargs.update(percentile=percentile, num_bins=2048)
    elif method == "entropy":
        kwargs.update(num_bins=128, num_quantized_bins=128)
    calibrator = cls(str(model_path), ops, str(augmented), **kwargs)
    calibrator.augment_graph()
    calibrator.create_inference_session()
    if method != "minmax":
        chunk = 1  # the histogram collector stacks a call's outputs with np.asarray: batches of different lengths fail
    start = time.time()
    for i in range(0, len(batches), chunk):  # every calibrator keeps all outputs until collect_data returns
        calibrator.collect_data(_ListReader(batches[i:i + chunk]))
    ranges = calibrator.compute_data()
    LOG.info("calibration (%s) on %d batches: %.1f s", method, len(batches), time.time() - start)
    del calibrator
    gc.collect()
    for leftover in work_dir.glob("augmented*"):
        leftover.unlink()
    return ranges


def quantize_static_int8(
    src: Path,
    dst: Path,
    ranges: Any,
    *,
    per_channel: bool = True,
    reduce_range: bool = False,
    activation_type: str = "s8",
    weight_type: str = "s8",
    nodes_to_exclude: Sequence[str] = (),
    quantize_embeddings: bool = False,
    quantize_activation_matmuls: bool = False,
) -> dict[str, Any]:
    """QDQ quantization with precomputed activation ranges (what ``quantize_static`` does after its own
    calibration, so one calibration serves static, mixed and every sensitivity trial).

    The QDQ MatMul handler of onnxruntime 1.23 ignores ``MatMulConstBOnly`` (only the QOperator path reads it),
    so the attention products (Q K^T, softmax V) would be int8 too; they are excluded here by name unless
    ``quantize_activation_matmuls``."""
    from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
    from onnxruntime.quantization.quant_utils import load_model_with_shape_infer

    ops = list(WEIGHT_OPS) + (["Gather"] if quantize_embeddings else [])
    extra = {"MatMulConstBOnly": True, "ActivationSymmetric": False, "WeightSymmetric": True}
    model = load_model_with_shape_infer(Path(src))
    exclude = list(nodes_to_exclude)
    activation_matmuls = activation_matmul_nodes(model)
    if not quantize_activation_matmuls:
        exclude += [name for name in activation_matmuls if name not in set(exclude)]
    quantizer = QDQQuantizer(
        model,
        per_channel,
        reduce_range,
        _quant_type(weight_type),
        _quant_type(activation_type),
        ranges,
        [],
        exclude,
        ops,
        extra,
    )
    quantizer.quantize_model()
    quantizer.model.save_model_to_file(str(dst), False)
    del quantizer, model
    gc.collect()
    return {
        "method": "static",
        "format": "QDQ",
        "op_types": ops,
        "per_channel": per_channel,
        "reduce_range": reduce_range,
        "activation_type": activation_type,
        "weight_type": weight_type,
        "nodes_excluded": len(nodes_to_exclude),
        "activation_matmuls": "int8" if quantize_activation_matmuls else f"fp32 ({len(activation_matmuls)} nodes)",
    }


# --------------------------------------------------------------------------------------------- sensitivity


def block_sensitivity(
    fp32_dir: Path,
    nodes: Sequence[dict[str, str]],
    blocks: Sequence[str],
    texts: Sequence[str],
    ref: np.ndarray,
    work_dir: Path,
    quantize_one: Any,
    *,
    threads: int | None = None,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Error of the export when only one block is int8 and the rest stays fp32, for every block in ``blocks``.

    ``quantize_one(dst, nodes_to_exclude)`` writes a quantized graph; the error is measured against the fp32
    vectors ``ref`` of ``texts``. Most sensitive first (largest 1 - mean cosine)."""
    trial_dir = work_dir / "trial"
    copy_variant_files(fp32_dir, trial_dir)
    write_meta(trial_dir, dict(read_meta(fp32_dir), variant="sensitivity-trial"))
    results = []
    for block in blocks:
        exclude = [n["name"] for n in nodes if n["block"] != block]  # everything but this block stays fp32
        start = time.time()
        quantize_one(trial_dir / MODEL_FILE, exclude)
        got = OnnxSplade(trial_dir, threads=threads).encode(texts, batch_size=batch_size)
        stats = compare_vectors(ref, got)
        results.append({
            "block": block,
            "nodes": sum(1 for n in nodes if n["block"] == block),
            "one_minus_cosine": 1.0 - stats["cosine"]["mean"],
            "active_jaccard": stats["active_jaccard"]["mean"],
            "dims_ratio": stats["dims_ratio"],
            "seconds": round(time.time() - start, 1),
        })
        LOG.info("sensitivity %-12s 1-cos %.5f  jaccard %.4f", block, results[-1]["one_minus_cosine"], results[-1]["active_jaccard"])
        (trial_dir / MODEL_FILE).unlink(missing_ok=True)
        gc.collect()
    shutil.rmtree(trial_dir, ignore_errors=True)
    return sorted(results, key=lambda r: -r["one_minus_cosine"])


# --------------------------------------------------------------------------------------------- pipeline


def check_variant(variant_dir: Path, texts: Sequence[str], ref: np.ndarray, threads: int | None) -> dict[str, Any]:
    start = time.time()
    got = OnnxSplade(variant_dir, threads=threads).encode(texts, batch_size=8)
    stats = compare_vectors(ref, got)
    stats["seconds"] = round(time.time() - start, 1)
    return stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    model_dir, out = Path(args.model), Path(args.out)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANT_NAMES]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {sorted(VARIANT_NAMES)}")
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)
    report: dict[str, Any] = {"source_model": str(model_dir.resolve()), "args": vars(args), "variants": {}}
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Calibration texts and disjoint check texts, one sample of the corpus.
    calib_quota = parse_quotas(args.calib_n)
    check_quota = parse_quotas(args.check_n)
    needs_calib = "static" in variants or ("mixed" in variants and args.mixed_method == "static")
    texts_by_kind: dict[str, list[str]] = {}
    if args.calib_texts:
        quotas = {k: calib_quota.get(k, 0) * needs_calib + check_quota.get(k, 0) for k in set(calib_quota) | set(check_quota)}
        texts_by_kind = sample_texts(args.calib_texts, quotas, seed=args.seed, field=args.text_field)
    elif needs_calib:
        raise SystemExit("static / mixed need --calib-texts")
    check_texts = [t for k, pool in sorted(texts_by_kind.items()) for t in pool[: check_quota.get(k, 0)]]
    calib_texts = [t for k, pool in sorted(texts_by_kind.items()) for t in pool[check_quota.get(k, 0):][: calib_quota.get(k, 0)]]
    report["texts"] = {"source": args.calib_texts, "check": len(check_texts), "calibration": len(calib_texts) if needs_calib else 0}

    fp32_dir = out / VARIANT_NAMES["fp32"]
    if (fp32_dir / META_FILE).is_file() and not args.overwrite:
        LOG.info("fp32 export exists in %s; reusing it (--overwrite to redo)", fp32_dir)
        meta = read_meta(fp32_dir)
    else:
        LOG.info("exporting %s -> %s (graph %s, opset %d)", model_dir, fp32_dir, args.graph, args.opset)
        meta = export_fp32(model_dir, fp32_dir, graph=args.graph, opset=args.opset, max_length=args.max_length,
                           check_texts=check_texts[: args.torch_check_n])
    report["variants"]["fp32"] = {"dir": str(fp32_dir), "bytes": (fp32_dir / MODEL_FILE).stat().st_size, "torch_check": meta.get("torch_check")}
    if variants == ["fp32"]:
        shutil.rmtree(work, ignore_errors=True)
        return report

    ref = OnnxSplade(fp32_dir, threads=args.threads).encode(check_texts, batch_size=8) if check_texts else None
    pre = preprocess(fp32_dir / MODEL_FILE, work / "model.pre.onnx")
    nodes = quantizable_nodes(pre, args.quantize_embeddings)
    blocks = block_order(n["block"] for n in nodes)
    report["blocks"] = {b: sum(1 for n in nodes if n["block"] == b) for b in blocks}
    LOG.info("quantizable nodes: %d in blocks %s", len(nodes), report["blocks"])
    if "other" in report["blocks"]:
        LOG.warning("%d nodes not mapped to a block (names like %s)", report["blocks"]["other"],
                    next(n["name"] for n in nodes if n["block"] == "other"))

    def finish(kind: str, model_file: Path, quant: dict[str, Any], extra_meta: dict[str, Any] | None = None) -> None:
        vdir = out / VARIANT_NAMES[kind]
        copy_variant_files(fp32_dir, vdir)
        shutil.move(str(model_file), vdir / MODEL_FILE)
        vmeta = dict(meta, variant=VARIANT_NAMES[kind], quantization=quant, created=_dt.datetime.now().isoformat(timespec="seconds"))
        vmeta.pop("torch_check", None)
        vmeta.update(extra_meta or {})
        write_meta(vdir, vmeta)
        if ref is not None:
            vmeta["fp32_check"] = check_variant(vdir, check_texts, ref, args.threads)
            LOG.info("%s vs fp32: cosine %.5f, active Jaccard %.4f, dims ratio %.3f", VARIANT_NAMES[kind],
                     vmeta["fp32_check"]["cosine"]["mean"], vmeta["fp32_check"]["active_jaccard"]["mean"], vmeta["fp32_check"]["dims_ratio"])
        write_meta(vdir, vmeta)
        report["variants"][kind] = {"dir": str(vdir), "bytes": (vdir / MODEL_FILE).stat().st_size,
                                    "quantization": quant, "fp32_check": vmeta.get("fp32_check")}
        gc.collect()

    dyn_kwargs = dict(per_channel=not args.per_tensor, reduce_range=args.reduce_range, weight_type=args.weight_type,
                      quantize_embeddings=args.quantize_embeddings)
    if "dynamic" in variants:
        tmp = work / "dynamic.onnx"
        quant = quantize_dynamic_int8(pre, tmp, **dyn_kwargs)
        finish("dynamic", tmp, quant)

    if needs_calib:
        cache = out / "calibration" / f"ranges-{args.calib_method}.pkl"
        if cache.is_file() and not args.overwrite:
            with cache.open("rb") as handle:
                ranges = pickle.load(handle)
            LOG.info("activation ranges from %s", cache)
        else:
            batches = calibration_batches(tokenizer, calib_texts, args.calib_batch_size, args.calib_max_length)
            ranges = calibrate(pre, batches, work, method=args.calib_method, percentile=args.calib_percentile,
                               chunk=args.calib_chunk, threads=args.threads, quantize_embeddings=args.quantize_embeddings)
            del batches
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as handle:
                pickle.dump(ranges, handle)
            (cache.parent / "calibration.json").write_text(json.dumps({
                "method": args.calib_method, "texts": args.calib_texts, "quota": calib_quota, "n": len(calib_texts),
                "batch_size": args.calib_batch_size, "max_length": args.calib_max_length, "seed": args.seed,
                "percentile": args.calib_percentile if args.calib_method == "percentile" else None,
            }, indent=2) + "\n")
        static_kwargs = dict(per_channel=not args.per_tensor, reduce_range=args.reduce_range, activation_type=args.activation_type,
                             weight_type=args.weight_type, quantize_embeddings=args.quantize_embeddings,
                             quantize_activation_matmuls=args.quantize_attention_matmuls)
        calib_info = {"calibration": {"method": args.calib_method, "n_texts": len(calib_texts), "max_length": args.calib_max_length}}

        if "static" in variants:
            tmp = work / "static.onnx"
            quant = quantize_static_int8(pre, tmp, ranges, **static_kwargs)
            finish("static", tmp, quant, calib_info)


    if "mixed" in variants:
        forced = [b.strip() for b in args.mixed_fp32.split(",") if b.strip()]
        missing = [b for b in forced if b not in blocks]
        if missing:
            LOG.warning("mixed: blocks %s not in the graph (have %s)", missing, blocks)
        auto: list[str] = []
        sensitivity = None
        if args.mixed_auto_k > 0:
            probe = [b for b in blocks if b not in forced]
            n_sens = min(args.sensitivity_n, len(check_texts))
            if n_sens == 0:
                raise SystemExit("--mixed-auto-k needs check texts (--calib-texts and --check-n)")

            def quantize_one(dst: Path, exclude: list[str]) -> None:
                if args.sensitivity_method == "dynamic" or args.mixed_method == "dynamic":
                    quantize_dynamic_int8(pre, dst, nodes_to_exclude=exclude, **dyn_kwargs)
                else:
                    quantize_static_int8(pre, dst, ranges, nodes_to_exclude=exclude, **static_kwargs)

            sensitivity = block_sensitivity(fp32_dir, nodes, probe, check_texts[:n_sens], ref[:n_sens], work,
                                            quantize_one, threads=args.threads)
            auto = [r["block"] for r in sensitivity[: args.mixed_auto_k]]
        fp32_blocks = block_order(forced + auto)
        exclude = nodes_in_blocks(nodes, fp32_blocks)
        tmp = work / "mixed.onnx"
        if args.mixed_method == "dynamic":
            quant = quantize_dynamic_int8(pre, tmp, nodes_to_exclude=exclude, **dyn_kwargs)
            info: dict[str, Any] = {}
        else:
            quant = quantize_static_int8(pre, tmp, ranges, nodes_to_exclude=exclude, **static_kwargs)
            info = dict(calib_info)
        quant.update(method=f"mixed ({args.mixed_method} int8 + fp32 blocks)", fp32_blocks=fp32_blocks, fp32_blocks_forced=forced,
                     fp32_blocks_auto=auto, fp32_nodes=exclude)
        finish("mixed", tmp, quant, dict(info, sensitivity=sensitivity))
        report["sensitivity"] = sensitivity

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint directory (AutoModelForMaskedLM + tokenizer)")
    p.add_argument("--out", required=True, help="output root; one subdirectory per variant")
    p.add_argument("--variants", default="fp32,dynamic,static,mixed", help="comma list of fp32, dynamic, static, mixed")
    p.add_argument("--graph", choices=["splade", "mlm"], default="splade",
                   help="splade: pooled sparse vector out of the graph (default); mlm: logits, the sentence-transformers/optimum layout")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--max-length", type=int, default=None, help="recorded in the metadata (default: the checkpoint's max_seq_length)")
    p.add_argument("--threads", type=int, default=4, help="torch and onnxruntime intra-op threads (leave cores to the queue)")
    p.add_argument("--overwrite", action="store_true", help="redo the fp32 export and the calibration even if cached")
    p.add_argument("--keep-work", action="store_true", help="keep out/_work (pre-processed graph)")
    # quantization
    p.add_argument("--per-tensor", action="store_true", help="per-tensor weight scales instead of per channel")
    p.add_argument("--reduce-range", action="store_true",
                   help="7-bit weights: avoids u8*s8 int16 saturation on AVX2/AVX-512 CPUs without VNNI (e.g. Zen 3)")
    p.add_argument("--weight-type", choices=["s8", "u8"], default="s8")
    p.add_argument("--activation-type", choices=["s8", "u8"], default="s8", help="static / mixed activations")
    p.add_argument("--quantize-embeddings", action="store_true", help="also int8 the embedding Gather (weight-only)")
    p.add_argument("--quantize-attention-matmuls", action="store_true",
                   help="static / mixed: also int8 the activation x activation MatMuls (Q K^T, softmax V); default fp32")
    # texts
    p.add_argument("--calib-texts", default=None, help="jsonl (text, kind) or .txt; e.g. datasets/kd/kd_v1.jsonl")
    p.add_argument("--text-field", default="text")
    p.add_argument("--calib-n", default="query=256,passage=256", help="calibration quota per kind")
    p.add_argument("--check-n", default="query=64,passage=64", help="held-out texts for the fp32 checks and the sensitivity probe")
    p.add_argument("--calib-method", choices=["minmax", "percentile", "entropy"], default="minmax")
    p.add_argument("--calib-percentile", type=float, default=99.999)
    p.add_argument("--calib-batch-size", type=int, default=4)
    p.add_argument("--calib-chunk", type=int, default=4, help="batches handed to the calibrator at once (RAM)")
    p.add_argument("--calib-max-length", type=int, default=256)
    p.add_argument("--torch-check-n", type=int, default=16, help="texts compared against the torch model after export")
    p.add_argument("--seed", type=int, default=13)
    # mixed
    p.add_argument("--mixed-fp32", default="head", help="blocks kept fp32 in the mixed variant (head, embeddings, layer.N)")
    p.add_argument("--mixed-method", choices=["static", "dynamic"], default="static",
                   help="int8 scheme of the mixed variant's quantized blocks (dynamic needs no calibration)")
    p.add_argument("--mixed-auto-k", type=int, default=0, help="additionally keep the k most sensitive blocks fp32")
    p.add_argument("--sensitivity-method", choices=["static", "dynamic"], default="static")
    p.add_argument("--sensitivity-n", type=int, default=64)
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # the GPU belongs to the training queue
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    LOG.setLevel(logging.INFO)  # onnxruntime's quantizer logs every tensor at INFO on the root logger
    import torch

    torch.set_num_threads(max(1, args.threads))
    report = run(args)
    out = Path(args.out) / "export_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    for kind, info in report["variants"].items():
        chk = info.get("fp32_check") or {}
        cos = chk.get("cosine", {}).get("mean")
        LOG.info("%-13s %7.1f MB%s", VARIANT_NAMES[kind], info["bytes"] / 2**20, f"  cosine {cos:.5f}" if cos is not None else "")
    LOG.info("report: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
