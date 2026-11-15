"""release.gpu_quant / release.gpu_bench without a GPU: variant grammar, head selection, postings, report table."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from release.fidelity import postings_per_query
from release.gpu_bench import markdown
from release.gpu_quant import GpuQuantError, is_head, linear_filter, parse_variant

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "spec,scheme,head,compile_mode,dtype,label",
    [
        ("fp16", "fp16", True, "default", "fp16", "fp16:compile"),
        ("fp16:eager", "fp16", True, "none", "fp16", "fp16:eager"),
        ("fp32:eager", "fp32", True, "none", "fp32", "fp32:eager"),
        ("int8-dyn", "int8-dyn", True, "default", "bf16", "int8-dyn:compile"),
        ("int8-dyn:nohead,eager", "int8-dyn", False, "none", "bf16", "int8-dyn:nohead,eager"),
        ("int8-dyn:fp16,max-autotune", "int8-dyn", True, "max-autotune", "fp16", "int8-dyn:fp16,max-autotune"),
        ("fp8-dyn:tensor,nohead", "fp8-dyn", False, "default", "bf16", "fp8-dyn:nohead,tensor,compile"),
    ],
)
def test_parse_variant(spec, scheme, head, compile_mode, dtype, label):
    v = parse_variant(spec)
    assert (v.scheme, v.head, v.compile_mode, v.base_dtype, v.label) == (scheme, head, compile_mode, dtype, label)
    assert parse_variant(v.label) == v  # labels round-trip, the parent hands them to the workers


@pytest.mark.parametrize(
    "spec,fast_head,pool_chunk,scope,label",
    [
        ("fp16:fasthead", 0, None, "body", "fp16:fasthead,body,compile"),
        ("fp16:fasthead64,eager", 64, None, "model", "fp16:fasthead64,eager"),
        ("fp16:poolchunk32", None, 32, "model", "fp16:poolchunk32,compile"),
        ("fp16:body", None, None, "body", "fp16:body,compile"),
        ("fp16:body,eager", None, None, "model", "fp16:eager"),
        ("int8-dyn:nohead,fasthead", 0, None, "body", "int8-dyn:fasthead,body,nohead,compile"),
    ],
)
def test_parse_head_options(spec, fast_head, pool_chunk, scope, label):
    v = parse_variant(spec)
    assert (v.fast_head, v.pool_chunk, v.compile_scope, v.label) == (fast_head, pool_chunk, scope, label)
    assert parse_variant(v.label) == v


@pytest.mark.parametrize("spec", ["fp16:fasthead,poolchunk8", "fp16:fastheadx", "fp16:poolchunk0", "fp16:poolchunk"])
def test_parse_head_options_reject(spec):
    with pytest.raises(GpuQuantError):
        parse_variant(spec)


def test_default_variants_parse():
    from release.gpu_bench import DEFAULT_VARIANTS

    labels = [parse_variant(v).label for v in DEFAULT_VARIANTS]
    assert len(set(labels)) == len(labels)


def test_parse_variant_default_compile():
    assert parse_variant("bf16", "none").compile_mode == "none"
    assert parse_variant("bf16:compile", "none").compile_mode == "default"


@pytest.mark.parametrize("spec", ["int4", "fp16:nohead", "fp16:bf16", "int8-dyn:tensor", "int8-dyn:sparse"])
def test_parse_variant_rejects(spec):
    with pytest.raises(GpuQuantError):
        parse_variant(spec)


@pytest.mark.parametrize(
    "fqn,head",
    [
        ("lm_head.decoder", True),
        ("lm_head.dense", True),
        ("cls.predictions.transform.dense", True),
        ("predictions.decoder", True),
        ("head.dense", True),
        ("decoder", True),
        ("roberta.encoder.layer.3.attention.self.query", False),
        ("model.layers.0.attn.Wqkv", False),
        ("albert.encoder.embedding_hidden_mapping_in", False),
    ],
)
def test_is_head(fqn, head):
    assert is_head(fqn) is head


def test_linear_filter_on_tiny_roberta():
    torch = pytest.importorskip("torch")
    from transformers import RobertaConfig, RobertaForMaskedLM

    model = RobertaForMaskedLM(RobertaConfig(vocab_size=30, hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
                                             intermediate_size=12, max_position_embeddings=20))
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    with_head = [n for n, m in model.named_modules() if linear_filter(True)(m, n)]
    no_head = [n for n, m in model.named_modules() if linear_filter(False)(m, n)]
    assert with_head == linears
    assert set(linears) - set(no_head) == {"lm_head.dense", "lm_head.decoder"}
    assert len(no_head) == 2 * 6  # q, k, v, attention output, intermediate, output per layer


def test_postings_per_query():
    q = np.array([[0.5, 0.0, 0.2], [0.0, 0.004, 0.0]], dtype=np.float32)  # the second query indexes nothing
    p = np.array([[0.1, 0.1, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.9]], dtype=np.float32)
    got = postings_per_query(q, p)
    assert got["per_query"]["mean"] == pytest.approx((2 + 1 + 0) / 2)  # dim 0 in 2 passages, dim 2 in 1
    assert got["per_query_per_1k_passages"] == pytest.approx(1.5 * 1000 / 3)
    assert got["n_passages"] == 3


def _result(label, speed, fid=None):
    r = {"label": label, "warmup_seconds": 5.0, "cuda_peak_mb": 900.0,
         "timings": {"query_ms_p50": 4.0, "queries_per_s": 800.0, "passage": {"32": {"passages_per_s": speed / 2},
                                                                              "128": {"passages_per_s": speed}}},
         "dims": {"query": 20.0, "passage": 150.0}, "postings": {"per_query": {"mean": 300.0}}}
    if fid:
        s = {"mean": 0.99}
        r["fidelity"] = {"query": {"cosine": s, "dims": {"mean": 21.0}}, "passage": {"cosine": s, "active_jaccard": s, "dims": {"mean": 151.0}},
                         "retrieval": {"symmetric": s, "query_only": s}, "postings": {"symmetric": {"per_query": {"mean": 310.0}}}}
    else:
        r["reference"] = True
    return r


def test_markdown_speedup_and_failure():
    report = {"batches": {"query": 64, "passage": [32, 128]}, "environment": {"gpu": "X", "capability": "12.0"},
              "results": [_result("fp16:eager", 1000.0), _result("int8-dyn:compile", 1500.0, fid=True),
                          {"label": "fp8-dyn:compile", "error": "exit 1: no kernel"}]}
    table = markdown(report)
    lines = table.splitlines()
    assert "1.00x" in lines[2] and "ref" in lines[2]
    assert "1.50x" in lines[3] and "0.9900" in lines[3]
    assert "failed: exit 1: no kernel" in lines[4]
    assert len({line.count("|") for line in lines[:5]}) == 1  # every row has all the columns


def test_imports_stay_light():
    code = "import sys, release.gpu_quant, release.gpu_bench; print('torch' in sys.modules, 'torchao' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True)
    assert out.stdout.strip() == "False False"
