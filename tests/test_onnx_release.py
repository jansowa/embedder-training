"""ONNX export of SPLADE models (release.onnx_splade), the numpy encoder and the CPU benchmark, on a tiny RoBERTa."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from distill_fixtures import make_teacher, write_corpus  # noqa: E402
from release.fidelity import compare_vectors, retrieval_agreement  # noqa: E402
from release.onnx_splade import block_order, node_block  # noqa: E402
from release.texts import parse_quotas, sample_texts  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name,block",
    [
        ("/inner/roberta/encoder/layer.3/attention/self/query/MatMul", "layer.3"),
        ("/inner/roberta/encoder/layer.11/output/dense/MatMul", "layer.11"),
        ("/inner/lm_head/dense/MatMul", "head"),
        ("/inner/lm_head/decoder/MatMul", "head"),
        ("/bert/encoder/layer.0/intermediate/dense/MatMul", "layer.0"),
        ("/cls/predictions/transform/dense/MatMul", "head"),
        ("/albert/encoder/albert_layer_groups.0/albert_layers.0/ffn/MatMul", "layer.0"),
        ("/albert/encoder/embedding_hidden_mapping_in/MatMul", "embeddings"),
        ("/predictions/decoder/MatMul", "head"),
        ("/model/layers.7/attn/Wqkv/MatMul", "layer.7"),
        ("/head/dense/MatMul", "head"),
        ("/decoder/MatMul", "head"),
        ("/roberta/embeddings/word_embeddings/Gather", "embeddings"),
        ("/something/else/MatMul", "other"),
        ("/roberta/encoder/layer_norm_extra/MatMul", "other"),
    ],
)
def test_node_blocks(name, block):
    assert node_block(name) == block


def test_block_order():
    assert block_order(["head", "layer.10", "layer.2", "embeddings", "layer.2"]) == ["embeddings", "layer.2", "layer.10", "head"]


def test_quotas_and_sampling(tmp_path):
    assert parse_quotas("query=3,passage=5") == {"query": 3, "passage": 5}
    assert parse_quotas("7") == {"all": 7}
    corpus = tmp_path / "c.jsonl"
    write_corpus(corpus, n=40)
    got = sample_texts(corpus, {"query": 4, "passage": 6, "all": 9}, seed=1)
    assert [len(got[k]) for k in ("query", "passage", "all")] == [4, 6, 9]
    assert got == sample_texts(corpus, {"query": 4, "passage": 6, "all": 9}, seed=1)


def test_fidelity_of_identical_vectors():
    rng = np.random.default_rng(0)
    vec = np.maximum(rng.normal(size=(6, 30)), 0).astype(np.float32)
    stats = compare_vectors(vec, vec)
    assert stats["cosine"]["mean"] == pytest.approx(1.0)
    assert stats["active_jaccard"]["mean"] == 1.0 and stats["dims_ratio"] == 1.0
    agree = retrieval_agreement(vec[:2], vec[2:], vec[:2], vec[2:], top_k=3)
    assert agree["symmetric"]["mean"] == 1.0 and agree["query_only"]["mean"] == 1.0


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    import onnx  # noqa: F401

    from release.onnx_splade import main

    root = tmp_path_factory.mktemp("onnx_release")
    make_teacher(root / "teacher", layers=2, hidden=32, intermediate=48)
    write_corpus(root / "corpus.jsonl", n=80)
    main([
        "--model", str(root / "teacher"), "--out", str(root / "onnx"), "--calib-texts", str(root / "corpus.jsonl"),
        "--calib-n", "query=8,passage=24", "--check-n", "query=8,passage=16", "--threads", "1", "--mixed-auto-k", "1",
    ])
    return root


def test_export_writes_four_self_contained_variants(exported):
    from release.onnx_encoder import read_meta

    out = exported / "onnx"
    for variant in ("fp32", "dynamic-int8", "static-int8", "mixed-int8"):
        for name in ("model.onnx", "onnx_splade.json", "tokenizer.json", "config.json"):
            assert (out / variant / name).is_file(), (variant, name)
        assert read_meta(out / variant)["variant"] == variant
    meta = read_meta(out / "fp32")
    assert meta["graph"] == "splade" and meta["output_dim"] == 16 and meta["max_length"] == 32
    assert meta["torch_check"]["max_abs_diff"] < 1e-4
    report = json.loads((out / "export_report.json").read_text())
    assert set(report["variants"]) == {"fp32", "dynamic", "static", "mixed"}
    assert report["blocks"] == {"layer.0": 6, "layer.1": 6, "head": 2}
    assert (out / "calibration" / "ranges-minmax.pkl").is_file()
    assert not (out / "_work").exists()
    for kind in ("dynamic", "static", "mixed"):
        assert report["variants"][kind]["fp32_check"]["cosine"]["mean"] > 0.95


def test_mixed_keeps_the_head_and_the_most_sensitive_block_in_fp32(exported):
    import onnx

    from release.onnx_encoder import read_meta

    out = exported / "onnx"
    quant = read_meta(out / "mixed-int8")["quantization"]
    assert quant["fp32_blocks_forced"] == ["head"] and len(quant["fp32_blocks_auto"]) == 1
    assert quant["fp32_blocks"][-1] == "head"
    sens = read_meta(out / "mixed-int8")["sensitivity"]
    assert {s["block"] for s in sens} == {"layer.0", "layer.1"}

    def int8_weight_nodes(path: Path) -> set[str]:
        graph = onnx.load(str(path)).graph
        dq_out = {o for n in graph.node if n.op_type == "DequantizeLinear" for o in n.output}
        return {n.name for n in graph.node if n.op_type == "MatMul" and n.input[1] in dq_out}

    def quantized_activation_matmuls(path: Path) -> set[str]:
        graph = onnx.load(str(path)).graph
        inits = {i.name for i in graph.initializer}
        producer = {o: n for n in graph.node for o in n.output}
        weight_of = lambda n: producer[n.input[1]].input[0] if n.input[1] in producer and producer[n.input[1]].op_type == "DequantizeLinear" else n.input[1]
        return {n.name for n in graph.node if n.op_type == "MatMul" and weight_of(n) not in inits
                and any(i in producer and producer[i].op_type == "DequantizeLinear" for i in n.input)}

    static_w, mixed_w = int8_weight_nodes(out / "static-int8" / "model.onnx"), int8_weight_nodes(out / "mixed-int8" / "model.onnx")
    assert any(node_block(n) == "head" for n in static_w) and len(static_w) == 14
    assert not any(node_block(n) in quant["fp32_blocks"] for n in mixed_w)
    assert len(mixed_w) == 6  # the remaining layer is still int8
    for variant in ("static-int8", "mixed-int8"):  # attention products stay fp32 (ORT's QDQ path ignores MatMulConstBOnly)
        assert not quantized_activation_matmuls(out / variant / "model.onnx")
    dyn = onnx.load(str(out / "dynamic-int8" / "model.onnx")).graph
    assert sum(n.op_type == "MatMulInteger" for n in dyn.node) == 14


def test_dynamic_mixed_keeps_the_head_fp32_without_calibration(exported, tmp_path):
    import onnx

    from release.onnx_encoder import read_meta
    from release.onnx_splade import main

    out = tmp_path / "onnx"
    main([
        "--model", str(exported / "teacher"), "--out", str(out), "--variants", "fp32,mixed", "--mixed-method", "dynamic",
        "--weight-type", "u8", "--calib-texts", str(exported / "corpus.jsonl"), "--calib-n", "query=8,passage=24",
        "--check-n", "query=8,passage=16", "--threads", "1",
    ])
    assert not (out / "calibration").exists()
    meta = read_meta(out / "mixed-int8")
    assert meta["quantization"]["method"] == "mixed (dynamic int8 + fp32 blocks)" and meta["quantization"]["fp32_blocks"] == ["head"]
    assert meta["fp32_check"]["cosine"]["mean"] > 0.95
    graph = onnx.load(str(out / "mixed-int8" / "model.onnx")).graph
    integer = [n.name for n in graph.node if n.op_type == "MatMulInteger"]
    assert len(integer) == 12 and not any(node_block(n) == "head" for n in integer)


def test_encoder_slicing_does_not_change_vectors(exported):
    from release.onnx_encoder import OnnxSplade

    texts = [json.loads(line)["text"] for line in (exported / "corpus.jsonl").open()][:12]
    whole = OnnxSplade(exported / "onnx" / "fp32", threads=1).encode(texts, batch_size=12)
    sliced = OnnxSplade(exported / "onnx" / "fp32", threads=1, max_batch_tokens=5).encode(texts, batch_size=12)
    np.testing.assert_allclose(whole, sliced, atol=1e-6)
    hf = OnnxSplade(exported / "onnx" / "fp32", threads=1, tokenizer_backend="transformers").encode(texts, batch_size=4)
    np.testing.assert_allclose(whole, hf, atol=1e-6)


def test_mlm_graph_pools_to_the_same_vectors(exported, tmp_path):
    from release.onnx_encoder import OnnxSplade
    from release.onnx_splade import export_fp32

    meta = export_fp32(exported / "teacher", tmp_path / "mlm", graph="mlm", check_texts=["ala ma kota", "pies"])
    assert meta["output"] == "logits" and meta["torch_check"]["max_abs_diff"] < 1e-4
    texts = ["ala ma kota i psy", "dom"]
    a = OnnxSplade(exported / "onnx" / "fp32", threads=1).encode(texts)
    b = OnnxSplade(tmp_path / "mlm", threads=1).encode(texts)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_benchmark_runs_each_variant_in_a_worker(exported, tmp_path):
    from release.onnx_bench import main

    out = tmp_path / "bench.json"
    variants = [str(exported / "onnx" / v) for v in ("fp32", "dynamic-int8", "mixed-int8")]
    main(["--variants", *variants, "--texts", str(exported / "corpus.jsonl"), "--n", "query=10,passage=20",
          "--threads", "1", "--latency-n", "5", "--warmup", "1", "--min-free-gb", "0", "--out", str(out)])
    report = json.loads(out.read_text())
    labels = [r["label"] for r in report["results"]]
    assert labels == ["fp32", "dynamic-int8", "mixed-int8"]
    assert report["results"][0]["reference"] and "fidelity" in report["results"][1]
    assert report["results"][2]["fidelity"]["query"]["cosine"]["mean"] > 0.95
    assert report["results"][1]["timings"]["1"]["passages_per_s"] > 0
    assert out.with_suffix(".md").read_text().startswith("| variant")


def test_encoder_import_does_not_pull_torch():
    code = "import sys, release.onnx_encoder, release.fidelity, release.texts; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
