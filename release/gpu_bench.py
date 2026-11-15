#!/usr/bin/env python
"""GPU benchmark of a SPLADE ``SparseEncoder`` in fp32 / fp16 / bf16 and with torchao 8-bit kernels.

Every variant (``release.gpu_quant.parse_variant``: ``fp16``, ``int8-dyn:nohead``, ``fp8-dyn:tensor,eager`` ...)
runs in its own subprocess, one after another, through the plain Sentence Transformers API
(``encode_query`` / ``encode_document``, which sorts the texts by length itself). Per variant:

  warm-up          seconds of the first passes (torch.compile happens here; users pay it once per process)
  query latency    ``encode_query`` of one query, ms per call: p50 / p90 / p99
  query speed      queries/s at ``--query-batch``
  passage speed    texts/s and tokens/s for every ``--passage-batch``, the median of ``--repeats`` passes
  memory           peak CUDA memory allocated during the timed passes (MB)

and against the reference variant (the first one): cosine, relative L1, Jaccard of the indexed dimensions
(``round(w * 100) > 0``), dims per query and per passage, the in-sample postings per query, top-10 term
overlap, and the in-sample top-10 retrieval agreement (symmetric and query-only), all from release.fidelity.

A variant that fails (no torchao, a kernel missing for the card, a compile error) is recorded with its error and
the run goes on. Retrieval quality on PIRB is a separate step. Only meaningful on an idle GPU: the parent
refuses to start a worker while another process holds the card, unless --allow-busy-gpu.

Usage (5070 Ti, after installing torchao into the venv that runs it):
  .venv/bin/python -m release.gpu_bench --model models/soup-v20-dqf-bias015 \
      --variants fp32:eager fp16:eager fp16 bf16 int8-dyn int8-dyn:nohead fp8-dyn fp8-dyn:nohead \
      --out experiments/polish-splade-2026-10/results/gpu-quant/soup-v20-dqf-bias015_5070ti.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from release.fidelity import compare_vectors, postings_per_query, retrieval_agreement
from release.gpu_quant import COMPILE_MODES, GpuQuantError, parse_variant
from release.onnx_bench import gpu_compute_processes, load_vectors, mem_available_gb, pct
from release.texts import parse_quotas, sample_texts

LOG_PREFIX = "[gpu_bench]"
# The first measurement on the 3060 Ti (docs/gpu-quant.md, plan of 15.11): baselines, the head, then torchao int8.
DEFAULT_VARIANTS = ["fp16:eager", "fp16", "fp16:poolchunk64,eager", "fp16:fasthead,eager", "fp16:fasthead",
                    "fp16:fasthead64", "int8-dyn", "int8-dyn:nohead", "int8-dyn:nohead,fasthead"]


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------- worker


def _dense(embeddings: Any) -> np.ndarray:
    return embeddings.to_dense().float().cpu().numpy() if embeddings.is_sparse else embeddings.float().cpu().numpy()


def dynamo_counters() -> dict[str, int]:
    """Graphs compiled, frames captured and graph breaks so far in this process (torch._dynamo counters)."""
    try:
        from torch._dynamo.utils import counters
    except ImportError:  # pragma: no cover
        return {}
    stats = counters.get("stats", {})
    return {"unique_graphs": int(stats.get("unique_graphs", 0)), "calls_captured": int(stats.get("calls_captured", 0)),
            "graph_breaks": int(sum(counters.get("graph_break", {}).values()))}


def worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Runs in a subprocess: load + quantize + compile, warm up, encode the check texts, time."""
    import torch

    from release.gpu_quant import load_sparse_encoder

    variant = parse_variant(spec["variant"], spec["compile"])
    device = spec.get("device", "cuda")
    on_gpu = torch.device(device).type == "cuda"

    def sync() -> None:
        if on_gpu:
            torch.cuda.synchronize()

    texts = json.loads(Path(spec["texts"]).read_text(encoding="utf-8"))
    queries, passages = texts["query"], texts["passage"]
    nq_check, np_check = spec["check"]["query"], spec["check"]["passage"]
    result: dict[str, Any] = {"label": variant.label}

    t0 = time.time()
    model, info = load_sparse_encoder(spec["model"], variant, device=device, max_seq_length=spec.get("max_length"))
    sync()
    result["load_seconds"] = round(time.time() - t0, 2)
    result.update(info)
    result["max_seq_length"] = model.max_seq_length
    to_cpu = spec["save_to_cpu"]

    def enc_q(batch: list[str], bs: int):
        return model.encode_query(batch, batch_size=bs, save_to_cpu=to_cpu, show_progress_bar=False)

    def enc_p(batch: list[str], bs: int):
        return model.encode_document(batch, batch_size=bs, save_to_cpu=to_cpu, show_progress_bar=False)

    # Warm-up: exactly the calls that are timed later, once each (every batch size including the last partial batch,
    # every padded length, the single-query path). Shorter warm-ups left recompilations inside the timed passes
    # (the "graphs (+timed)" column; seen in the CPU smoke of 15.11). Costs one extra pass per batch size.
    t0 = time.time()
    for bs in spec["passage_batches"]:
        enc_p(passages, bs)
    for q in queries[: spec["latency_n"]]:
        enc_q([q], 1)
    enc_q(queries, spec["query_batch"])
    sync()
    result["warmup_seconds"] = round(time.time() - t0, 2)

    # Vectors of the check texts (dense, saved sparse).
    with torch.inference_mode():
        q_vec = _dense(model.encode_query(queries[:nq_check], batch_size=spec["query_batch"], convert_to_sparse_tensor=False,
                                          save_to_cpu=True, show_progress_bar=False))
        p_vec = _dense(model.encode_document(passages[:np_check], batch_size=spec["passage_batches"][0],
                                             convert_to_sparse_tensor=False, save_to_cpu=True, show_progress_bar=False))
    vec = np.concatenate([q_vec, p_vec])
    r, c = np.nonzero(vec)
    np.savez(spec["vectors"], rows=r, cols=c, vals=vec[r, c], shape=np.array(vec.shape))
    del vec, q_vec, p_vec

    tok = model.tokenizer
    lens = [len(x) for x in tok(passages, truncation=True, max_length=model.max_seq_length)["input_ids"]]
    p_tokens = int(sum(lens))
    result["mean_tokens"] = {
        "query": float(np.mean([len(x) for x in tok(queries, truncation=True, max_length=model.max_seq_length)["input_ids"]])),
        "passage": p_tokens / max(1, len(passages)),
    }

    # Snapshot after the warm-up and the agreement encodes, right before the timed passes.
    result["compile_after_warmup"] = dynamo_counters()
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    timings: dict[str, Any] = {}
    q_ms = []
    for q in queries[: spec["latency_n"]]:
        s = time.perf_counter()
        enc_q([q], 1)
        sync()
        q_ms.append((time.perf_counter() - s) * 1000)
    timings["query_ms_p50"], timings["query_ms_p90"], timings["query_ms_p99"] = pct(q_ms, 50), pct(q_ms, 90), pct(q_ms, 99)
    s = time.perf_counter()
    enc_q(queries, spec["query_batch"])
    sync()
    timings["queries_per_s"] = len(queries) / max(1e-9, time.perf_counter() - s)
    passage = {}
    for bs in spec["passage_batches"]:
        runs = []
        for _ in range(spec["repeats"]):
            s = time.perf_counter()
            enc_p(passages, bs)
            sync()
            runs.append(time.perf_counter() - s)
        sec = float(np.median(runs))
        passage[str(bs)] = {"passages_per_s": len(passages) / sec, "tokens_per_s": p_tokens / sec,
                            "seconds_median": sec, "seconds_all": runs}
    timings["passage"] = passage
    result["timings"] = timings
    result["cuda_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20 if on_gpu else float("nan")
    # Graphs compiled during the timed passes: anything above the warm-up count means the timings include
    # recompilation (a length or batch size the warm-up did not cover).
    result["compile_after_timing"] = dynamo_counters()
    best = max(passage.values(), key=lambda x: x["passages_per_s"])
    log(f"{variant.label:>22}: query p50 {timings['query_ms_p50']:.1f} ms, best {best['passages_per_s']:.0f} passages/s, "
        f"warm-up {result['warmup_seconds']:.0f} s")
    return result


# --------------------------------------------------------------------------------------------- parent


def _worker_env(compile_threads: int) -> dict[str, str]:
    repo = str(Path(__file__).resolve().parents[1])
    path = os.environ.get("PYTHONPATH", "")
    env = dict(os.environ, PYTHONPATH=f"{repo}{os.pathsep}{path}" if path else repo)
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(compile_threads))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_worker(spec: dict[str, Any], spec_path: Path, compile_threads: int, timeout: float) -> dict[str, Any]:
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    try:
        proc = subprocess.run([sys.executable, "-m", "release.gpu_bench", "--worker", str(spec_path)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              env=_worker_env(compile_threads), timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout:.0f} s"}
    sys.stderr.write(proc.stderr[-4000:] if proc.returncode else "".join(
        line + "\n" for line in proc.stderr.splitlines() if line.startswith(LOG_PREFIX)))
    if proc.returncode != 0:
        tail = [line.strip() for line in proc.stderr.strip().splitlines() if line.strip()][-1:]
        return {"error": f"exit {proc.returncode}: " + " ".join(tail)}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def fidelity(ref: np.ndarray, vec: np.ndarray, nq: int) -> dict[str, Any]:
    return {
        "query": compare_vectors(ref[:nq], vec[:nq]),
        "passage": compare_vectors(ref[nq:], vec[nq:]),
        "retrieval": retrieval_agreement(ref[:nq], ref[nq:], vec[:nq], vec[nq:]),
        "postings": {
            "symmetric": postings_per_query(vec[:nq], vec[nq:]),
            "query_only": postings_per_query(vec[:nq], ref[nq:]),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    batches = report["batches"]["passage"]
    head = ["variant", "warm-up s", "graphs (+timed)", "q p50 ms", f"q/s (b{report['batches']['query']})"]
    head += [f"passages/s (b{b})" for b in batches] + ["speed-up", "CUDA MB"]
    head += ["cos q", "cos p", "Jacc p", "dims q", "dims p", "postings q", "top10 sym", "top10 q-only"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    ref_speed = None
    for r in report["results"]:
        if "error" in r:
            lines.append("| " + " | ".join([r["label"], "failed: " + r["error"][:160].replace("|", "/")] + [""] * (len(head) - 2)) + " |")
            continue
        tm = r["timings"]
        speeds = [tm["passage"][str(b)]["passages_per_s"] for b in batches]
        if ref_speed is None:
            ref_speed = max(speeds)
        cw, ct = r.get("compile_after_warmup") or {}, r.get("compile_after_timing") or {}
        graphs = f"{cw.get('unique_graphs', 0)} (+{ct.get('unique_graphs', 0) - cw.get('unique_graphs', 0)})" if cw else "-"
        row = [r["label"], f"{r['warmup_seconds']:.0f}", graphs, f"{tm['query_ms_p50']:.1f}", f"{tm['queries_per_s']:.0f}"]
        row += [f"{s:.0f}" for s in speeds] + [f"{max(speeds) / ref_speed:.2f}x", f"{r['cuda_peak_mb']:.0f}"]
        f = r.get("fidelity")
        if f:
            row += [f"{f['query']['cosine']['mean']:.4f}", f"{f['passage']['cosine']['mean']:.4f}",
                    f"{f['passage']['active_jaccard']['mean']:.3f}", f"{f['query']['dims']['mean']:.1f}",
                    f"{f['passage']['dims']['mean']:.1f}", f"{f['postings']['symmetric']['per_query']['mean']:.0f}",
                    f"{f['retrieval']['symmetric']['mean']:.3f}", f"{f['retrieval']['query_only']['mean']:.3f}"]
        else:
            row += ["ref", "ref", "ref", f"{r['dims']['query']:.1f}", f"{r['dims']['passage']:.1f}",
                    f"{r['postings']['per_query']['mean']:.0f}", "ref", "ref"]
        lines.append("| " + " | ".join(row) + " |")
    env = report["environment"]
    lines.append("")
    lines.append(f"GPU {env.get('gpu')} (cc {env.get('capability')}), torch {env.get('torch')}, torchao {env.get('torchao')}; "
                 f"speed-up = best passages/s over the reference's best; postings q = in-sample, symmetric.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--model", help="SPLADE checkpoint in the sentence-transformers layout")
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, help="variant specs; the first is the reference")
    p.add_argument("--compile", default="default", choices=COMPILE_MODES, help="torch.compile mode unless a variant says otherwise")
    p.add_argument("--texts", default="datasets/kd/kd_v1.jsonl")
    p.add_argument("--n", default="query=1024,passage=4096", help="texts per kind for timing (query and passage required)")
    p.add_argument("--check", default="query=512,passage=1024", help="the first texts of each kind used for agreement")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--query-batch", type=int, default=64)
    p.add_argument("--passage-batch", default="16,32",
                   help="comma list of encode batch sizes; SpladePooling holds [batch, 512, 50001] logits (fp16: 3.3 GB at "
                        "batch 32, plus temporaries), so 64+ needs a 16 GB card")
    p.add_argument("--latency-n", type=int, default=300, help="single queries timed")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-length", type=int, default=None, help="max_seq_length override (default: the model's)")
    p.add_argument("--keep-on-gpu", action="store_true", help="time without moving the outputs to the CPU")
    p.add_argument("--compile-threads", type=int, default=1, help="TORCHINDUCTOR_COMPILE_THREADS (RAM on the 16 GB box)")
    p.add_argument("--worker-timeout", type=float, default=3600)
    p.add_argument("--min-free-gb", type=float, default=4.0, help="refuse to start a worker with less RAM available")
    p.add_argument("--device", default="cuda", help=argparse.SUPPRESS)  # "cpu": fp32:eager plumbing smoke test only
    p.add_argument("--allow-busy-gpu", action="store_true", help="start workers while other processes use the GPU")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    if args.worker:
        result = worker(json.loads(Path(args.worker).read_text(encoding="utf-8")))
        print(json.dumps(result))
        return 0
    if not args.model:
        p.error("--model is required")
    try:
        variants = [parse_variant(v, args.compile) for v in args.variants]
    except GpuQuantError as exc:
        p.error(str(exc))
    labels = list(dict.fromkeys(v.label for v in variants))
    if len(labels) != len(variants):
        log("duplicate variants dropped")

    quotas, check = parse_quotas(args.n), parse_quotas(args.check)
    if not ({"query", "passage"} <= set(quotas) and {"query", "passage"} <= set(check)):
        p.error("--n and --check need query=... and passage=...")
    texts = sample_texts(args.texts, quotas, seed=args.seed)
    check = {k: min(check[k], len(texts[k])) for k in ("query", "passage")}
    log(f"{len(texts['query'])} queries, {len(texts['passage'])} passages from {args.texts}; agreement on {check}")
    passage_batches = [int(b) for b in args.passage_batch.split(",")]

    results: list[dict[str, Any]] = []
    environment: dict[str, Any] = {"gpu_compute_processes_start": gpu_compute_processes(), "mem_available_gb_start": mem_available_gb()}
    with tempfile.TemporaryDirectory(prefix="gpu_bench.", dir=os.environ.get("GPU_BENCH_TMP")) as tmp:
        tmp_dir = Path(tmp)
        texts_path = tmp_dir / "texts.json"
        texts_path.write_text(json.dumps({"query": texts["query"], "passage": texts["passage"]}, ensure_ascii=False), encoding="utf-8")
        nq = check["query"]
        ref_vec = None
        for i, label in enumerate(labels):
            busy = gpu_compute_processes() if args.device != "cpu" else 0
            if busy and not args.allow_busy_gpu:
                raise SystemExit(f"{busy} process(es) on the GPU; not starting {label} (--allow-busy-gpu to override)")
            avail = mem_available_gb()
            if avail < args.min_free_gb:
                raise SystemExit(f"only {avail:.1f} GB RAM available (< --min-free-gb {args.min_free_gb}); not starting {label}")
            spec = {
                "model": args.model, "variant": label, "compile": args.compile, "texts": str(texts_path), "check": check,
                "query_batch": args.query_batch, "passage_batches": passage_batches, "latency_n": args.latency_n,
                "repeats": args.repeats, "max_length": args.max_length,
                "save_to_cpu": not args.keep_on_gpu, "device": args.device, "vectors": str(tmp_dir / f"vectors_{i}.npz"),
            }
            log(f"worker {label}")
            result = run_worker(spec, tmp_dir / f"spec_{i}.json", args.compile_threads, args.worker_timeout)
            result.setdefault("label", label)
            if "error" in result:
                log(f"{label} failed: {result['error']}")
                results.append(result)
                continue
            for key in ("gpu", "capability", "torch", "torchao"):
                environment.setdefault(key, result.get(key))
            vec = load_vectors(Path(spec["vectors"]))
            result["dims"] = {
                "query": float((np.rint(vec[:nq] * 100) > 0).sum(axis=1).mean()),
                "passage": float((np.rint(vec[nq:] * 100) > 0).sum(axis=1).mean()),
            }
            result["postings"] = postings_per_query(vec[:nq], vec[nq:])
            if ref_vec is None:
                ref_vec = vec
                result["reference"] = True
            else:
                result["fidelity"] = fidelity(ref_vec, vec, nq)
            del vec
            Path(spec["vectors"]).unlink(missing_ok=True)
            results.append(result)

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": args.model,
        "reference": next((r["label"] for r in results if r.get("reference")), None),  # the first variant that ran
        "texts": {"source": args.texts, "seed": args.seed, "query": len(texts["query"]), "passage": len(texts["passage"]), "check": check},
        "batches": {"query": args.query_batch, "passage": passage_batches}, "save_to_cpu": not args.keep_on_gpu,
        "environment": environment, "results": results,
    }
    table = markdown(report)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        out.with_suffix(".md").write_text(table, encoding="utf-8")
        log(f"wrote {out} and {out.with_suffix('.md')}")
    return 0 if any("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
