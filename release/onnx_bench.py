#!/usr/bin/env python
"""CPU benchmark of ONNX SPLADE exports: inference speed and agreement with the fp32 reference.

Every variant (and optionally the PyTorch fp32 model) is measured in its own subprocess, one after another,
so the peak RSS is that variant's alone and nothing stays resident between them. Per variant and per
thread count:

  query latency     batch ``--query-batch`` (default 1), ms per session run: p50 / p90 / p99
  passage speed     batches of ``--passage-batch``, texts/s and tokens/s, ms per batch p50 / p90
  tokenization      ms per query batch (the same for every variant; reported once per run)
  memory            RSS after loading the session and peak RSS of the worker (MB), file size (MB)

and against the reference (the first ``--variants`` entry unless ``--reference``): cosine, relative L1,
Jaccard of the indexed dimensions (``round(w * 100) > 0``), dims per query and per passage, top-10 term
overlap, and an in-sample retrieval check (top-10 passages per query, symmetric and query-only).

Retrieval quality on PIRB is a separate step (``scripts/pirb_regime_eval.sh`` accepts an export directory
as MODEL_DIR or PIRB_QUERY_MODEL; see docs/onnx-release.md). Speed numbers are only meaningful on an idle
machine: the load average and any GPU compute processes at start are written into the result.

Usage:
  .venv/bin/python -m release.onnx_bench --variants models/onnx/n5kj/{fp32,dynamic-int8,static-int8,mixed-int8} \
      --torch-model models/arm-dense-kl-n5kj-ens400 --texts datasets/kd/kd_v1.jsonl --threads 1,6 \
      --out experiments/polish-splade-2026-10/results/onnx/n5kj_cpu.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from release.fidelity import compare_vectors, retrieval_agreement
from release.texts import parse_quotas, sample_texts

LOG_PREFIX = "[onnx_bench]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)


def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("nan")


def rss_mb() -> float:
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return float("nan")


def gpu_compute_processes() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return len([line for line in out.stdout.splitlines() if line.strip()])
    except (OSError, subprocess.SubprocessError):
        return None


def pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


# --------------------------------------------------------------------------------------------- worker


class _TorchSplade:
    """PyTorch fp32 baseline on the CPU with the same tokenization and pooling as the exports."""

    def __init__(self, model_dir: str, threads: int, max_length: int):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
        self.output_dim = min(len(self.tokenizer.get_vocab()), int(self.model.get_output_embeddings().weight.shape[0]))
        self.max_length = max_length

    def tokenize(self, texts):
        enc = self.tokenizer(list(texts), padding="longest", truncation=True, max_length=self.max_length, return_tensors="np")
        return {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}

    def run(self, feeds):
        torch = self.torch
        with torch.inference_mode():
            ids, mask = torch.from_numpy(feeds["input_ids"]), torch.from_numpy(feeds["attention_mask"])
            logits = self.model(input_ids=ids, attention_mask=mask).logits[..., : self.output_dim]
            vec = (torch.log1p(torch.relu(logits)) * mask.unsqueeze(-1)).amax(dim=1)
        return vec.numpy()

    def set_threads(self, threads: int) -> None:
        self.torch.set_num_threads(threads)


def _load(kind: str, path: str, threads: int, max_length: int | None, arena: bool):
    if kind == "torch":
        return _TorchSplade(path, threads, max_length or 512)
    from release.onnx_encoder import OnnxSplade

    return OnnxSplade(path, threads=threads, max_length=max_length, arena=arena, max_batch_tokens=1 << 30)


def worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Runs in a subprocess: encode all texts (vectors to ``spec['vectors']``), then time each thread count."""
    texts = json.loads(Path(spec["texts"]).read_text(encoding="utf-8"))
    queries, passages = texts["query"], texts["passage"]
    result: dict[str, Any] = {"label": spec["label"], "kind": spec["kind"], "path": spec["path"], "rss_start_mb": rss_mb()}
    threads_list = spec["threads"]
    t0 = time.time()
    enc = _load(spec["kind"], spec["path"], max(threads_list), spec.get("max_length"), spec["arena"])
    result["load_seconds"] = round(time.time() - t0, 2)
    result["rss_loaded_mb"] = rss_mb()
    if spec["kind"] != "torch":
        result["file_mb"] = Path(enc.model_path).stat().st_size / 2**20
        result["variant"] = enc.meta.get("variant")
        result["quantization"] = enc.meta.get("quantization")

    # Vectors for the agreement statistics, at the max thread count (sparse on disk: rows, cols, values).
    rows, cols, vals = [], [], []
    tokens = {"query": 0, "passage": 0}
    all_texts = [("query", t) for t in queries] + [("passage", t) for t in passages]
    for start in range(0, len(all_texts), 16):
        chunk = all_texts[start:start + 16]
        feeds = enc.tokenize([t for _, t in chunk])
        for (kind, _), n in zip(chunk, feeds["attention_mask"].sum(axis=1)):
            tokens[kind] += int(n)
        out = enc.run(feeds)
        r, c = np.nonzero(out)
        rows.append(r + start)
        cols.append(c)
        vals.append(out[r, c])
    np.savez(spec["vectors"], rows=np.concatenate(rows), cols=np.concatenate(cols), vals=np.concatenate(vals),
             shape=np.array([len(all_texts), enc.output_dim]))
    result["mean_tokens"] = {k: tokens[k] / max(1, len(queries if k == "query" else passages)) for k in tokens}

    # Timing.
    qb, pb, warm = spec["query_batch"], spec["passage_batch"], spec["warmup"]
    q_batches = [queries[i:i + qb] for i in range(0, min(len(queries), spec["latency_n"]), qb)]
    p_batches = [passages[i:i + pb] for i in range(0, len(passages), pb)]
    q_feeds = [enc.tokenize(b) for b in q_batches]
    p_feeds = [enc.tokenize(b) for b in p_batches]
    tok_times = []
    for b in q_batches[:200]:
        s = time.perf_counter()
        enc.tokenize(b)
        tok_times.append((time.perf_counter() - s) * 1000)
    result["tokenize_query_ms_p50"] = pct(tok_times, 50)
    timings = {}
    loaded_threads = max(threads_list)
    for threads in threads_list:
        if threads != loaded_threads:  # reload on every change (the old check left 1-thread sessions timing the max)
            del enc
            enc = _load(spec["kind"], spec["path"], threads, spec.get("max_length"), spec["arena"])
            loaded_threads = threads
        for feeds in q_feeds[:warm] + p_feeds[:max(1, warm // 4)]:
            enc.run(feeds)
        q_ms = []
        for feeds in q_feeds:
            s = time.perf_counter()
            enc.run(feeds)
            q_ms.append((time.perf_counter() - s) * 1000)
        p_ms, p_tokens = [], 0
        s_all = time.perf_counter()
        for feeds in p_feeds:
            s = time.perf_counter()
            enc.run(feeds)
            p_ms.append((time.perf_counter() - s) * 1000)
            p_tokens += int(feeds["attention_mask"].sum())
        p_total = time.perf_counter() - s_all
        timings[str(threads)] = {
            "query_ms_p50": pct(q_ms, 50), "query_ms_p90": pct(q_ms, 90), "query_ms_p99": pct(q_ms, 99),
            "queries_per_s": len(q_ms) * qb / max(1e-9, sum(q_ms) / 1000),
            "passage_batch_ms_p50": pct(p_ms, 50), "passage_batch_ms_p90": pct(p_ms, 90),
            "passages_per_s": len(passages) / max(1e-9, p_total), "passage_tokens_per_s": p_tokens / max(1e-9, p_total),
        }
        log(f"{spec['label']:>14} threads {threads}: query p50 {timings[str(threads)]['query_ms_p50']:.1f} ms, "
            f"passages {timings[str(threads)]['passages_per_s']:.1f}/s")
    result["timings"] = timings
    result["rss_peak_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result


# --------------------------------------------------------------------------------------------- parent


def load_vectors(path: Path) -> np.ndarray:
    data = np.load(path)
    n, dim = (int(x) for x in data["shape"])
    out = np.zeros((n, dim), dtype=np.float32)
    out[data["rows"], data["cols"]] = data["vals"]
    return out


def _worker_env() -> dict[str, str]:
    repo = str(Path(__file__).resolve().parents[1])
    path = os.environ.get("PYTHONPATH", "")
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{repo}{os.pathsep}{path}" if path else repo)


def run_worker(spec: dict[str, Any], spec_path: Path) -> dict[str, Any]:
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "release.onnx_bench", "--worker", str(spec_path)],
                          stdout=subprocess.PIPE, text=True, env=_worker_env())
    if proc.returncode != 0:
        raise RuntimeError(f"worker for {spec['label']} failed with exit code {proc.returncode}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def markdown(report: dict[str, Any]) -> str:
    threads = report["threads"]
    head = ["variant", "MB", "RSS peak"]
    for t in threads:
        head += [f"q p50 ms ({t}t)", f"passages/s ({t}t)"]
    head += ["cos q", "cos p", "Jacc p", "dims q", "dims p", "top10 sym", "top10 q-only"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in report["results"]:
        row = [r["label"], f"{r.get('file_mb', float('nan')):.0f}", f"{r['rss_peak_mb']:.0f}"]
        for t in threads:
            tm = r["timings"][str(t)]
            row += [f"{tm['query_ms_p50']:.1f}", f"{tm['passages_per_s']:.1f}"]
        f = r.get("fidelity")
        if f:
            row += [f"{f['query']['cosine']['mean']:.4f}", f"{f['passage']['cosine']['mean']:.4f}",
                    f"{f['passage']['active_jaccard']['mean']:.3f}", f"{f['query']['dims']['mean']:.1f}",
                    f"{f['passage']['dims']['mean']:.1f}", f"{f['retrieval']['symmetric']['mean']:.3f}",
                    f"{f['retrieval']['query_only']['mean']:.3f}"]
        else:
            row += ["ref", "ref", "ref", f"{r['dims']['query']:.1f}", f"{r['dims']['passage']:.1f}", "ref", "ref"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--variants", nargs="+", help="export directories (onnx_splade.json inside)")
    p.add_argument("--reference", default=None, help="reference export (default: the first --variants entry)")
    p.add_argument("--torch-model", default=None, help="also time the PyTorch fp32 checkpoint on the CPU")
    p.add_argument("--texts", default="datasets/kd/kd_v1.jsonl")
    p.add_argument("--n", default="query=512,passage=256", help="texts per kind (query and passage are both required)")
    p.add_argument("--seed", type=int, default=7, help="not the export's calibration seed (13)")
    p.add_argument("--threads", default="1,6", help="comma list of intra-op thread counts")
    p.add_argument("--query-batch", type=int, default=1)
    p.add_argument("--passage-batch", type=int, default=8)
    p.add_argument("--latency-n", type=int, default=300, help="queries timed per thread count")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--max-length", type=int, default=None, help="tokenizer cap (default: the export's max_length)")
    p.add_argument("--no-arena", action="store_true", help="onnxruntime without the CPU memory arena")
    p.add_argument("--min-free-gb", type=float, default=4.0, help="refuse to start a worker with less RAM available")
    p.add_argument("--out", required=False)
    args = p.parse_args(argv)

    if args.worker:
        result = worker(json.loads(Path(args.worker).read_text(encoding="utf-8")))
        print(json.dumps(result))
        return 0
    if not args.variants:
        p.error("--variants is required")

    threads = [int(t) for t in args.threads.split(",")]
    quotas = parse_quotas(args.n)
    if not {"query", "passage"} <= set(quotas):
        p.error("--n needs query=... and passage=...")
    texts = sample_texts(args.texts, quotas, seed=args.seed)
    log(f"{len(texts['query'])} queries, {len(texts['passage'])} passages from {args.texts}")
    environment = {
        "loadavg_start": os.getloadavg(), "gpu_compute_processes": gpu_compute_processes(),
        "mem_available_gb_start": mem_available_gb(), "cpu_count": os.cpu_count(),
    }
    if environment["loadavg_start"][0] > 1.0 or environment["gpu_compute_processes"]:
        log(f"WARNING: machine not idle (load {environment['loadavg_start'][0]:.1f}, GPU processes "
            f"{environment['gpu_compute_processes']}); speed numbers will be noisy")

    reference = args.reference or args.variants[0]
    specs = [("onnx", v) for v in dict.fromkeys([reference] + list(args.variants))]
    if args.torch_model:
        specs.append(("torch", args.torch_model))

    results = []
    with tempfile.TemporaryDirectory(prefix="onnx_bench.", dir=os.environ.get("ONNX_BENCH_TMP")) as tmp:
        tmp_dir = Path(tmp)
        texts_path = tmp_dir / "texts.json"
        texts_path.write_text(json.dumps({"query": texts["query"], "passage": texts["passage"]}, ensure_ascii=False), encoding="utf-8")
        nq = len(texts["query"])
        ref_vec = None
        for i, (kind, path) in enumerate(specs):
            avail = mem_available_gb()
            if avail < args.min_free_gb:
                raise SystemExit(f"only {avail:.1f} GB RAM available (< --min-free-gb {args.min_free_gb}); not starting {path}")
            label = "torch-fp32" if kind == "torch" else Path(path).name
            spec = {
                "label": label, "kind": kind, "path": str(path), "texts": str(texts_path), "threads": threads,
                "query_batch": args.query_batch, "passage_batch": args.passage_batch, "latency_n": args.latency_n,
                "warmup": args.warmup, "max_length": args.max_length, "arena": not args.no_arena,
                "vectors": str(tmp_dir / f"vectors_{i}.npz"),
            }
            log(f"worker {label} ({path})")
            result = run_worker(spec, tmp_dir / f"spec_{i}.json")
            vec = load_vectors(Path(spec["vectors"]))
            result["dims"] = {
                "query": float((np.rint(vec[:nq] * 100) > 0).sum(axis=1).mean()),
                "passage": float((np.rint(vec[nq:] * 100) > 0).sum(axis=1).mean()),
            }
            if ref_vec is None:
                ref_vec = vec
                result["reference"] = True
            else:
                result["fidelity"] = {
                    "query": compare_vectors(ref_vec[:nq], vec[:nq]),
                    "passage": compare_vectors(ref_vec[nq:], vec[nq:]),
                    "retrieval": retrieval_agreement(ref_vec[:nq], ref_vec[nq:], vec[:nq], vec[nq:]),
                }
            del vec
            Path(spec["vectors"]).unlink(missing_ok=True)
            results.append(result)

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "reference": str(reference), "threads": threads,
        "texts": {"source": args.texts, "seed": args.seed, "query": len(texts["query"]), "passage": len(texts["passage"])},
        "batches": {"query": args.query_batch, "passage": args.passage_batch},
        "environment": dict(environment, loadavg_end=os.getloadavg()), "results": results,
    }
    table = markdown(report)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        out.with_suffix(".md").write_text(table, encoding="utf-8")
        log(f"wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
