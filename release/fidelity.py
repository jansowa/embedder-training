"""Agreement of a quantized SPLADE export with its fp32 reference, numpy only.

Per text: cosine, relative L1 error, Jaccard of the dimensions PIRB indexes (``round(w * 100) > 0``), the
active dimension counts of both, and the overlap of the 10 strongest terms. Optionally a small in-sample
retrieval check: each query ranks the passages of the sample by dot product under both models, and the
overlap of the two top-10 lists is reported (a cheap proxy before the PIRB runs).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from release.onnx_encoder import active_mask


def summary(values: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "p10": math.nan, "p50": math.nan, "p90": math.nan}
    return {
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def compare_vectors(ref: np.ndarray, cand: np.ndarray, quantization_factor: int = 100, top_k: int = 10) -> dict[str, Any]:
    ref = np.asarray(ref, dtype=np.float32)
    cand = np.asarray(cand, dtype=np.float32)
    dot = (ref * cand).sum(axis=1)
    norms = np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1)
    cosine = np.where(norms > 0, dot / np.maximum(norms, 1e-12), np.where((ref == cand).all(axis=1), 1.0, 0.0))
    rel_l1 = np.abs(ref - cand).sum(axis=1) / np.maximum(np.abs(ref).sum(axis=1), 1e-12)
    a, b = active_mask(ref, quantization_factor), active_mask(cand, quantization_factor)
    inter, union = (a & b).sum(axis=1), (a | b).sum(axis=1)
    jaccard = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    k = min(top_k, ref.shape[1])
    top_ref = np.argpartition(-ref, k - 1, axis=1)[:, :k]
    top_cand = np.argpartition(-cand, k - 1, axis=1)[:, :k]
    top_overlap = [len(set(r.tolist()) & set(c.tolist())) / k for r, c in zip(top_ref, top_cand)]
    return {
        "n": int(ref.shape[0]),
        "cosine": summary(cosine),
        "rel_l1": summary(rel_l1),
        "active_jaccard": summary(jaccard),
        f"top{k}_term_overlap": summary(top_overlap),
        "dims_ref": summary(a.sum(axis=1)),
        "dims": summary(b.sum(axis=1)),
        "dims_ratio": float(b.sum() / max(1, a.sum())),
    }


def retrieval_agreement(
    ref_q: np.ndarray, ref_p: np.ndarray, cand_q: np.ndarray, cand_p: np.ndarray, top_k: int = 10
) -> dict[str, Any]:
    """Overlap of the top-k passages each query retrieves from the sample, fp32 vs candidate (both sides encoded
    by the candidate: the symmetric deployment), plus the same with only the queries from the candidate
    (asymmetric: an fp32 index queried by a quantized encoder)."""
    k = min(top_k, ref_p.shape[0])

    def top(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.argsort(-(q @ p.T), axis=1, kind="stable")[:, :k]

    base = top(ref_q, ref_p)
    out = {}
    for name, (q, p) in {"symmetric": (cand_q, cand_p), "query_only": (cand_q, ref_p)}.items():
        got = top(q, p)
        out[name] = summary([len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(base, got)])
    out["k"] = k
    out["n_queries"] = int(ref_q.shape[0])
    out["n_passages"] = int(ref_p.shape[0])
    return out
