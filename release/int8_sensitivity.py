#!/usr/bin/env python
"""Which nodes break a dynamic int8 export of a SPLADE model, and the smallest repair.

Works on an existing fp32 export (``release.onnx_splade --variants fp32``); nothing is exported from torch here.

1. Input ranges: for every quantizable node (MatMul with a constant weight, Gemm) the range of its input activation
   on a few texts in fp32. Dynamic int8 maps [min, max] of the whole tensor of one run onto 256 levels, so an input
   whose range is set by a few large values leaves the rest of the signal with a handful of levels.
2. Trials against the fp32 vectors of the check texts (cosine, active Jaccard, dims ratio):
   the usual mixed export (everything int8 but the head), every group alone in int8, every group back in fp32,
   every node alone in int8.
3. Repair: nodes go back to fp32 in the order of their own error until the mean cosine reaches ``--target-cosine``;
   the result is written as a complete variant directory (``OUT/mixed-int8-repaired``).

  python -m release.int8_sensitivity --fp32-dir PACK/onnx/TAG/fp32 --texts datasets/kd/kd_v1.jsonl --out OUT
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from release.fidelity import compare_vectors
from release.onnx_encoder import MODEL_FILE, OnnxSplade, read_meta
from release.onnx_splade import copy_variant_files, node_block, preprocess, quantizable_nodes, quantize_dynamic_int8, write_meta
from release.texts import parse_quotas, sample_texts

LOG = logging.getLogger("release.int8_sensitivity")

_ALBERT_GROUP = re.compile(r"albert_layer_groups\.(\d+)")
_SCOPE_TAIL = re.compile(r"(?:albert_layers|layer|layers)\.\d+/(.+)$")


def fine_group(name: str) -> str:
    """Like ``onnx_splade.node_block``, but the layer groups of an ALBERT stay apart (they all hold ``albert_layers.0``)."""
    match = _ALBERT_GROUP.search(name)
    if match and node_block(name) != "head":
        return f"group.{match.group(1)}"
    return node_block(name)


def role(name: str) -> str:
    """The part of a node name inside its layer: ``attention/query``, ``ffn_output``, ..."""
    path = name.rsplit("/", 1)[0] if "/" in name else name
    match = _SCOPE_TAIL.search(path)
    return match.group(1) if match else path.strip("/")


def group_order(groups: Sequence[str]) -> list[str]:
    def key(group: str) -> tuple[int, int]:
        if group == "embeddings":
            return (0, 0)
        if "." in group and group.split(".")[1].isdigit():
            return (1, int(group.split(".")[1]))
        return (2, 0) if group == "head" else (3, 0)

    return sorted(set(groups), key=key)


def input_ranges(
    pre: Path, nodes: Sequence[dict[str, str]], feeds: dict[str, np.ndarray], work: Path, threads: int, *, rows: int = 4, max_tokens: int = 128
) -> dict[str, dict[str, float]]:
    """Statistics of the activation input of every node in ``nodes`` on fp32 runs of ``feeds``.

    The texts are cut to ``max_tokens`` and run ``rows`` at a time (the graph holds the [rows, tokens, vocab] logits);
    the statistics are over all runs together."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(pre))
    wanted = {n["name"] for n in nodes}
    tensor_of = {node.name: node.input[0] for node in model.graph.node if node.name in wanted}
    existing = {o.name for o in model.graph.output}
    for tensor in sorted(set(tensor_of.values()) - existing):
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(tensor))
    probe = work / "probe.onnx"
    onnx.save(model, str(probe))
    del model
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep the tensor names
    opts.intra_op_num_threads = threads
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(probe), opts, providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    inputs = {i.name for i in session.get_inputs()}
    collected: dict[str, list[np.ndarray]] = {tensor: [] for tensor in set(tensor_of.values())}
    n = feeds["input_ids"].shape[0]
    for start in range(0, n, rows):
        part = {k: v[start:start + rows, :max_tokens] for k, v in feeds.items() if k in inputs}
        width = max(1, int(part["attention_mask"].sum(axis=1).max()))
        part = {k: v[:, :width] for k, v in part.items()}
        for name, value in zip(names, session.run(None, part)):
            if name in collected:
                collected[name].append(np.asarray(value, dtype=np.float32).ravel())
    del session
    probe.unlink(missing_ok=True)
    stats = {}
    for node, tensor in tensor_of.items():
        x = np.concatenate(collected[tensor]).astype(np.float64)
        lo, hi = min(float(x.min()), 0.0), max(float(x.max()), 0.0)  # DynamicQuantizeLinear always includes zero
        step = (hi - lo) / 255.0 if hi > lo else 0.0
        p_lo, p_hi = np.percentile(x, [0.1, 99.9])
        centred = x - x.mean()
        stats[node] = {
            "min": lo, "max": hi, "mean": float(x.mean()), "std": float(x.std()), "p0.1": float(p_lo), "p99.9": float(p_hi),
            "step": step,
            # how many of the 256 levels the central 99.8 % of the values span, and the spread in steps
            "levels_central": float((p_hi - p_lo) / step) if step else 0.0,
            "std_in_steps": float(centred.std() / step) if step else 0.0,
            "share_below_one_step": float((np.abs(x) < step).mean()) if step else 0.0,
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32-dir", required=True, type=Path, help="an fp32 variant directory of release.onnx_splade")
    ap.add_argument("--texts", required=True, help="jsonl (text, kind), e.g. datasets/kd/kd_v1.jsonl")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--check-n", default="query=64,passage=64", help="the quota of the export check (same seed = same texts)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--range-n", type=int, default=8, help="texts per kind for the input ranges")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fp32-groups", default="head", help="groups that stay fp32 in every trial, as in the release export")
    ap.add_argument("--weight-type", choices=["s8", "u8"], default="u8")
    ap.add_argument("--per-tensor", action="store_true")
    ap.add_argument("--target-cosine", type=float, default=0.97, help="stop the repair at this mean cosine against fp32")
    ap.add_argument("--max-repair-nodes", type=int, default=16)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)

    out, fp32_dir = args.out, args.fp32_dir
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    quota = parse_quotas(args.check_n)
    by_kind = sample_texts(args.texts, quota, seed=args.seed, field=args.text_field)
    texts = [t for kind, pool in sorted(by_kind.items()) for t in pool[: quota.get(kind, 0)]]
    range_texts = [t for kind, pool in sorted(by_kind.items()) for t in pool[: args.range_n]]
    LOG.info("%d check texts (%s), %d texts for the input ranges", len(texts), args.check_n, len(range_texts))

    fp32 = OnnxSplade(fp32_dir, threads=args.threads)
    ref = fp32.encode(texts, batch_size=8)
    range_feeds = fp32.tokenize(range_texts)
    del fp32
    pre = preprocess(fp32_dir / MODEL_FILE, work / "model.pre.onnx")
    nodes = quantizable_nodes(pre)
    for node in nodes:
        node["group"], node["role"] = fine_group(node["name"]), role(node["name"])
    groups = group_order([n["group"] for n in nodes])
    kept = [g.strip() for g in args.fp32_groups.split(",") if g.strip()]
    base_fp32 = [n["name"] for n in nodes if n["group"] in kept]
    probed = [n for n in nodes if n["group"] not in kept]
    LOG.info("quantizable nodes: %s; always fp32: %s", {g: sum(1 for n in nodes if n["group"] == g) for g in groups}, kept)

    try:
        ranges = input_ranges(pre, probed, range_feeds, work, args.threads)
    except Exception as exc:  # the trials below do not need the ranges
        LOG.warning("input ranges failed: %s: %s", type(exc).__name__, exc)
        ranges = {}
    for node in probed:
        r = ranges.get(node["name"])
        if r is None:
            continue
        LOG.info("range %-9s %-28s [%9.2f, %9.2f] std %7.3f  central 99.8 %% on %5.1f levels, std = %5.2f steps, |x| < 1 step: %4.1f %%",
                 node["group"], node["role"], r["min"], r["max"], r["std"], r["levels_central"], r["std_in_steps"], 100 * r["share_below_one_step"])

    trial_dir = work / "trial"
    copy_variant_files(fp32_dir, trial_dir)
    meta = read_meta(fp32_dir)
    meta.pop("torch_check", None)
    write_meta(trial_dir, dict(meta, variant="sensitivity-trial"))
    dyn = dict(per_channel=not args.per_tensor, weight_type=args.weight_type)
    all_names = [n["name"] for n in nodes]

    def trial(label: str, int8_nodes: Sequence[str]) -> dict[str, Any]:
        """Only ``int8_nodes`` are quantized, every other node stays fp32."""
        start = time.time()
        wanted = set(int8_nodes)
        quantize_dynamic_int8(pre, trial_dir / MODEL_FILE, nodes_to_exclude=[n for n in all_names if n not in wanted], **dyn)
        size = (trial_dir / MODEL_FILE).stat().st_size
        stats = compare_vectors(ref, OnnxSplade(trial_dir, threads=args.threads).encode(texts, batch_size=8))
        gc.collect()
        row = {"trial": label, "int8_nodes": len(wanted), "cosine": stats["cosine"]["mean"], "cosine_p10": stats["cosine"]["p10"],
               "active_jaccard": stats["active_jaccard"]["mean"], "top10_term_overlap": stats["top10_term_overlap"]["mean"],
               "dims_ratio": stats["dims_ratio"], "bytes": size, "seconds": round(time.time() - start, 1)}
        LOG.info("%-46s int8 nodes %2d  cosine %.4f  jaccard %.4f  dims ratio %.3f  %5.0f MB", label, len(wanted), row["cosine"],
                 row["active_jaccard"], row["dims_ratio"], size / 1e6)
        return row

    report: dict[str, Any] = {
        "fp32_dir": str(fp32_dir), "texts": {"source": args.texts, "check": len(texts), "seed": args.seed}, "quantization": dict(dyn, fp32_groups=kept),
        "nodes": [dict(n, input_range=ranges.get(n["name"])) for n in nodes],
    }
    probed_names = [n["name"] for n in probed]
    report["release_form"] = trial("release form: all int8 but " + "+".join(kept), probed_names)
    report["only_group_int8"], report["group_back_to_fp32"], report["only_node_int8"] = [], [], []
    for group in [g for g in groups if g not in kept]:
        members = [n["name"] for n in probed if n["group"] == group]
        report["only_group_int8"].append(dict(trial(f"only {group} int8", members), group=group))
    for group in [g for g in groups if g not in kept]:
        rest = [n["name"] for n in probed if n["group"] != group]
        report["group_back_to_fp32"].append(dict(trial(f"{group} back to fp32", rest), group=group))
    for node in probed:
        try:
            row = trial(f"only {node['group']} {node['role']} int8", [node["name"]])
        except Exception as exc:  # onnxruntime cannot quantize some nodes alone (seen on the extra ALBERT layer: scale assertion)
            LOG.warning("only %s %s int8: the quantizer failed (%s: %s); the group is repaired as a whole", node["group"], node["role"], type(exc).__name__, exc)
            row = {"trial": f"only {node['group']} {node['role']} int8", "int8_nodes": 1, "cosine": None, "error": f"{type(exc).__name__}: {exc}"}
        report["only_node_int8"].append(dict(row, node=node["name"], group=node["group"], role=node["role"]))

    # Repair: the units with the largest own error go back to fp32 one by one. A unit is a single node; a group with a
    # node the quantizer could not handle alone is one unit, ranked by the error of the whole group.
    failed_groups = {r["group"] for r in report["only_node_int8"] if r["cosine"] is None}
    group_cosine = {r["group"]: r["cosine"] for r in report["only_group_int8"]}
    units = [{"label": f"{r['group']} {r['role']}", "cosine": r["cosine"], "nodes": [r["node"]]}
             for r in report["only_node_int8"] if r["group"] not in failed_groups]
    units += [{"label": f"{g} (whole group)", "cosine": group_cosine[g], "nodes": [n["name"] for n in probed if n["group"] == g]} for g in sorted(failed_groups)]
    order = sorted(units, key=lambda u: u["cosine"])
    fp32_extra: list[str] = []
    steps = []
    used = 0
    reached = report["release_form"]["cosine"] >= args.target_cosine
    while not reached and used < len(order) and len(fp32_extra) < args.max_repair_nodes:
        nxt = order[used]
        used += 1
        fp32_extra.extend(nxt["nodes"])
        row = trial(f"repair {used}: + {nxt['label']} fp32", [n for n in probed_names if n not in fp32_extra])
        steps.append(dict(row, fp32_nodes=list(fp32_extra)))
        reached = row["cosine"] >= args.target_cosine
    report["repair"] = {"target_cosine": args.target_cosine, "reached": reached, "steps": steps, "fp32_nodes": base_fp32 + fp32_extra}

    if reached and fp32_extra:
        vdir = out / "mixed-int8-repaired"
        copy_variant_files(fp32_dir, vdir)
        exclude = base_fp32 + fp32_extra
        quant = quantize_dynamic_int8(pre, vdir / MODEL_FILE, nodes_to_exclude=exclude, **dyn)
        quant.update(method="mixed (dynamic int8 + fp32 nodes chosen by release.int8_sensitivity)", fp32_groups=kept, fp32_nodes=exclude)
        vmeta = dict(meta, variant="mixed-int8", quantization=quant, created=_dt.datetime.now().isoformat(timespec="seconds"))
        write_meta(vdir, vmeta)
        vmeta["fp32_check"] = compare_vectors(ref, OnnxSplade(vdir, threads=args.threads).encode(texts, batch_size=8))
        write_meta(vdir, vmeta)
        report["repaired_variant"] = {"dir": str(vdir), "bytes": (vdir / MODEL_FILE).stat().st_size, "fp32_check": vmeta["fp32_check"]}
        LOG.info("repaired variant %s: cosine %.4f, active Jaccard %.4f, dims ratio %.3f, %d nodes fp32, %.0f MB", vdir,
                 vmeta["fp32_check"]["cosine"]["mean"], vmeta["fp32_check"]["active_jaccard"]["mean"], vmeta["fp32_check"]["dims_ratio"],
                 len(exclude), report["repaired_variant"]["bytes"] / 1e6)
    elif not reached:
        LOG.info("the target cosine %.3f was not reached with %d nodes back in fp32", args.target_cosine, len(fp32_extra))
    shutil.rmtree(work, ignore_errors=True)
    (out / "int8_sensitivity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOG.info("wrote %s", out / "int8_sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
