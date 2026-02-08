#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cache teacher embeddings to disk (Parquet chunks), with MetricX-based filtering.

Input: local Parquet shards (scored) with columns:
  english (str), polish (str), metricx_pred (float), idx (int)

Output: cached Parquet chunks with columns:
  english (str), non_english (str), label (vector), label_dtype (str)

label storage formats:
  - float32
  - float16
  - bf16_u16  (bfloat16 bits packed into uint16; safest for Parquet/Arrow)
"""
from datasets import disable_caching
disable_caching()

import pyarrow.parquet as pq
import argparse
import gc
import hashlib
import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import model_info
from sentence_transformers import LoggingHandler, SentenceTransformer

def parquet_has_column(path: Path, col: str) -> bool:
    try:
        schema = pq.read_schema(path)  # metadata-only
        return col in schema.names
    except Exception:
        return False


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_TEACHER_MAX_SEQ_LENGTH = 4096
DEFAULT_INFERENCE_BATCH_SIZE = 8
DEFAULT_CHUNK_SIZE_SAMPLES = 20_480

DEFAULT_SCORES_ROOT = "outputs_google__metricx-24-hybrid-large-v2p6_parquet"
DEFAULT_QUALITY_COLUMN = "metricx_pred"
DEFAULT_QUALITY_THRESHOLD = 7.0
DEFAULT_EMBED_CACHE_COMPRESSION = "zstd"
DEFAULT_NUM_ENCODE_WORKERS = max(1, (os.cpu_count() or 2) // 2)

# -----------------------------
# Helpers
# -----------------------------
def _parse_langs(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        for part in v.split(","):
            p = part.strip()
            if p:
                out.append(p)
    return out

def _has_st_modules(model_name: str) -> bool:
    path = Path(model_name)
    if path.exists():
        return (path / "modules.json").is_file()
    try:
        info = model_info(model_name)
        return any(s.rfilename == "modules.json" for s in info.siblings)
    except Exception:
        logger.exception("Failed to check modules.json for '%s'. Assuming base Transformers model.", model_name)
        return False

@dataclass(frozen=True)
class CacheConfig:
    cache_dir: Path
    chunk_size_samples: int
    num_encode_workers: int
    embed_cache_compression: str
    embed_cache_dtype: str  # float32 | float16 | bf16_u16

@dataclass(frozen=True)
class FilterRule:
    col: str
    op: str  # >=, >, <=, <
    thr: float

_FILTER_RE = re.compile(r"^\s*([A-Za-z0-9_\-\.]+)\s*(>=|<=|>|<)\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")

def parse_filter_rules(rules: list[str]) -> list[FilterRule]:
    out: list[FilterRule] = []
    for r in rules:
        m = _FILTER_RE.match(r)
        if not m:
            raise ValueError(f"Invalid --filter-rule '{r}'. Expected: metricx_pred<=5.0")
        out.append(FilterRule(col=m.group(1), op=m.group(2), thr=float(m.group(3))))
    return out

def filter_tag(rules: list[FilterRule]) -> str:
    if not rules:
        return "nofilter"
    s = ";".join([f"{x.col}{x.op}{x.thr:g}" for x in rules])
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"f_{h}"

def roots_tag(roots: list[Path]) -> str:
    if not roots:
        return "no_local"
    s = ";".join(sorted([str(p.resolve()) for p in roots]))
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"local_{h}"

def get_cached_dataset_root(
    cfg: CacheConfig,
    dataset_key: str,
    subset: str,
    split: str,
    teacher_model_name: str,
    cache_tag: str,
) -> Path:
    safe_dataset = dataset_key.replace("/", "__").replace(":", "__")
    safe_teacher = teacher_model_name.replace("/", "__")
    dir_name = (
        f"{safe_dataset}__{subset}__{split}__{safe_teacher}"
        f"__cs{cfg.chunk_size_samples}__{cache_tag}"
        f"__dtype{cfg.embed_cache_dtype}"
        f"__parquet_{cfg.embed_cache_compression.lower()}"
    )
    return cfg.cache_dir / dir_name

def collect_local_parquet_files(roots: list[Path], subset: str, split: str) -> list[str]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            logger.warning("Local root does not exist: %s", str(root))
            continue
        for p in root.rglob("*.parquet"):
            if p.name.endswith(".parquet.tmp") or p.name.endswith(".tmp"):
                continue
            parts = p.parts
            if subset in parts and split in parts:
                si = parts.index(subset)
                try:
                    sj = parts.index(split, si + 1)
                except ValueError:
                    continue
                if sj > si:
                    files.append(p)
    files_sorted = sorted(set(files))
    return [str(p) for p in files_sorted]

def load_local_parquet_dataset(parquet_files: list[str]) -> Optional[Dataset]:
    if not parquet_files:
        return None
    return load_dataset("parquet", data_files=parquet_files, split="train")

def apply_rules_mask(batch: dict[str, Any], rules: list[FilterRule]) -> np.ndarray:
    if not rules:
        first_key = next(iter(batch.keys()))
        return np.ones(len(batch[first_key]), dtype=bool)

    mask = None
    for rule in rules:
        if rule.col not in batch:
            raise KeyError(f"Filter column '{rule.col}' missing. Available: {list(batch.keys())}")
        vals = np.asarray(batch[rule.col], dtype=np.float32)
        if mask is None:
            mask = np.ones(vals.shape[0], dtype=bool)

        if rule.op == ">=":
            mask &= vals >= rule.thr
        elif rule.op == ">":
            mask &= vals > rule.thr
        elif rule.op == "<=":
            mask &= vals <= rule.thr
        elif rule.op == "<":
            mask &= vals < rule.thr
        else:
            raise ValueError(f"Unsupported op: {rule.op}")

    assert mask is not None
    return mask

def _select_list_by_mask(xs: list[Any], mask: np.ndarray) -> list[Any]:
    idxs = np.nonzero(mask)[0].tolist()
    return [xs[i] for i in idxs]

def _ensure_text_list(xs: list[Any]) -> list[str]:
    return ["" if x is None else str(x) for x in xs]

def _pack_bf16_u16(x_f32: np.ndarray) -> np.ndarray:
    """
    Convert float32 [N,D] -> bf16 bits stored in uint16 [N,D]
    """
    t = torch.from_numpy(x_f32.astype(np.float32, copy=False)).to(torch.bfloat16)
    u16 = t.view(torch.uint16).cpu().numpy()
    return u16

def _prepare_label_for_disk(cfg: CacheConfig, embeddings_f32: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Returns (label_array, label_dtype_str)
    """
    dt = cfg.embed_cache_dtype
    if dt == "float32":
        return embeddings_f32.astype(np.float32, copy=False), "float32"
    if dt == "float16":
        return embeddings_f32.astype(np.float16, copy=False), "float16"
    if dt == "bf16_u16":
        return _pack_bf16_u16(embeddings_f32), "bf16_u16"
    raise ValueError(f"Unsupported --embed-cache-dtype: {dt}")

def load_or_prepare_split_with_embeddings(
    *,
    hf_dataset: Dataset,
    dataset_key: str,
    subset: str,
    split: str,
    teacher_model: SentenceTransformer,
    teacher_model_name: str,
    inference_batch_size: int,
    cfg: CacheConfig,
    cache_tag: str,
    src_col: str,
    tgt_col: str,
    rules: list[FilterRule],
    max_samples: int | None = None,
) -> list[Path]:
    cache_root = get_cached_dataset_root(cfg, dataset_key, subset, split, teacher_model_name, cache_tag)
    chunks_root = cache_root / "chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    total_samples = len(hf_dataset)
    num_samples = min(total_samples, max_samples) if max_samples is not None else total_samples

    logger.info(
        "[%s | %s | %s] Preparing chunks: %d / %d samples (limit=%s) | filter=%s | dtype=%s | comp=%s",
        dataset_key, subset, split, num_samples, total_samples, str(max_samples),
        cache_tag, cfg.embed_cache_dtype, cfg.embed_cache_compression,
    )

    if num_samples == 0:
        return []

    chunk_files: list[Path] = []
    save_futures = []

    def _save_chunk_parquet(chunk_file_str: str, english: list[str], non_english: list[str], label_arr: np.ndarray, label_dtype: str) -> None:
        ds_chunk = Dataset.from_dict(
            {
                "english": english,
                "non_english": non_english,
                "label": label_arr,
                "label_dtype": [label_dtype] * len(english),
            }
        )
        comp = cfg.embed_cache_compression
        if comp.upper() == "NONE":
            ds_chunk.to_parquet(chunk_file_str)
        else:
            ds_chunk.to_parquet(chunk_file_str, compression=comp)

    with ThreadPoolExecutor(max_workers=2) as executor:
        chunk_id = 0
        for start in range(0, num_samples, cfg.chunk_size_samples):
            end = min(start + cfg.chunk_size_samples, num_samples)
            chunk_file = chunks_root / f"chunk_{chunk_id:05d}.parquet"
            empty_sentinel = chunks_root / f"chunk_{chunk_id:05d}.empty.json"

            if empty_sentinel.exists():
                logger.info("Reusing cached EMPTY chunk %d (%d:%d)", chunk_id, start, end)
                chunk_id += 1
                continue

            if chunk_file.exists() and chunk_file.stat().st_size > 0:
                if parquet_has_column(chunk_file, "label"):
                    logger.info("Reusing cached chunk %d (%d:%d)", chunk_id, start, end)
                    chunk_files.append(chunk_file)
                    chunk_id += 1
                    continue
                else:
                    logger.info("Chunk %d exists but missing 'label' column; recomputing.", chunk_id)

            logger.info("Computing chunk %d (%d:%d)", chunk_id, start, end)
            slice_ds = hf_dataset.select(range(start, end))
            batch = slice_ds[:]  # dict of lists

            if src_col not in batch or tgt_col not in batch:
                raise KeyError(f"Missing src/tgt columns: src='{src_col}', tgt='{tgt_col}'. Available: {list(batch.keys())}")

            en_all = _ensure_text_list(batch[src_col])
            tgt_all = _ensure_text_list(batch[tgt_col])

            mask = apply_rules_mask(batch, rules) if rules else np.ones(len(en_all), dtype=bool)
            keep_n = int(mask.sum())

            if keep_n == 0:
                empty_sentinel.write_text(
                    json.dumps({"start": start, "end": end, "subset": subset, "split": split, "filter": cache_tag}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("Chunk %d filtered to 0 samples -> marked EMPTY.", chunk_id)
                chunk_id += 1
                del slice_ds, batch, en_all, tgt_all, mask
                continue

            en = _select_list_by_mask(en_all, mask)
            non_en = _select_list_by_mask(tgt_all, mask)

            # Always compute/normalize in float32 for stability, then cast for disk
            emb = teacher_model.encode(
                en,
                batch_size=inference_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                num_workers=cfg.num_encode_workers,
            ).astype(np.float32, copy=False)

            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-12, None)

            label_arr, label_dtype = _prepare_label_for_disk(cfg, emb)

            logger.info("Saving chunk %d (kept=%d) to: %s", chunk_id, keep_n, chunk_file)
            future = executor.submit(_save_chunk_parquet, str(chunk_file), en, non_en, label_arr, label_dtype)
            save_futures.append(future)

            chunk_files.append(chunk_file)
            chunk_id += 1

            del slice_ds, batch, en_all, tgt_all, mask, en, non_en, emb, label_arr
            gc.collect()

        for f in save_futures:
            f.result()

    return chunk_files

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache teacher embeddings (Parquet chunks) with MetricX filtering.")

    # Teacher
    p.add_argument("--teacher-model", type=str, default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--teacher-max-seq-length", type=int, default=DEFAULT_TEACHER_MAX_SEQ_LENGTH)
    p.add_argument("--inference-batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)

    # Langs / subsets
    p.add_argument("--source-langs", nargs="+", default=["en"])
    p.add_argument("--target-langs", nargs="+", default=["pl"])

    # MetricX root + filtering
    p.add_argument("--scores-parquet-root", type=str, default=DEFAULT_SCORES_ROOT)
    p.add_argument("--quality-column", type=str, default=DEFAULT_QUALITY_COLUMN)
    p.add_argument("--quality-threshold", type=float, default=DEFAULT_QUALITY_THRESHOLD)
    p.add_argument("--filter-rule", action="append", default=[], help="Extra AND rules, e.g. metricx_pred<=4.5")

    # Local column names
    p.add_argument("--local-src-col", type=str, default="english")
    p.add_argument("--local-tgt-col", type=str, default="polish")

    # Cache config
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE_SAMPLES)
    p.add_argument("--cache-dir", type=str, default="embedding_dataset_cache")
    p.add_argument("--num-encode-workers", type=int, default=DEFAULT_NUM_ENCODE_WORKERS)
    p.add_argument("--embed-cache-compression", type=str, default=DEFAULT_EMBED_CACHE_COMPRESSION, choices=["zstd", "snappy", "gzip", "NONE"])
    p.add_argument("--embed-cache-dtype", type=str, default="bf16_u16", choices=["float32", "float16", "bf16_u16"])

    # Limits
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = no limit")
    p.add_argument("--max-dev-samples", type=int, default=0, help="0 = no limit")

    args = p.parse_args()
    args.source_langs = _parse_langs(args.source_langs)
    args.target_langs = _parse_langs(args.target_langs)
    return args

def main() -> None:
    args = parse_args()

    source_languages = set(args.source_langs)
    target_languages = set(args.target_langs)

    # Base quality rule + extras
    auto_quality_rule = f"{args.quality_column}<={args.quality_threshold}"
    rules = parse_filter_rules([auto_quality_rule]) + parse_filter_rules(args.filter_rule)
    ftag = filter_tag(rules)

    scores_root = Path(args.scores_parquet_root)
    local_roots = [scores_root]
    ltag = roots_tag(local_roots)

    cache_cfg = CacheConfig(
        cache_dir=Path(args.cache_dir),
        chunk_size_samples=args.chunk_size,
        num_encode_workers=args.num_encode_workers,
        embed_cache_compression=args.embed_cache_compression,
        embed_cache_dtype=args.embed_cache_dtype,
    )
    cache_cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    # Teacher model
    teacher_model_name = args.teacher_model
    teacher_model = SentenceTransformer(
        teacher_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {},
    )
    teacher_model.max_seq_length = args.teacher_max_seq_length
    logger.info("Teacher model: %s", teacher_model_name)

    dataset_key = f"metricx_scores:{ltag}"
    cache_tag_local = f"{ftag}__metricx__{ltag}"

    total_chunks = 0
    for src in source_languages:
        for tgt in target_languages:
            subset = f"{src}-{tgt}"

            # TRAIN
            train_files = collect_local_parquet_files(local_roots, subset=subset, split="train")
            if not train_files:
                logger.warning("No local train shards found for subset=%s under %s", subset, str(scores_root))
                continue
            train_ds = load_local_parquet_dataset(train_files)
            if train_ds is None or len(train_ds) == 0:
                continue

            # DEV (optional)
            dev_files = collect_local_parquet_files(local_roots, subset=subset, split="dev")
            dev_ds = load_local_parquet_dataset(dev_files) if dev_files else None
            if dev_ds is not None and len(dev_ds) == 0:
                dev_ds = None

            max_train = None if args.max_train_samples <= 0 else int(args.max_train_samples)
            max_dev = None if args.max_dev_samples <= 0 else int(args.max_dev_samples)

            train_chunks = load_or_prepare_split_with_embeddings(
                hf_dataset=train_ds,
                dataset_key=dataset_key,
                subset=subset,
                split="train",
                teacher_model=teacher_model,
                teacher_model_name=teacher_model_name,
                inference_batch_size=args.inference_batch_size,
                cfg=cache_cfg,
                cache_tag=cache_tag_local,
                src_col=args.local_src_col,
                tgt_col=args.local_tgt_col,
                rules=rules,
                max_samples=max_train,
            )
            total_chunks += len(train_chunks)

            if dev_ds is not None and len(dev_ds) > 0:
                dev_chunks = load_or_prepare_split_with_embeddings(
                    hf_dataset=dev_ds,
                    dataset_key=dataset_key,
                    subset=subset,
                    split="dev",
                    teacher_model=teacher_model,
                    teacher_model_name=teacher_model_name,
                    inference_batch_size=args.inference_batch_size,
                    cfg=cache_cfg,
                    cache_tag=cache_tag_local,
                    src_col=args.local_src_col,
                    tgt_col=args.local_tgt_col,
                    rules=rules,
                    max_samples=max_dev,
                )
                total_chunks += len(dev_chunks)

            # free
            del train_ds, dev_ds
            gc.collect()

    logger.info("DONE. Total chunk files prepared/reused: %d", total_chunks)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("Fatal error:\n%s", traceback.format_exc())
        raise
