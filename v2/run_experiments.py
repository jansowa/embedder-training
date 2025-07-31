#!/usr/bin/env python
"""
Grid‑search dla FlagEmbedding + ewaluacja po każdej epoce w MTEB.

Użycie:
    python run_experiments.py configs/grid.yaml
Jeśli plik konfiguracyjny nie zostanie podany – użyje wbudowanego przykładu.

Wymagania:
    pip install flagembedding transformers sentence-transformers \
               mteb wandb pyyaml deepspeed accelerate
"""
import sys
import yaml
import subprocess
import itertools
from pathlib import Path
import wandb
import mteb

from convert_utils import convert_to_sentence_transformer, run_mteb

# ------------------------------------------------------------------
# 1. Wczytanie siatki eksperymentów z YAML‑a  lub fallback
# ------------------------------------------------------------------
if len(sys.argv) > 1:
    with open(sys.argv[1]) as f:
        grid = yaml.safe_load(f)
else:
    grid = {
        "architectures": [
            "answerdotai/ModernBERT-base",
            "answerdotai/ModernBERT-large",
        ],
        "hparams": [
            {"lr": 2.5e-5, "epochs": 1},
            {"lr": 2.5e-5, "epochs": 2},
            {"lr": 2.5e-5, "epochs": 3},
            {"lr": 5e-5,   "epochs": 1},
            {"lr": 5e-5,   "epochs": 2},
            {"lr": 5e-5,   "epochs": 3},
        ],
    }

TASKS = mteb.get_benchmarks(names=["NanoBEIR"])   # jeden lekki benchmark
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 2. Pętla po kombinacjach architektura × hiperparametry
# ------------------------------------------------------------------
for arch, cfg in itertools.product(grid["architectures"], grid["hparams"]):
    safe_arch = arch.replace("/", "_")
    run_name = f"{safe_arch}-{cfg['lr']}lr-{cfg['epochs']}ep"

    run = wandb.init(
        project="flagembed-ir",
        name=run_name,
        config={**cfg, "arch": arch},
    )

    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------- TRENING -----------------------------
    cmd = [
        "torchrun", "--nproc_per_node", "1", "-m",
        "FlagEmbedding.finetune.embedder.encoder_only.base",
        "--model_name_or_path", arch,
        "--cache_dir", "./cache/model",
        "--train_data", "./dataset-no_in_batch_neg",
        "--cache_path", "./cache/data",
        "--train_group_size", "6",
        "--query_max_len", "512",
        "--passage_max_len", "512",
        "--pad_to_multiple_of", "8",
        "--query_instruction_for_retrieval",
        "Represent this sentence for searching relevant passages: ",
        "--query_instruction_format", "{}{}",
        "--knowledge_distillation", "True",
        "--output_dir", str(output_dir),
        "--overwrite_output_dir",
        "--learning_rate", str(cfg["lr"]),
        "--fp16",
        "--num_train_epochs", str(cfg["epochs"]),
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "16",
        "--dataloader_drop_last", "True",
        "--warmup_ratio", "0.1",
        "--gradient_checkpointing",
        "--deepspeed", "./ds_stage0.json",
        "--logging_steps", "1",
        "--save_strategy", "epoch",
        "--save_total_limit", str(cfg["epochs"]),
        "--negatives_cross_device",
        "--temperature", "0.02",
        "--sentence_pooling_method", "cls",
        "--normalize_embeddings", "True",
        "--kd_loss_type", "kl_div",
        "--report_to", "wandb",
        "--run_name", run_name,
    ]

    print(">>> LAUNCH:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # ----------------------- EWALUACJA ------------------------------
    ckpt_dirs = sorted(output_dir.glob("checkpoint-*"),
                       key=lambda p: p.stat().st_mtime)

    for idx, ckpt in enumerate(ckpt_dirs, start=1):
        st_dir = ckpt.with_name(f"{ckpt.name}-st")
        convert_to_sentence_transformer(str(ckpt), str(st_dir))
        metrics = run_mteb(str(st_dir), TASKS)

        wandb.log({f"epoch{idx}/{k}": v for k, v in metrics.items()},
                  step=idx)

    run.finish()