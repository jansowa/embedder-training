import argparse
import itertools
import subprocess
import sys
from pathlib import Path
import shutil

import mteb
import wandb
import yaml
import os

from convert_utils import convert_to_sentence_transformer, run_mteb, run_pirb

parser = argparse.ArgumentParser(description="Grid launcher for FlagEmbedding fine-tuning.")
parser.add_argument(
    "--grid-configuration-file",
    default="configs/grid.yaml",
    help="Path to YAML file with architectures and hyper-parameters.",
)
parser.add_argument(
    "--benchmark-name",
    default="NanoBEIR",
    help="Name of the MTEB benchmark to use (default: NanoBEIR).",
)
parser.add_argument(
    "--run_mteb",
    action="store_true",
    help="Uruchom benchmark MTEB"
)
parser.add_argument(
    "--remove-checkpoints",
    action="store_true",
    help="Po zakończeniu runu usuń wszystkie katalogi checkpoint-* w celu zwolnienia miejsca na dysku.",
)
parser.add_argument(
    "--run_pirb",
    action="store_true",
    help="Włącz uruchomienie PIRB (flaga logiczna).",
)
parser.add_argument(
    "--pirb_scope",
    choices=["tiny", "small", "all"],
    default="tiny",
    help='Typ PIRB do uruchomienia: "tiny", "small", "medium" lub "all" (domyślnie: tiny).',
)
args = parser.parse_args()

DEFAULT_GRID = {
    "architectures": [
        "answerdotai/ModernBERT-base",
        "answerdotai/ModernBERT-large",
    ],
    "hparams": [
        {"learning_rate": 2.5e-5, "num_train_epochs": 2},
    ],
}

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mining-tests")

try:
    with open(args.grid_configuration_file) as fh:
        grid_yaml = yaml.safe_load(fh) or {}
except FileNotFoundError:
    print(f"[WARN] File {args.grid_configuration_file} not found; using built-in defaults.", file=sys.stderr)
    grid_yaml = {}

GRID = {
    "architectures": grid_yaml.get("architectures", DEFAULT_GRID["architectures"]),
    "hparams": grid_yaml.get("hparams", DEFAULT_GRID["hparams"]),
}

# TODO: extract from this place
query_instruction_for_retrieval = "zapytanie: "
STATIC_ARGS = {
    "cache_dir": "./cache/model",
    "train_data": "./dataset-no_in_batch_neg",
    "cache_path": "./cache/data",
    "train_group_size": 6,
    "query_max_len": 512,
    "passage_max_len": 512,
    "pad_to_multiple_of": 8,
    "query_instruction_for_retrieval": query_instruction_for_retrieval,
    "query_instruction_format": "{}{}",
    "knowledge_distillation": True,
    "fp16": False,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "dataloader_drop_last": True,
    "warmup_ratio": 0.1,
    "gradient_checkpointing": True,
    "deepspeed": "./ds_stage0.json",
    "logging_steps": 1,
    "save_strategy": "epoch",
    "negatives_cross_device": True,
    "temperature": 0.02,
    "sentence_pooling_method": "cls",
    "normalize_embeddings": True,
    "kd_loss_type": "kl_div",
}

TASKS = mteb.get_benchmarks(names=[args.benchmark_name])
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

for arch, cfg in itertools.product(GRID["architectures"], GRID["hparams"]):
    full_args = {**STATIC_ARGS, **cfg}

    lr = full_args.get("learning_rate")
    epochs = full_args.get("num_train_epochs")
    dataset_path = full_args.get("train_data")
    safe_arch = arch.replace("/", "_")
    run_name = f"{safe_arch}-{lr}lr-{epochs}ep-{dataset_path}"

    run = wandb.init(project=WANDB_PROJECT, name=run_name, config={**full_args, "arch": arch})

    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun", "--nproc_per_node", "1", "-m",
        "FlagEmbedding.finetune.embedder.encoder_only.base",
        "--model_name_or_path", arch,
        "--overwrite_output_dir",
        "--output_dir", str(output_dir),
        "--report_to", "wandb",
        "--run_name", run_name,
        "--trust_remote_code", "True"
    ]

    for k, v in full_args.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])

    print(">>> LAUNCH:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.update({
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_NAME": run_name,
        "WANDB_RUN_GROUP": safe_arch,
    })
    subprocess.run(cmd, check=True)

    ckpt_dirs = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)

    if not ckpt_dirs:
        st_dir = output_dir / "base-st"
        convert_to_sentence_transformer(arch, str(st_dir))
        if args.run_mteb:
            metrics_mteb = run_mteb(str(st_dir), TASKS)
            wandb.log({f"epoch0/{k}": v for k, v in metrics_mteb.items()}, step=0)
        if args.run_pirb:
            metrics_pirb = run_pirb(str(st_dir.resolve()),
                                    query_instruction_for_retrieval=query_instruction_for_retrieval,
                                    scope=args.pirb_scope)
            wandb.log({f"epoch0/{k}": v for k, v in metrics_pirb.items()}, step=0)

    for idx, ckpt in enumerate(ckpt_dirs, start=1):
        st_dir = ckpt.with_name(f"{ckpt.name}-st")
        convert_to_sentence_transformer(str(ckpt), str(st_dir))
        if args.run_mteb:
            metrics_mteb = run_mteb(str(st_dir), TASKS)
            wandb.log({f"epoch{idx}/{k}": v for k, v in metrics_mteb.items()}, step=idx)
        if args.run_pirb:
            metrics_pirb = run_pirb(str(st_dir.resolve()), query_instruction_for_retrieval=query_instruction_for_retrieval, scope=args.pirb_scope)
            wandb.log({f"epoch{idx}/{k}": v for k, v in metrics_pirb.items()}, step=idx)

    if args.remove_checkpoints:
        print(f"[INFO] Usuwanie checkpointów z {output_dir}...", flush=True)
        for ckpt_dir in output_dir.glob("checkpoint-*"):
            shutil.rmtree(ckpt_dir)

    run.finish()