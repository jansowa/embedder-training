import argparse
import itertools
import subprocess
import sys
from pathlib import Path
import shutil
from dataclasses import dataclass
from time import time

import mteb
import wandb
import yaml
import os

from convert_utils import convert_to_sentence_transformer, run_mteb, run_pirb, ensure_sentence_transformer

parser = argparse.ArgumentParser(description="Grid launcher for FlagEmbedding fine-tuning.")

parser.add_argument(
    "--mode",
    choices=["train", "benchmark"],
    default="train",
    help=(
        "Tryb działania: "
        "'train' - trenowanie + opcjonalne benchmarki (domyślnie); "
        "'benchmark' - tylko benchmarki istniejących modeli/checkpointów."
    ),
)

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
    "--remove_checkpoints",
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
    help='Typ PIRB do uruchomienia: "tiny", "small" lub "all" (domyślnie: tiny).',
)

# TODO: add pooling type to benchmark-target ("mean" by default)
parser.add_argument(
    "--benchmark-target",
    dest="benchmark_targets",
    action="append",
    help=(
        "[benchmark mode] Cel benchmarku w formacie "
        "'MODEL_ALBO_ŚCIEŻKA::QUERY_INSTRUCTION'. "
        "Przykłady:\n"
        "  --benchmark-target 'BAAI/bge-small-en-v1.5::Represent this sentence for searching relevant passages: '\n"
        "  --benchmark-target 'runs::zapytanie: '\n"
        "Jeśli część po '::' będzie pusta, użyty zostanie --default-query-instruction."
    ),
)
parser.add_argument(
    "--default-query-instruction",
    default="",
    help=(
        "Domyślna query_instruction_for_retrieval używana wtedy, "
        "gdy w --benchmark-target nie podano części po '::'. "
        "Domyślnie pusty string (brak prefiksu)."
    ),
)

args = parser.parse_args()

# ----------------- KONFIG / GRID ----------------- #

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
    print(
        f"[WARN] File {args.grid_configuration_file} not found; using built-in defaults.",
        file=sys.stderr,
    )
    grid_yaml = {}

GRID = {
    "architectures": grid_yaml.get("architectures", DEFAULT_GRID["architectures"]),
    "hparams": grid_yaml.get("hparams", DEFAULT_GRID["hparams"]),
}

# TODO: extract from this place
query_instruction_for_retrieval_default = "zapytanie: "
STATIC_ARGS = {
    "cache_dir": "./cache/model",
    "train_data": "./dataset-no_in_batch_neg",
    "cache_path": "./cache/data",
    "train_group_size": 6,
    "query_max_len": 512,
    "passage_max_len": 512,
    "pad_to_multiple_of": 8,
    "query_instruction_for_retrieval": query_instruction_for_retrieval_default,
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

def log_benchmarks_for_st_model(
    st_model_dir,
    epoch_idx,
    query_instruction,
):
    """
    Wspólna funkcja do odpalania MTEB + PIRB dla modelu SentenceTransformer
    i logowania wyników do W&B.
    """
    prefix = f"epoch{epoch_idx}/" if epoch_idx is not None else ""

    metrics_to_log = {}

    if args.run_mteb:
        metrics_mteb = run_mteb(st_model_dir, TASKS)
        metrics_to_log.update({f"{prefix}{k}": v for k, v in metrics_mteb.items()})

    if args.run_pirb:
        metrics_pirb = run_pirb(
            st_model_dir,
            query_instruction_for_retrieval=query_instruction,
            scope=args.pirb_scope,
        )
        metrics_to_log.update({f"{prefix}{k}": v for k, v in metrics_pirb.items()})

    if metrics_to_log:
        wandb.log(metrics_to_log, step=epoch_idx or 0)


# ----------------- TRYB: TRENING ----------------- #

def run_training_mode():
    for arch, cfg in itertools.product(GRID["architectures"], GRID["hparams"]):
        full_args = {**STATIC_ARGS, **cfg}

        lr = full_args.get("learning_rate")
        epochs = full_args.get("num_train_epochs")
        dataset_path = full_args.get("train_data")
        safe_arch = arch.replace("/", "_").replace(".", "_")
        run_name = f"{safe_arch}-{lr}lr-{epochs}ep-{dataset_path}-{int(time())}"

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

        ckpt_dirs = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
        )

        # Benchmarkowanie po trenowaniu (jak dotychczas), ale z wykorzystaniem wspólnej funkcji.
        qi = full_args.get(
            "query_instruction_for_retrieval",
            query_instruction_for_retrieval_default,
        )

        if not ckpt_dirs:
            st_dir = output_dir / "base-st"
            convert_to_sentence_transformer(arch, str(st_dir), pooling_method=full_args.get("sentence_pooling_method"))
            log_benchmarks_for_st_model(
                str(st_dir.resolve()),
                epoch_idx=0,
                query_instruction=qi,
            )
        else:
            for idx, ckpt in enumerate(ckpt_dirs, start=1):
                st_dir = ckpt.with_name(f"{ckpt.name}-st")
                convert_to_sentence_transformer(str(ckpt), str(st_dir), pooling_method=full_args.get("sentence_pooling_method"))
                log_benchmarks_for_st_model(
                    str(st_dir.resolve()),
                    epoch_idx=idx,
                    query_instruction=qi,
                )

        if args.remove_checkpoints:
            print(f"[INFO] Usuwanie checkpointów z {output_dir}...", flush=True)
            for ckpt_dir in output_dir.glob("checkpoint-*"):
                shutil.rmtree(ckpt_dir)

        run.finish()


# ----------------- TRYB: BENCHMARK ----------------- #

@dataclass
class BenchmarkTarget:
    raw: str
    model_or_path: str
    query_instruction: str


def parse_benchmark_target(raw, default_qi):
    """
    Oczekuje formatu 'MODEL_OR_PATH::QUERY_INSTRUCTION'.
    Jeśli '::' nie wystąpi, QUERY_INSTRUCTION = default_qi.
    """
    if "::" in raw:
        model_part, qi_part = raw.split("::", 1)
        model_part = model_part.strip()
        # nie stripujemy qi_part – pozostawiamy potencjalne spacje / dwukropek na końcu
        if qi_part == "":
            qi_part = default_qi
    else:
        model_part = raw.strip()
        qi_part = default_qi

    return BenchmarkTarget(
        raw=raw,
        model_or_path=model_part,
        query_instruction=qi_part,
    )


def iter_benchmark_targets_from_args(args):
    if not args.benchmark_targets:
        return []
    return [
        parse_benchmark_target(raw, args.default_query_instruction)
        for raw in args.benchmark_targets
    ]


def run_benchmark_for_flagembedding_run(run_dir, query_instruction):
    """
    Benchmarkuje jeden katalog runu z FlagEmbedding:
      - jeśli istnieje base-st -> traktujemy jako epoch0,
      - dla każdego checkpoint-*:
          * ensure_sentence_transformer(...) robi ewentualną konwersję,
          * odpalamy MTEB/PIRB i logujemy z prefixem epochX/.
    """
    print(f"Query instruction for benchmark: {query_instruction=}")
    run_dir = Path(run_dir)
    ckpt_dirs = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: p.stat().st_mtime,
    )
    base_st = run_dir / "base-st"

    if not ckpt_dirs and not base_st.exists():
        print(f"[WARN] {run_dir} nie wygląda na run FlagEmbedding (brak checkpoint-* i base-st).")
        return

    safe_name = run_dir.name.replace("/", "_").replace(".", "_")
    run_name = f"{safe_name}-eval"

    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "mode": "benchmark",
            "source_dir": str(run_dir.resolve()),
            "query_instruction_for_retrieval": query_instruction,
        },
    )

    # epoch0: base-st jeśli istnieje
    if base_st.exists():
        print(f"[INFO] Benchmarkuję base-st w {run_dir} (QI: {query_instruction!r})")
        log_benchmarks_for_st_model(
            str(base_st.resolve()),
            epoch_idx=0,
            query_instruction=query_instruction,
        )

    # kolejne epoki: checkpoint-*
    start_idx = 1
    for idx, ckpt in enumerate(ckpt_dirs, start=start_idx):
        print(f"[INFO] Benchmarkuję checkpoint {ckpt} jako epoch{idx} (QI: {query_instruction!r})")
        st_dir = ensure_sentence_transformer(str(ckpt))
        log_benchmarks_for_st_model(
            st_dir,
            epoch_idx=idx,
            query_instruction=query_instruction,
        )

    wandb_run.finish()


def run_benchmark_for_single_model(model_name_or_path, query_instruction):
    """
    Benchmark pojedynczego modelu:
      - nazwa HF lub lokalny katalog z modelem.
    """
    st_dir = ensure_sentence_transformer(model_name_or_path)

    safe_name = model_name_or_path.replace("/", "_").replace(".", "_")
    run_name = f"benchmark-{safe_name}"

    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "mode": "benchmark",
            "model": model_name_or_path,
            "query_instruction_for_retrieval": query_instruction,
        },
    )

    print(f"[INFO] Benchmarkuję model {model_name_or_path} (QI: {query_instruction!r})")
    log_benchmarks_for_st_model(
        st_dir,
        epoch_idx=0,
        query_instruction=query_instruction,
    )

    wandb_run.finish()


def run_benchmark_mode():
    targets = iter_benchmark_targets_from_args(args)

    if not targets:
        print(
            "[ERROR] W trybie --mode benchmark musisz podać co najmniej jedno "
            "--benchmark-target 'MODEL_OR_PATH::QUERY_INSTRUCTION'."
        )
        return

    for t in targets:
        p = Path(t.model_or_path)
        qi = t.query_instruction

        if p.exists() and p.is_dir():
            # 1) katalog wygląda jak pojedynczy run FlagEmbedding
            has_ckpt = any(p.glob("checkpoint-*"))
            has_base_st = (p / "base-st").exists()
            if has_ckpt or has_base_st:
                print(f"[INFO] Benchmarkuję run FlagEmbedding: {p} (QI: {qi!r})")
                run_benchmark_for_flagembedding_run(p, query_instruction=qi)
                continue

            # 2) katalog jest kontenerem wielu runów (np. 'runs/')
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            subdirs_with_runs = [
                d
                for d in subdirs
                if any(d.glob("checkpoint-*")) or (d / "base-st").exists()
            ]
            if subdirs_with_runs:
                print(
                    f"[INFO] Benchmarkuję wszystkie runy w katalogu {p} "
                    f"(QI: {qi!r}, liczba runów: {len(subdirs_with_runs)})"
                )
                for run_dir in sorted(subdirs_with_runs):
                    run_benchmark_for_flagembedding_run(run_dir, query_instruction=qi)
                continue

            # 3) katalog, ale nie wygląda jak run(y) – traktujemy jako pojedynczy model
            print(f"[INFO] Benchmarkuję pojedynczy lokalny model: {p} (QI: {qi!r})")
            run_benchmark_for_single_model(str(p), query_instruction=qi)
            continue

        # 4) nieistniejąca ścieżka → traktujemy jako HF ID
        print(f"[INFO] Benchmarkuję model z HF: {t.model_or_path} (QI: {qi!r})")
        run_benchmark_for_single_model(t.model_or_path, query_instruction=qi)


# ----------------- MAIN ----------------- #

if __name__ == "__main__":
    if args.mode == "train":
        run_training_mode()
    else:
        run_benchmark_mode()
