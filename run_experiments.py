import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys


WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mining-tests")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid launcher for training and benchmarks.")
    parser.add_argument(
        "--mode",
        choices=["train", "benchmark"],
        default="train",
        help=(
            "Execution mode: "
            "'train' - training plus optional benchmarks (default); "
            "'benchmark' - benchmark existing models/checkpoints only."
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
    parser.add_argument("--run_mteb", "--run-mteb", dest="run_mteb", action="store_true", help="Run the MTEB benchmark.")
    parser.add_argument(
        "--remove_checkpoints",
        "--remove-checkpoints",
        dest="remove_checkpoints",
        action="store_true",
        help="Remove checkpoint-* directories after the run to free disk space.",
    )
    parser.add_argument("--run_pirb", "--run-pirb", dest="run_pirb", action="store_true", help="Run the PIRB benchmark.")
    parser.add_argument(
        "--pirb_scope",
        "--pirb-scope",
        dest="pirb_scope",
        choices=["tiny", "small", "all"],
        default="tiny",
        help='PIRB scope to run: "tiny", "small", or "all" (default: tiny).',
    )
    parser.add_argument(
        "--benchmark-target",
        dest="benchmark_targets",
        action="append",
        help=(
            "[benchmark mode] Benchmark target in the format "
            "'MODEL_OR_PATH::QUERY_INSTRUCTION'. "
            "If the part after '::' is empty, --default-query-instruction is used."
        ),
    )
    parser.add_argument(
        "--default-query-instruction",
        default="",
        help="Default query_instruction_for_retrieval used for --benchmark-target.",
    )
    return parser


def _load_wandb():
    import wandb

    return wandb


def _load_mteb():
    import mteb

    return mteb


def run_training_mode(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "training.train",
        "--backend",
        "flagembedding",
        "--training-type",
        "embedder",
        "--config",
        args.grid_configuration_file,
        "--benchmark-name",
        args.benchmark_name,
        "--pirb-scope",
        args.pirb_scope,
    ]
    if args.run_mteb:
        cmd.append("--run-mteb")
    if args.run_pirb:
        cmd.append("--run-pirb")
    if args.remove_checkpoints:
        cmd.append("--remove-checkpoints")

    subprocess.run(cmd, check=True)
    return 0


@dataclass
class BenchmarkTarget:
    raw: str
    model_or_path: str
    query_instruction: str


def parse_benchmark_target(raw, default_qi):
    """
    Expect the format 'MODEL_OR_PATH::QUERY_INSTRUCTION'.
    If '::' is missing, QUERY_INSTRUCTION = default_qi.
    """
    if "::" in raw:
        model_part, qi_part = raw.split("::", 1)
        model_part = model_part.strip()
        if qi_part == "":
            qi_part = default_qi
    else:
        model_part = raw.strip()
        qi_part = default_qi

    return BenchmarkTarget(raw=raw, model_or_path=model_part, query_instruction=qi_part)


def iter_benchmark_targets_from_args(args):
    if not args.benchmark_targets:
        return []
    return [parse_benchmark_target(raw, args.default_query_instruction) for raw in args.benchmark_targets]


def log_benchmarks_for_st_model(st_model_dir, epoch_idx, query_instruction, args, tasks):
    """
    Run MTEB and PIRB for a SentenceTransformer model and log results to W&B.
    """
    from convert_utils import run_mteb, run_pirb

    prefix = f"epoch{epoch_idx}/" if epoch_idx is not None else ""
    metrics_to_log = {}

    if args.run_mteb:
        metrics_mteb = run_mteb(st_model_dir, tasks)
        metrics_to_log.update({f"{prefix}{key}": value for key, value in metrics_mteb.items()})

    if args.run_pirb:
        metrics_pirb = run_pirb(
            st_model_dir,
            query_instruction_for_retrieval=query_instruction,
            scope=args.pirb_scope,
        )
        metrics_to_log.update({f"{prefix}{key}": value for key, value in metrics_pirb.items()})

    if metrics_to_log:
        _load_wandb().log(metrics_to_log, step=epoch_idx or 0)


def run_benchmark_for_flagembedding_run(run_dir, query_instruction, args, tasks):
    """
    Benchmark one FlagEmbedding run directory:
      - if base-st exists, treat it as epoch0;
      - for each checkpoint-*:
          * ensure_sentence_transformer(...) performs conversion if needed;
          * run MTEB/PIRB and log with an epochX/ prefix.
    """
    from convert_utils import ensure_sentence_transformer

    print(f"Query instruction for benchmark: {query_instruction=}")
    run_dir = Path(run_dir)
    ckpt_dirs = sorted(run_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    base_st = run_dir / "base-st"

    if not ckpt_dirs and not base_st.exists():
        print(f"[WARN] {run_dir} does not look like a FlagEmbedding run (no checkpoint-* or base-st).")
        return

    safe_name = run_dir.name.replace("/", "_").replace(".", "_")
    run_name = f"{safe_name}-eval"

    wandb_run = _load_wandb().init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "mode": "benchmark",
            "source_dir": str(run_dir.resolve()),
            "query_instruction_for_retrieval": query_instruction,
        },
    )

    if base_st.exists():
        print(f"[INFO] Benchmarking base-st in {run_dir} (QI: {query_instruction!r})")
        log_benchmarks_for_st_model(str(base_st.resolve()), epoch_idx=0, query_instruction=query_instruction, args=args, tasks=tasks)

    for idx, ckpt in enumerate(ckpt_dirs, start=1):
        print(f"[INFO] Benchmarking checkpoint {ckpt} as epoch{idx} (QI: {query_instruction!r})")
        st_dir = ensure_sentence_transformer(str(ckpt))
        log_benchmarks_for_st_model(st_dir, epoch_idx=idx, query_instruction=query_instruction, args=args, tasks=tasks)

    wandb_run.finish()


def run_benchmark_for_single_model(model_name_or_path, query_instruction, args, tasks):
    """
    Benchmark a single model: either a Hugging Face name or a local model directory.
    """
    from convert_utils import ensure_sentence_transformer

    st_dir = ensure_sentence_transformer(model_name_or_path)
    safe_name = model_name_or_path.replace("/", "_").replace(".", "_")
    run_name = f"benchmark-{safe_name}"

    wandb_run = _load_wandb().init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "mode": "benchmark",
            "model": model_name_or_path,
            "query_instruction_for_retrieval": query_instruction,
        },
    )

    print(f"[INFO] Benchmarking model {model_name_or_path} (QI: {query_instruction!r})")
    log_benchmarks_for_st_model(st_dir, epoch_idx=0, query_instruction=query_instruction, args=args, tasks=tasks)
    wandb_run.finish()


def run_benchmark_mode(args, tasks) -> int:
    targets = iter_benchmark_targets_from_args(args)

    if not targets:
        print(
            "[ERROR] In --mode benchmark you must provide at least one "
            "--benchmark-target 'MODEL_OR_PATH::QUERY_INSTRUCTION'."
        )
        return 2

    for target in targets:
        path = Path(target.model_or_path)
        query_instruction = target.query_instruction

        if path.exists() and path.is_dir():
            has_ckpt = any(path.glob("checkpoint-*"))
            has_base_st = (path / "base-st").exists()
            if has_ckpt or has_base_st:
                print(f"[INFO] Benchmarking FlagEmbedding run: {path} (QI: {query_instruction!r})")
                run_benchmark_for_flagembedding_run(path, query_instruction=query_instruction, args=args, tasks=tasks)
                continue

            subdirs = [subdir for subdir in path.iterdir() if subdir.is_dir()]
            subdirs_with_runs = [
                subdir for subdir in subdirs if any(subdir.glob("checkpoint-*")) or (subdir / "base-st").exists()
            ]
            if subdirs_with_runs:
                print(
                    f"[INFO] Benchmarking all runs in directory {path} "
                    f"(QI: {query_instruction!r}, run count: {len(subdirs_with_runs)})"
                )
                for run_dir in sorted(subdirs_with_runs):
                    run_benchmark_for_flagembedding_run(run_dir, query_instruction=query_instruction, args=args, tasks=tasks)
                continue

            print(f"[INFO] Benchmarking single local model: {path} (QI: {query_instruction!r})")
            run_benchmark_for_single_model(str(path), query_instruction=query_instruction, args=args, tasks=tasks)
            continue

        print(f"[INFO] Benchmarking Hugging Face model: {target.model_or_path} (QI: {query_instruction!r})")
        run_benchmark_for_single_model(target.model_or_path, query_instruction=query_instruction, args=args, tasks=tasks)

    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "train":
        return run_training_mode(args)

    tasks = _load_mteb().get_benchmarks(names=[args.benchmark_name]) if args.run_mteb else None
    return run_benchmark_mode(args, tasks)


if __name__ == "__main__":
    raise SystemExit(main())
