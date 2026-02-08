# train_pl_student_kd.py
# KD: student (xtremedistil) -> teacher embeddings (768) stored in parquet
#
# Parquet columns:
#   - english: str
#   - non_english: str  (Polish)
#   - label: list[float] length 768  (teacher embedding)
#
# Model (standard Sentence-Transformers pipeline, saved to modules.json):
#   Transformer(student) -> Pooling(mean) -> Dense(256 -> 768)
#
# Training:
#   - streaming load from parquet (no full RAM load)
#   - eval: random eval_size base rows (Polish side) via shuffle + take
#   - bf16
#   - default: two phases (Dense-only, then full model)
#   - phase2 uses ALL remaining training data exactly once (1 epoch over train split)
#
# Windows-friendly:
#   - multiprocessing.freeze_support()
#   - dataloader_num_workers defaults to 0

from __future__ import annotations

import json
import time
from transformers import TrainerCallback
import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Iterator, List

from itertools import islice
import torch
from torch import nn

from datasets import load_dataset, Dataset, IterableDataset
from sentence_transformers import SentenceTransformer, models, SentenceTransformerTrainer
from sentence_transformers.losses import MSELoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import numpy as np
from transformers.trainer_callback import PrinterCallback


from datasets import Features, Value

def decode_label_prefix(row: Dict[str, Any], target_dim: int) -> List[float]:
    lab = row["label"]
    dtype = row.get("label_dtype", None)

    # 1) weź prefiks bez materializacji całej listy, gdy się da
    if hasattr(lab, "__len__") and len(lab) < target_dim:
        raise ValueError(f"Label dim mismatch: got {len(lab)}, expected >= {target_dim}")

    if hasattr(lab, "shape"):          # np.ndarray
        lab_prefix = lab[:target_dim]
    elif hasattr(lab, "slice"):        # czasem Arrow ma slice()
        lab_prefix = lab.slice(0, target_dim)
    else:                               # python list / iterowalne
        # UWAGA: to kopiuje tylko target_dim elementów, nie 2560
        if isinstance(lab, list):
            lab_prefix = lab[:target_dim]
        else:
            lab_prefix = list(islice(lab, target_dim))

    # 2) dekoduj tylko prefiks
    if dtype == "bf16_u16":
        u16 = np.asarray(lab_prefix, dtype=np.uint16)
        # int16/uint16 -> bf16 (dokładnie tak jak masz), a potem do float32 (jak dotychczas)
        t = torch.from_numpy(u16).view(torch.bfloat16).float()
        return t.numpy().astype(np.float32, copy=False).tolist()

    if dtype == "float16":
        x = np.asarray(lab_prefix, dtype=np.float16)
        return x.astype(np.float32).tolist()

    x = np.asarray(lab_prefix, dtype=np.float32)
    return x.tolist()

try:
    # nowsze datasets mają Array1D
    from datasets import Array1D
    def make_label_feature(dim: int):
        return Array1D(shape=(dim,), dtype="float32")
except ImportError:
    # fallback dla starszych wersji
    from datasets import Sequence
    def make_label_feature(dim: int):
        return Sequence(Value("float32"), length=dim)

def log_eval_baseline(trainer, phase: str, log_path: str):
    """
    Liczy eval przed startem treningu danej fazy i loguje:
      - do konsoli
      - do JSONL (ten sam format co LossToFileCallback)
    """
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", None)

    msg = f"[{phase}] baseline eval_loss = {eval_loss}"
    print(msg)

    # Zapis do JSONL w tym samym stylu
    record = {
        "ts": time.time(),
        "phase": phase,
        "step": int(getattr(trainer.state, "global_step", 0)),
        "event": "baseline_eval",
    }
    if eval_loss is not None:
        record["eval_loss"] = float(eval_loss)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return metrics

class PreciseSentenceTransformerTrainer(SentenceTransformerTrainer):
    """
    Transformers Trainer domyślnie robi round(..., 4) w log().
    Ten wariant loguje bez obcinania (lub z inną precyzją, jeśli ustawisz).
    """

    def __init__(self, *args, log_round_ndigits=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_round_ndigits = log_round_ndigits  # None = nie zaokrąglaj

    def log(self, logs: Dict[str, Any]) -> None:
        if logs is None:
            return

        # konwersja tensor/np -> python
        clean = {}
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            elif isinstance(v, (np.floating, np.integer)):
                v = v.item()
            clean[k] = v

        # epoch jak w oryginalnym Trainer
        if self.state.epoch is not None:
            clean["epoch"] = float(self.state.epoch)

        # opcjonalne zaokrąglenie do N miejsc (domyślnie None => pełna precyzja)
        if self.log_round_ndigits is not None:
            for k, v in list(clean.items()):
                if isinstance(v, float):
                    clean[k] = round(v, self.log_round_ndigits)

        # zapis do historii + callbacki (bez “round(...,4)”)
        output = dict(clean)
        output["step"] = int(self.state.global_step)
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, clean)


class LossToFileCallback(TrainerCallback):
    def __init__(self, filepath: str, phase: str = "", precision: int = 8, print_train: bool = True):
        self.filepath = filepath
        self.phase = phase
        self.precision = precision
        self.print_train = print_train

    def _is_main_process(self, args) -> bool:
        pi = getattr(args, "process_index", 0)
        return pi == 0

    def _write(self, record: dict):
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not self._is_main_process(args):
            return

        step = int(state.global_step)
        rec_base = {"ts": time.time(), "phase": self.phase, "step": step}

        if "loss" in logs:
            loss_val = float(logs["loss"])
            if self.print_train:
                print(f"[{self.phase}] step {step} train_loss={loss_val:.{self.precision}f}")
            rec = dict(rec_base)
            rec["loss"] = loss_val
            rec["loss_fmt"] = f"{loss_val:.{self.precision}f}"
            if "learning_rate" in logs:
                rec["learning_rate"] = float(logs["learning_rate"])
            self._write(rec)

        if "eval_loss" in logs:
            ev = float(logs["eval_loss"])
            rec = dict(rec_base)
            rec["eval_loss"] = ev
            rec["eval_loss_fmt"] = f"{ev:.{self.precision}f}"
            self._write(rec)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or not self._is_main_process(args):
            return
        if "eval_loss" in metrics:
            step = int(state.global_step)
            ev = float(metrics["eval_loss"])
            rec = {"ts": time.time(), "phase": self.phase, "step": step, "event": "evaluate"}
            rec["eval_loss"] = ev
            rec["eval_loss_fmt"] = f"{ev:.{self.precision}f}"
            self._write(rec)



def list_parquet_files(chunks_dir: Path) -> List[str]:
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks dir not found: {chunks_dir}")

    # Common cases: *.parquet; sometimes users keep extra suffixes
    exts = (".parquet", ".parquet.zst", ".parquet.zstd")
    files = [str(p) for p in chunks_dir.rglob("*") if p.is_file() and p.name.lower().endswith(exts)]

    if not files:
        raise FileNotFoundError(f"No parquet files found under: {chunks_dir}")
    return files


def count_parquet_rows(parquet_files: List[str]) -> int:
    # Read only metadata -> fast, no full scan, no RAM blow-up
    import pyarrow.parquet as pq

    total = 0
    for f in parquet_files:
        pf = pq.ParquetFile(f)
        total += pf.metadata.num_rows
    return total


def distill_generator(source, include_english, include_polish, target_dim):
    for row in source:
        label = decode_label_prefix(row, target_dim=target_dim)
        if include_english:
            yield {"sentence": row["english"], "label": label}
        if include_polish:
            yield {"sentence": row["non_english"], "label": label}




def build_student(student_name: str, max_seq_length: int, out_emb_dim: int | None, adapter_hidden_dim: int) -> SentenceTransformer:
    transformer = models.Transformer(student_name, max_seq_length=max_seq_length)
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")

    student_dim = pooling.get_sentence_embedding_dimension()  # np. 256
    final_dim = out_emb_dim if out_emb_dim is not None else student_dim

    # Adapter MLP (zawsze obecny)
    dense1 = models.Dense(
        in_features=student_dim,
        out_features=adapter_hidden_dim,
        activation_function=nn.GELU(),
    )
    dense2 = models.Dense(
        in_features=adapter_hidden_dim,
        out_features=final_dim,
        activation_function=nn.Identity(),
    )

    return SentenceTransformer(modules=[transformer, pooling, dense1, dense2])

def build_optimizer_with_param_groups(model: SentenceTransformer, lr_backbone: float, lr_head: float, weight_decay: float = 0.01):
    head_params, backbone_params = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # if name.startswith("0.auto_model.") or ".auto_model." in name:
        if name.startswith("0.auto_model"):
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})

    if not param_groups:
        raise RuntimeError("No trainable parameters found (all requires_grad=False?).")

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)





def freeze_all_but_last_k_layers_and_adapter(model: SentenceTransformer, k: int) -> None:
    # zamrażamy wszystko
    for p in model.parameters():
        p.requires_grad = False

    auto_model = model[0].auto_model

    # wybór listy bloków
    if hasattr(auto_model, "transformer") and hasattr(auto_model.transformer, "layer"):
        layers = auto_model.transformer.layer
    elif hasattr(auto_model, "encoder") and hasattr(auto_model.encoder, "layer"):
        layers = auto_model.encoder.layer
    else:
        raise RuntimeError("Unknown transformer layer structure; add a case for this backbone.")

    # odmrażamy ostatnie k bloków transformera (k może być 0)
    if k > 0:
        for layer in layers[-k:]:
            for p in layer.parameters():
                p.requires_grad = True

    # odmrażamy adapter (zakładamy: 0=Transformer, 1=Pooling, 2+=adapter)
    for idx in range(2, len(model)):
        for p in model[idx].parameters():
            p.requires_grad = True



def unfreeze_all(model: SentenceTransformer) -> None:
    for p in model.parameters():
        p.requires_grad = True


def make_args(
    output_dir: str,
    max_steps: int,
    train_bs: int,
    eval_bs: int,
    grad_accum: int,
    eval_steps: int,
    save_steps: int,
    save_total_limit: int,
    dataloader_num_workers: int,
    seed: int,
    warmup_ratio: float = 0.0, lr_scheduler_type: str = "linear", max_grad_norm: float = 1.0
) -> SentenceTransformerTrainingArguments:
    # transformers / ST use eval_strategy in newer versions
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,

        max_steps=max_steps,

        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,

        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,

        logging_strategy="steps",
        logging_steps=max(1, eval_steps // 2),

        bf16=True,
        fp16=False,
        bf16_full_eval=True,

        remove_unused_columns=False,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,

        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,

        seed=seed,
    )


def compute_steps_for_one_epoch(
    total_base_rows: int,
    eval_size: int,
    examples_per_row: int,
    train_batch_size: int,
    grad_accum: int,
) -> int:
    train_rows = max(0, total_base_rows - eval_size)
    total_train_examples = train_rows * examples_per_row
    denom = max(1, train_batch_size * grad_accum)
    return max(1, math.ceil(total_train_examples / denom))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--parquet-chunks-dir",
        type=str,
        required=True,
        help='Path to parquet chunks dir, e.g. "embedding_dataset_cache/.../parquet_zstd/chunks"',
    )
    ap.add_argument("--output-dir", type=str, required=True)

    ap.add_argument("--student", type=str, default="microsoft/xtremedistil-l6-h256-uncased")
    ap.add_argument("--max-seq-length", type=int, default=256)

    ap.add_argument("--eval-size", type=int, default=2000)
    ap.add_argument("--shuffle-buffer-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--include-english", action="store_true", help="Also train english -> label (optional).")
    ap.add_argument(
        "--include-polish",
        dest="include_polish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train polish(non_english) -> label (default: True).",
    )

    # DEFAULT = two phases, can be disabled with --no-two-phase
    ap.add_argument(
        "--two-phase",
        dest="two_phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Two phases: Dense-only -> full model (default: True). Use --no-two-phase to disable.",
    )

    # Phase 1 (Dense-only) defaults: eff BS ~256 (64 * 4), LR 2e-3, 5k steps
    ap.add_argument("--phase1-train-batch-size", type=int, default=128)
    ap.add_argument("--phase1-eval-batch-size", type=int, default=128)
    ap.add_argument("--phase1-grad-accum", type=int, default=2)
    ap.add_argument("--phase1-lr", type=float, default=5e-5)
    ap.add_argument("--phase1-steps", type=int, default=30000)

    # Phase 2 (full) defaults: eff BS ~128 (64 * 2), LR 3e-5, 1 epoch over all data
    ap.add_argument("--phase2-train-batch-size", type=int, default=128)
    ap.add_argument("--phase2-eval-batch-size", type=int, default=128)
    ap.add_argument("--phase2-grad-accum", type=int, default=1)
    ap.add_argument("--phase2-lr-backbone", type=float, default=5e-6)
    ap.add_argument("--phase2-lr-head", type=float, default=5e-5)
    ap.add_argument("--phase2-weight-decay", type=float, default=0.01)

    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--lr-scheduler-type", type=str, default="cosine")  # "linear" też ok
    ap.add_argument("--max-grad-norm", type=float, default=1.0)

    ap.add_argument(
        "--phase2-epochs",
        type=int,
        default=3,
        help="How many full passes over the training split in phase2 (default: 3).",
    )
    ap.add_argument(
        "--phase2-max-steps",
        type=int,
        default=0,
        help="Override computed max_steps for phase2. 0 means auto (use all data).",
    )

    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--save-total-limit", type=int, default=2)

    ap.add_argument("--dataloader-num-workers", type=int, default=0)
    ap.add_argument(
        "--phase2-save-each-epoch",
        dest="phase2_save_each_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a checkpoint at the end of each epoch in phase2 (default: True). Use --no-phase2-save-each-epoch to disable.",
    )
    ap.add_argument("--phase1-last-k-layers", type=int, default=0,
                    help="Phase1: train only last K transformer blocks (default: 0).")
    ap.add_argument(
        "--out-emb-dim",
        type=int,
        default=None,
        help="Docelowy wymiar embeddingu. Jeśli nie podasz, zostaje wymiar ucznia (np. 256).",
    )
    ap.add_argument(
        "--adapter-hidden-dim",
        type=int,
        default=512,
        help="Wymiar ukryty w adapterze MLP (domyślnie 512).",
    )

    args = ap.parse_args()

    model = build_student(args.student, args.max_seq_length, out_emb_dim=args.out_emb_dim,
                          adapter_hidden_dim=args.adapter_hidden_dim)
    target_dim = model.get_sentence_embedding_dimension()

    FEATURES = Features({
        "sentence": Value("string"),
        "label": make_label_feature(target_dim),
    })
    chunks_dir = Path(args.parquet_chunks_dir)
    parquet_files = list_parquet_files(chunks_dir)

    # We need row count to translate "use all data" (1 epoch) -> max_steps for iterable training
    total_rows = count_parquet_rows(parquet_files)

    # Streaming load (no full RAM load)
    base = load_dataset(
        "parquet",
        data_files={"train": parquet_files},
        split="train",
        streaming=True,
    )

    # Shuffle BEFORE take/skip split (HF recommendation for streaming splits)
    shuffled = base.shuffle(buffer_size=args.shuffle_buffer_size, seed=args.seed)

    eval_source = shuffled.take(args.eval_size)
    train_source = shuffled.skip(args.eval_size)



    # Eval must implement __len__ -> materialize small eval set into memory Dataset
    eval_rows = list(eval_source)
    eval_examples = [
        {"sentence": r["non_english"], "label": decode_label_prefix(r, target_dim)}
        for r in eval_rows
    ]
    eval_dataset = Dataset.from_list(eval_examples)

    loss = MSELoss(model)

    loss_log_path = str(Path(args.output_dir) / "loss_logs" / "losses.jsonl")

    def make_train_iterable() -> IterableDataset:
        return IterableDataset.from_generator(
            distill_generator,
            gen_kwargs={
                "source": train_source,
                "include_english": bool(args.include_english),
                "include_polish": bool(args.include_polish),
                "target_dim": int(target_dim),
            },
            features=FEATURES,
        )

    examples_per_row = (1 if args.include_english else 0) + (1 if args.include_polish else 0)
    if examples_per_row <= 0:
        raise ValueError("At least one of --include-english / --include-polish must be enabled.")

    # Phase2: "use all data" => compute steps for 1 epoch over training split, optionally multiply by epochs
    auto_steps_1epoch = compute_steps_for_one_epoch(
        total_base_rows=total_rows,
        eval_size=args.eval_size,
        examples_per_row=examples_per_row,
        train_batch_size=args.phase2_train_batch_size,
        grad_accum=args.phase2_grad_accum,
    )
    auto_phase2_steps = auto_steps_1epoch * max(1, args.phase2_epochs)
    phase2_steps = args.phase2_max_steps if args.phase2_max_steps and args.phase2_max_steps > 0 else auto_phase2_steps

    # If enabled: save exactly at the end of each epoch in phase2
    phase2_save_steps = auto_steps_1epoch if args.phase2_save_each_epoch else args.save_steps

    # Ensure we don't delete epoch checkpoints by accident (keep at least all epochs)
    phase2_save_total_limit = (
        max(args.save_total_limit, args.phase2_epochs) if args.phase2_save_each_epoch else args.save_total_limit
    )

    if args.two_phase:
        # Phase 1: Dense only (quick adaptation of projection 256 -> 768)
        freeze_all_but_last_k_layers_and_adapter(model, k=args.phase1_last_k_layers)

        train_dataset_1 = make_train_iterable()
        args1 = make_args(
            output_dir=str(Path(args.output_dir) / "phase1_last_layers"),
            max_steps=args.phase1_steps,
            train_bs=args.phase1_train_batch_size,
            eval_bs=args.phase1_eval_batch_size,
            grad_accum=args.phase1_grad_accum,
            eval_steps=min(args.eval_steps, 200) ,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            dataloader_num_workers=args.dataloader_num_workers,
            seed=args.seed,
        )

        trainer1 = SentenceTransformerTrainer(
            model=model,
            args=args1,
            train_dataset=train_dataset_1,
            eval_dataset=eval_dataset,
            loss=loss,
            tokenizer=model.tokenizer,
        )
        trainer1.remove_callback(PrinterCallback)
        trainer1.add_callback(LossToFileCallback(loss_log_path, phase="phase1", precision=8, print_train=True))
        log_eval_baseline(trainer1, phase="phase1", log_path=loss_log_path)
        trainer1.train()
        trainer1.evaluate()

        # Phase 2: full model, one full pass over train split by default
        unfreeze_all(model)

        train_dataset_2 = make_train_iterable()
        args2 = make_args(
            output_dir=str(Path(args.output_dir) / "phase2_full"),
            max_steps=phase2_steps,
            train_bs=args.phase2_train_batch_size,
            eval_bs=args.phase2_eval_batch_size,
            grad_accum=args.phase2_grad_accum,
            eval_steps=args.eval_steps,
            save_steps=phase2_save_steps,
            save_total_limit=phase2_save_total_limit,
            dataloader_num_workers=args.dataloader_num_workers,
            seed=args.seed,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            max_grad_norm=args.max_grad_norm,
        )

        optimizer = build_optimizer_with_param_groups(
            model,
            lr_backbone=args.phase2_lr_backbone,
            lr_head=args.phase2_lr_head,
            weight_decay=args.phase2_weight_decay,
        )

        trainer2 = SentenceTransformerTrainer(
            model=model,
            args=args2,
            train_dataset=train_dataset_2,
            eval_dataset=eval_dataset,
            loss=loss,
            tokenizer=model.tokenizer,
            optimizers=(optimizer, None),
        )
        print("Optimizer LRs:", [g["lr"] for g in optimizer.param_groups])
        trainer2.remove_callback(PrinterCallback)
        trainer2.add_callback(LossToFileCallback(loss_log_path, phase="phase2", precision=8, print_train=True))
        log_eval_baseline(trainer2, phase="phase2", log_path=loss_log_path)
        trainer2.train()
        trainer2.evaluate()

    else:
        # Single phase: full model, one full pass over train split by default
        unfreeze_all(model)

        train_dataset = make_train_iterable()
        args_single = make_args(
            output_dir=str(Path(args.output_dir) / "single_phase"),
            lr=args.phase2_lr,
            max_steps=phase2_steps,
            train_bs=args.phase2_train_batch_size,
            eval_bs=args.phase2_eval_batch_size,
            grad_accum=args.phase2_grad_accum,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            dataloader_num_workers=args.dataloader_num_workers,
            seed=args.seed,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args_single,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            tokenizer=model.tokenizer,
        )
        trainer.train()
        trainer.evaluate()

    # Final save: writes standard SentenceTransformer layout incl. modules.json
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    print(f"Saved final SentenceTransformer to: {final_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
