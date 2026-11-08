"""Preserve RoBERTa's tied MLM bias across affected Transformers loaders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_decoder_bias_alias(checkpoint: Path) -> bool:
    """Add an equal decoder alias to a newly saved checkpoint, atomically."""
    import torch
    from safetensors.torch import load_file, save_file

    path = checkpoint / "model.safetensors"
    tensors = load_file(str(path), device="cpu")
    bias = tensors.get("lm_head.bias")
    if bias is None:
        return False
    alias = tensors.get("lm_head.decoder.bias")
    if alias is not None:
        if not torch.equal(alias, bias):
            raise ValueError("Conflicting saved MLM bias aliases")
        return False
    tensors["lm_head.decoder.bias"] = bias.clone()
    temporary = path.with_name(path.name + ".bias-compat.tmp")
    save_file(tensors, str(temporary), metadata={"format": "pt"})
    temporary.replace(path)
    return True


def compatible_export(source: Path, target: Path) -> Path:
    """Export weights with both bias names; leave every original file untouched."""
    source = source.resolve()
    signature = {"source": str(source), "weights_sha256": _sha(source / "model.safetensors"),
                 "config_sha256": _sha(source / "config.json"), "compatibility": "explicit-roberta-decoder-bias-v1"}
    manifest = target / "checkpoint-compatibility.json"
    if target.exists():
        recorded = json.loads(manifest.read_text())
        if any(recorded.get(k) != v for k, v in signature.items()) or _sha(target / "model.safetensors") != recorded["export_weights_sha256"]:
            raise ValueError("Existing compatibility export differs from the source")
        return target
    temporary = target.with_name(target.name + ".creating")
    if temporary.exists():
        raise ValueError("Incomplete previous compatibility export")
    temporary.mkdir(parents=True)
    for p in source.iterdir():
        if p.name in {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin"}:
            continue
        if p.name == "model.safetensors":
            shutil.copy2(p, temporary / p.name)
        else:
            (temporary / p.name).symlink_to(p, target_is_directory=p.is_dir())
    add_decoder_bias_alias(temporary)
    signature["export_weights_sha256"] = _sha(temporary / "model.safetensors")
    (temporary / manifest.name).write_text(json.dumps(signature, indent=2) + "\n")
    temporary.rename(target)
    return target


def same_model_source(evaluated: Path, expected: Path) -> bool:
    """Accept an evaluation of the exact source of a verified alias-only export."""
    if evaluated.resolve() == expected.resolve():
        return True
    manifest = expected / "checkpoint-compatibility.json"
    if not manifest.is_file():
        return False
    recorded = json.loads(manifest.read_text())
    return (recorded.get("compatibility") == "explicit-roberta-decoder-bias-v1"
            and Path(recorded["source"]).resolve() == evaluated.resolve()
            and _sha(evaluated / "model.safetensors") == recorded["weights_sha256"]
            and _sha(evaluated / "config.json") == recorded["config_sha256"]
            and _sha(expected / "model.safetensors") == recorded["export_weights_sha256"])


def restore_and_verify_mlm_weights(model, checkpoint: Path) -> dict:
    """Repair the loaded bias, then compare all serialized tensors exactly."""
    import torch
    from safetensors import safe_open

    auto = model[0].auto_model
    repaired = False
    checked = 0
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as saved:
        if "lm_head.bias" in saved.keys():
            expected = saved.get_tensor("lm_head.bias")
            with torch.no_grad():
                auto.get_output_embeddings().bias.copy_(expected)
                auto.lm_head.bias.copy_(expected)
            repaired = True
        actual = auto.state_dict()
        for name in saved.keys():
            if name not in actual or not torch.equal(actual[name].detach().cpu(), saved.get_tensor(name)):
                raise ValueError(f"Loaded MLM tensor differs from the checkpoint: {name}")
            checked += 1
    return {"verified_tensors": checked, "decoder_bias_restored": repaired}


def reload_sparse_model_on_cpu(trainer, checkpoint: Path) -> dict:
    """Copy checkpoint tensors into the existing model without a temporary GPU model."""
    import gc
    import torch

    model = trainer.model
    loaded = model.__class__(str(checkpoint), device="cpu", trust_remote_code=model.trust_remote_code)
    model.load_state_dict(loaded.state_dict())
    del loaded
    gc.collect()
    verification = restore_and_verify_mlm_weights(model, checkpoint)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return verification
