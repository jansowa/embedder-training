"""systemd entry point: recover P1 from the newest intact checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from queue_5070_p1_train import ROOT, EXP, PYTHON, STATE, digest, now, save_json
from queue_5070_p1_resume import CONFIG, environment, locate_run, validate_checkpoint

SERVICE_STATE = ROOT / "logs/mlm-p1-systemd-service.json"
BASE_APPROVAL = ROOT / "logs/mlm-p1-resume-approval.json"


def intact_checkpoint(path: Path, *, max_steps: int = 32595) -> dict:
    state = validate_checkpoint(path, expected_max_steps=max_steps)
    weights = path / "model.safetensors"
    with weights.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size > weights.stat().st_size - 8:
            raise ValueError("Incomplete safetensors header")
        header = json.loads(handle.read(header_size))
    data_end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    if 8 + header_size + data_end != weights.stat().st_size:
        raise ValueError("Incomplete safetensors data")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        with zipfile.ZipFile(path / name) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt checkpoint archive: {name}")
    return state


def latest_intact(run: Path) -> tuple[Path, dict]:
    paths = list(run.glob("checkpoint-*")) + list((run / "epoch-checkpoints").glob("epoch-*-step-*"))
    paths = [p for p in paths if p.name.split("-")[-1].isdigit()]
    paths.sort(key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    errors = []
    for path in paths:
        try:
            return path, intact_checkpoint(path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, struct.error) as exc:
            errors.append(f"{path}: {exc}")
    raise ValueError("No intact checkpoint; " + "; ".join(errors))


def recovery_approval(run: Path, checkpoint: Path, state: dict, attempt: str) -> Path:
    base = json.loads(BASE_APPROVAL.read_text())
    if digest(CONFIG) != base["config_sha256"]:
        raise ValueError("Original training config changed")
    approval = dict(base, approved_at=now(), run=str(run), checkpoint=str(checkpoint),
                    resume_step=state["global_step"], resume_epoch=state["epoch"], attempt_id=attempt,
                    checkpoint_sha256={name: digest(checkpoint / name) for name in
                                       ("model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")})
    path = ROOT / f"logs/mlm-p1-systemd-{attempt}-approval.json"
    if path.exists():
        raise ValueError("Recovery attempt already exists")
    save_json(path, approval)
    return path


def run_kl() -> int:
    previous = json.loads(SERVICE_STATE.read_text()) if SERVICE_STATE.exists() else {}
    attempts = previous.get("consecutive_failures", 0) + int(previous.get("status") == "running")
    recovery = {"status": "recovering", "updated_at": now(), "pid": os.getpid(),
                "supervision": "systemd-user", "consecutive_failures": attempts}
    save_json(STATE, recovery)
    save_json(SERVICE_STATE, recovery)
    record = recovery
    if attempts >= 3:
        record.update(status="failed", error="Three consecutive unclean stops or failed recovery attempts")
        save_json(SERVICE_STATE, record)
        save_json(STATE, record)
        return 2
    try:
        run = locate_run()
        checkpoint, state = latest_intact(run)
        attempt = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        approval = recovery_approval(run, checkpoint, state, attempt)
        record = dict(recovery, status="running", resume_step=state["global_step"], checkpoint=str(checkpoint),
                      attempt=attempt, consecutive_failures=attempts, approval=str(approval))
        save_json(SERVICE_STATE, record)
        print(json.dumps(record), flush=True)
        command = [str(PYTHON), str(Path(__file__).with_name("queue_5070_p1_resume.py")), "--approval", str(approval)]
        if state["global_step"] == 32595 and (run / "final/model.safetensors").exists():
            command.append("--finish-only")
        else:
            command.append("--resume-after-evaluation")
        completed = subprocess.run(command, cwd=ROOT, env=environment())
        if completed.returncode:
            raise RuntimeError(f"KL supervisor exited with {completed.returncode}")
        save_json(SERVICE_STATE, dict(record, status="completed", updated_at=now(), consecutive_failures=0))
        return 0
    except Exception as exc:
        attempts += 1
        failed = attempts >= 3
        record = dict(record, status="failed" if failed else "recovering", updated_at=now(),
                      consecutive_failures=attempts, error=str(exc))
        save_json(SERVICE_STATE, record)
        save_json(STATE, record)
        print(json.dumps(record), flush=True)
        return 2 if failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("kl", "ab"), required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.phase == "kl":
        if os.environ.get("P1_TRAINING_APPROVED") != "yes":
            raise RuntimeError("Training approval required")
        result = EXP / "results/data/mlm-p1-v2-training.json"
        if result.exists() and json.loads(result.read_text()).get("status") == "completed":
            saved = json.loads(result.read_text())
            save_json(STATE, {"status": "completed", "updated_at": now(), "supervision": "systemd-user",
                              "best_checkpoint": saved.get("best_checkpoint")})
            print("P1 KL already completed", flush=True)
            return 0
        return run_kl()
    if os.environ.get("P1_AB_APPROVED") != "yes":
        raise RuntimeError("Continuation A/B approval required")
    result = EXP / "results/data/mlm-p1-v2-continuation-ab.json"
    if result.exists() and json.loads(result.read_text()).get("status") == "completed":
        print("Continuation A/B already completed", flush=True)
        return 0
    command = [str(PYTHON), str(Path(__file__).with_name("queue_5070_p1_continue_ab.py"))]
    if (ROOT / "logs/mlm-p1-continue-ab-bound.json").exists():
        command.append("--resume")
    code = subprocess.run(command,
                          cwd=ROOT, env=environment()).returncode
    # Resume requires matching frozen provenance and an intact optimizer checkpoint.
    return 0 if code == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
