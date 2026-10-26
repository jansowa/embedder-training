#!/usr/bin/env python3
"""Terminate GPU workers that have lost their queue.

When one queue script takes over from another it kills the previous bash and
its direct child (run_experiments.py), but the grandchild (python -m
training.train, run_benchmark.py) is reparented and keeps the 8 GB card, so the
next job OOMs. Every 20 s this scans for python workers of this repository and
terminates those that

  * have no ancestor that is one of our shell scripts (bash scripts/*.sh),
  * have no controlling terminal (so a run started by hand in a terminal is
    never touched),
  * are older than 30 s (so a worker whose parent is still exec-ing is left),
  * do not carry ORPHAN_GUARD_EXEMPT=1 in their environment (set it for ad-hoc
    GPU runs started from a tool shell).

Start: setsid nohup python3 scripts/orphan_guard.py > logs/orphan_guard.out 2>&1 &
"""

from __future__ import annotations

import os
import re
import signal
import time
from pathlib import Path

WORKER = re.compile(r"(\.venv|venv-[a-z0-9-]+)/bin/python.*(training\.train|run_experiments\.py|run_benchmark\.py|"
                    r"compare_splade_sparsity|splade_index_cost|splade_pair_loss|merge_splade_weights)")
LEGIT_ANCESTOR = re.compile(r"bash .*scripts/\S+\.sh")
LOG = Path("logs/orphan_guard.log")
CLK = os.sysconf("SC_CLK_TCK")


def log(msg: str) -> None:
    with LOG.open("a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def stat_fields(pid: int) -> tuple[int, int, int] | None:
    """(ppid, tty_nr, starttime_ticks) or None."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    rest = raw[raw.rindex(")") + 2:].split()
    return int(rest[1]), int(rest[4]), int(rest[19])


def exempt(pid: int) -> bool:
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return True  # not ours to inspect: leave it alone
    return b"ORPHAN_GUARD_EXEMPT=1" in env


def has_legit_ancestor(pid: int) -> bool:
    seen = 0
    while pid > 1 and seen < 64:
        st = stat_fields(pid)
        if st is None:
            return False
        pid = st[0]
        if LEGIT_ANCESTOR.search(cmdline(pid)):
            return True
        seen += 1
    return False


def uptime() -> float:
    return float(Path("/proc/uptime").read_text().split()[0])


def scan() -> None:
    now = uptime()
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        pid = int(entry)
        cmd = cmdline(pid)
        if not WORKER.search(cmd):
            continue
        st = stat_fields(pid)
        if st is None:
            continue
        ppid, tty_nr, start_ticks = st
        age = now - start_ticks / CLK
        if age < 30 or tty_nr != 0 or exempt(pid) or has_legit_ancestor(pid):
            continue
        log(f"orphan {pid} (ppid {ppid}, age {age:.0f}s): {cmd[:140]}")
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(15)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            log(f"could not kill {pid}: {exc}")


def main() -> None:
    os.chdir(Path(__file__).resolve().parent.parent)
    LOG.parent.mkdir(exist_ok=True)
    log(f"guard started (pid {os.getpid()})")
    while True:
        try:
            scan()
        except Exception as exc:  # never die on a transient /proc race
            log(f"scan error: {exc!r}")
        time.sleep(20)


if __name__ == "__main__":
    main()
