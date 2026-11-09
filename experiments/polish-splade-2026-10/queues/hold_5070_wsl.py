"""Keep a Windows wsl.exe client attached until the approved P1 pipeline ends."""

import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main():
    started = time.monotonic()
    while True:
        kl = read(ROOT / "logs/mlm-p1-train-queue.json")
        ab = read(ROOT / "logs/mlm-p1-ab-queue.json")
        if kl.get("status") == "completed" and ab.get("status") == "completed":
            return 0
        if time.monotonic() - started > 300 and (kl.get("status") == "failed" or ab.get("status") == "failed"):
            return 2
        if (ROOT / "logs/mlm-p1-wsl-keepalive.stop").exists():
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
