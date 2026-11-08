"""Seeded text samples for calibration and benchmarks (jsonl rows with ``text`` / ``kind``, or plain text lines)."""

from __future__ import annotations

import json
import random
from pathlib import Path


def sample_texts(path: str | Path, quotas: dict[str, int], *, seed: int = 0, field: str = "text") -> dict[str, list[str]]:
    """``quotas[kind]`` texts of each kind, a seeded reservoir sample over the whole file, as {kind: texts}.

    The corpora are ordered by source, so a head sample would be one source. Rows without a ``kind`` are
    kind ``"all"``; a quota under ``"all"`` takes every row regardless of kind. Plain ``.txt`` files: one
    text per line, kind ``"all"``. To get disjoint sets (calibration and check texts) sample once with the
    summed quota and split the lists.
    """
    rng = random.Random(seed)
    want = {kind: n for kind, n in quotas.items() if n > 0}
    pools: dict[str, list[str]] = {kind: [] for kind in want}
    seen: dict[str, int] = {kind: 0 for kind in want}
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if path.suffix == ".txt":
                text, kind = line.strip(), "all"
            else:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text, kind = row.get(field), str(row.get("kind", "all"))
            if not (isinstance(text, str) and text.strip()):
                continue
            for key in {kind, "all"}:
                if key not in want:
                    continue
                seen[key] += 1
                pool = pools[key]
                if len(pool) < want[key]:
                    pool.append(text)
                else:
                    j = rng.randrange(seen[key])
                    if j < want[key]:
                        pool[j] = text
    for kind in sorted(pools):
        random.Random(f"{seed}-{kind}").shuffle(pools[kind])
    return pools


def parse_quotas(spec: str) -> dict[str, int]:
    """``"query=256,passage=256"`` -> {"query": 256, "passage": 256}; a bare number means kind ``all``."""
    spec = spec.strip()
    if spec.isdigit():
        return {"all": int(spec)}
    out = {}
    for part in spec.split(","):
        kind, _, n = part.partition("=")
        out[kind.strip()] = int(n)
    return out
