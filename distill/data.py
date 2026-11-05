"""Unlabeled text corpus for distillation: JSON lines with a text field, length-bucketed batches.

The corpus is held in memory as a list of strings (1.5 M passages of ~450 characters are about 1 GB
of Python strings), because the data drive is a spinning disk and random seeks would dominate.
Use ``max_items`` for probes. Batches group texts of similar character length (a proxy for the token
length) so that padding stays small: a seeded shuffle, mega-buckets of ``bucket_batches`` batches
sorted by length, batches cut inside each bucket, then the batches shuffled.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CorpusStats:
    files: int
    items: int
    chars: int
    by_kind: dict[str, int]
    by_source: dict[str, int]


class TextCorpus:
    def __init__(
        self,
        paths: list[Path],
        field: str = "text",
        kinds: set[str] | None = None,
        max_items: int = 0,
        min_chars: int = 1,
        seed: int = 42,
    ) -> None:
        self.texts: list[str] = []
        by_kind: dict[str, int] = {}
        by_source: dict[str, int] = {}
        chars = 0
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = str(row.get("kind", "passage"))
                    if kinds is not None and kind not in kinds:
                        continue
                    text = row.get(field)
                    if not isinstance(text, str) or len(text) < min_chars:
                        continue
                    self.texts.append(text)
                    chars += len(text)
                    by_kind[kind] = by_kind.get(kind, 0) + 1
                    source = str(row.get("source", Path(path).stem))
                    by_source[source] = by_source.get(source, 0) + 1
        if max_items and len(self.texts) > max_items:
            rng = random.Random(seed)
            keep = sorted(rng.sample(range(len(self.texts)), max_items))
            self.texts = [self.texts[i] for i in keep]
            chars = sum(len(t) for t in self.texts)
            by_kind = {"sampled": len(self.texts)}
            by_source = {"sampled": len(self.texts)}
        self.stats = CorpusStats(files=len(paths), items=len(self.texts), chars=chars, by_kind=by_kind, by_source=by_source)

    def __len__(self) -> int:
        return len(self.texts)

    def batches(self, batch_size: int, seed: int, epoch: int, bucket_batches: int = 50) -> list[list[int]]:
        return length_bucketed_batches([len(t) for t in self.texts], batch_size, seed + epoch, bucket_batches)

    def iter_batches(self, batch_size: int, seed: int, epoch: int, bucket_batches: int = 50) -> Iterator[list[str]]:
        for batch in self.batches(batch_size, seed, epoch, bucket_batches):
            yield [self.texts[i] for i in batch]


def length_bucketed_batches(lengths: list[int], batch_size: int, seed: int, bucket_batches: int = 50) -> list[list[int]]:
    """Indices grouped into batches of similar length; every index appears exactly once."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    bucket = max(batch_size * max(bucket_batches, 1), batch_size)
    batches: list[list[int]] = []
    for start in range(0, len(order), bucket):
        chunk = sorted(order[start : start + bucket], key=lambda i: lengths[i])
        for b in range(0, len(chunk), batch_size):
            batches.append(chunk[b : b + batch_size])
    rng.shuffle(batches)
    return batches
