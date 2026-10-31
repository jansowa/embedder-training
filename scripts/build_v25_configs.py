#!/usr/bin/env python
"""Configs for queue v25 (31.10 -> 1.11 on the 3060 Ti): the new speakleash sets as components.

  mnrl_<set>_2073      single-set screen: <set>-2073 rows alone (gate 23.5 on the gemma score of
                       the English pair, the gate of fiqa-2073 / squadv2-2073; length rules of
                       scripts/build_speakleash_cuts.py), reference recipe, 2 x 16.
  n4x_<set>            the n4 pattern with the new set as the third component:
                       fiqa-2073 + squadv2-2073 + <set>-2073 (datasets/arms/fiqa-squadv2-<set>-6219.jsonl).
  n4_3060              the control: n4's own rows (datasets/arms/mix3-6219.jsonl, MS MARCO-clean
                       as the third component) at 2 x 16 here (n4-5070 ran at 8 x 4 on the 5070 Ti).
  *_cb32               CachedSpladeLoss fallback (true batch 32, mini-batch 2), used after an OOM.

Sets: fever, hotpotqa, nq. Files: configs/next/3060/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v23_configs import FILTER, OUT, config, write  # noqa: E402

RECIPE = "recipe mnrl (cluster-like), 512 tokens, 10 negatives, batch 2 x 16, pooling chunk 8, gate filter"
SETS = ("fever", "hotpotqa", "nq")


def pair(name: str, tag: str, desc: str, data: str) -> None:
    write(name, config(tag, f"{desc}; {RECIPE}.", data, 2, 16, None, FILTER))
    write(f"{name}_cb32", config(tag, f"{desc}; fallback: CachedSpladeLoss at batch 32 (mini-batch 2).", data, 32, 1, 2, FILTER))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        pair(f"mnrl_{s}_2073", f"mnrl-{s}-2073",
             f"Single-set screen: speakleash {s} cut of 2073 gated rows (scripts/build_speakleash_cuts.py)",
             f"datasets/arms/{s}-2073.jsonl")
        pair(f"n4x_{s}", f"n4x-{s}",
             f"n4 pattern with {s} as the third component: fiqa-2073 + squadv2-2073 + {s}-2073",
             f"datasets/arms/fiqa-squadv2-{s}-6219.jsonl")
    pair("n4_3060", "n4-3060", "Control: n4's rows (fiqa-2073 + squadv2-2073 + msmarco-clean-2073) at 2 x 16 on the 3060",
         "datasets/arms/mix3-6219.jsonl")


if __name__ == "__main__":
    main()
