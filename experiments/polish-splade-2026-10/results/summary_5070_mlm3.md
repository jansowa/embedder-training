# Night 3: paired-run preflight — not started

Both configs start from their original MLM, use the same ordered input files and seed 42, effective batch 32, bias initialization target 400, scale 1, LR 2e-5, 250 LR warmup steps, lambda warmup 1000 and density reference 436. M uses BF16; A uses FP16. Requested checkpoints: s2500/s5000. No resume is configured.

| Source | Gated rows | Training examples |
|---|---:|---:|
| datasets/arms/squadv2-v2-all.jsonl | 56093 | 56437 |
| datasets/judged/msmarco-rejected-20000.jsonl | 20000 | 20000 |
| datasets/judged/msmarco-clean-20000.jsonl | 20000 | 20000 |
| datasets/judged/msmarco-plain-20000.jsonl | 20000 | 20000 |
| datasets/arms/squadv2-fresh-16000.jsonl | 16000 | 16086 |
| datasets/arms/squadv2-2073.jsonl | 2073 | 2089 |
| datasets/arms/fiqa-2073.jsonl | 2073 | 3161 |

**Total: 137773 examples.** 5000 steps = 160000 presentations = 1.1613 epochs.
Approximate one-epoch step count: 4306.
MS MARCO selection: fallback msmarco-rejected-20000. HF request returned 401; authorized fallback selected.
Free C: space after preparation: 36.377 GB.

## Start blockers

- One epoch versus 5000 steps: 137773 examples, but 5000x32=160000 presentations (1.1613 epochs).
- Disk prerequisite: 36.377 GB free on C:, below required 40 GB.

No GPU training, throughput, peak VRAM or night completion time is reported because the run has not started. Inputs and configs are ready for review; the original stop-before-start instruction is honored.

Step 0: M s1500 = 55.63 at 356 dims; M s2000 = 56.61 at 404 dims; A s1875 = 55.70 at 391 dims. M s2000 − A s1875 = +0.91 nDCG points at similar density; E5 s2500 excluded.
