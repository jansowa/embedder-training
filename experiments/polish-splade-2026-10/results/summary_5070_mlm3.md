# Noc 3 — parowany bieg MLM, seed 42

Updated: 2026-11-05T18:47:08+01:00

**137 773 przykładów po bramce i rozwinięciu pozytywów; identyczna lista i kolejność dla M i A.** MS MARCO: fallback `datasets/judged/msmarco-rejected-20000.jsonl` (HF 401). Natywna bramka SQuAD-v2: 56 093 wiersze / 56 437 przykładów. Zatwierdzony plan: nominalnie jedna epoka, max_steps=4306, checkpointy 2150/4306, próg miejsca przed parą 35 GB. M: BF16, A: FP16; oba od oryginalnego MLM, bias shift auto→400, scale=1, LR=2e-5, warmup LR 250, lambda warmup 1000, regulator 436×[0.75,1.05]. Żaden retry nie używa resume. Optymalizator/scheduler są usuwane dopiero po zakończeniu procesu treningu.

| Składnik | Wiersze po bramce | Przykłady |
|---|---:|---:|
| datasets/arms/squadv2-v2-all.jsonl | 56093 | 56437 |
| datasets/judged/msmarco-rejected-20000.jsonl | 20000 | 20000 |
| datasets/judged/msmarco-clean-20000.jsonl | 20000 | 20000 |
| datasets/judged/msmarco-plain-20000.jsonl | 20000 | 20000 |
| datasets/arms/squadv2-fresh-16000.jsonl | 16000 | 16086 |
| datasets/arms/squadv2-2073.jsonl | 2073 | 2089 |
| datasets/arms/fiqa-2073.jsonl | 2073 | 3161 |

| Model | Status | krok | s/krok | peak CUDA / NVIDIA sampled MiB | ETA treningu |
|---|---|---:|---:|---|---|
| M | finished | 4300 | 3.87 | 11540.23 / 13073.00 | — |
| A | finished | 4300 | 2.62 | 9770.77 / 11411.00 | — |

| Model | checkpoint | dims/doc (FP32, 512) | tiny10 nDCG@10 |
|---|---:|---:|---:|
| M | 2150 | 364.27 | 54.16 |
| M | 4306 | 327.85 | 53.75 |
| A | 2150 | 49240.46 | 55.14 |
| A | 4306 | 49671.75 | 56.73 |

**GM s04306: nieporównywalne przy tej gęstości.** Wyłącznie końcowe checkpointy obu modeli w paśmie 327–458; stara krzywa A i E5 s2500 nie są używane do GM.

## Per task — ten sam quantized tiny10, 512 tokenów

| Task | M s2150 | A s2150 | M s4306 | A s4306 | base | mlm-msmarco-ep1 |
|---|---:|---:|---:|---:|---:|---:|
| maupqa-gpt3-cc | 18.13 | 18.62 | 17.83 | 18.96 | 18.43 | 18.69 |
| maupqa-mkqa | 48.77 | 51.18 | 49.36 | 53.95 | 63.92 | 61.66 |
| maupqa-poleval2021-pairs | 70.16 | 73.92 | 72.30 | 76.07 | 82.67 | 80.65 |
| maupqa-templates | 66.45 | 69.64 | 69.03 | 71.02 | 75.02 | 74.17 |
| poleval-2022-test-A-allegro-faq | 64.10 | 60.96 | 60.59 | 64.19 | 73.76 | 71.52 |
| poleval-2022-test-B-allegro-faq | 60.09 | 58.56 | 56.50 | 60.47 | 71.81 | 67.76 |
| arguana-pl | 34.75 | 35.43 | 27.17 | 37.92 | 37.14 | 29.94 |
| nfcorpus-pl | 30.99 | 29.71 | 29.59 | 30.46 | 30.07 | 32.57 |
| scifact-pl | 51.60 | 56.26 | 59.74 | 56.84 | 64.52 | 57.95 |
| gpt-exams | 96.57 | 97.17 | 95.40 | 97.43 | 98.63 | 98.12 |

Referencje: base-splade-wrapped 61.60, mlm-msmarco-ep1 59.29. Stara A: s2500=56.33 przy 297 dims, s5000=55.37 przy 311 dims — tylko odniesienie, stała lambda i dryfująca gęstość.

Krok 0: M s1500=55.63 /356 dims, M s2000=56.61 /404 dims; A s1875=55.70 /391 dims; M−A=+0.91. E5 s2500 wyłączony z werdyktu.

Free C: before pair: 47.23 GB. Training timeouts: 12h, retry from zero 18h. CUDA peak is allocator peak; NVIDIA peak is sampled total card use.
