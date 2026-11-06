# v36 (06.11.2026 09:00, old-pyserini regime, nDCG@10): the best candidates against the PIRB leaderboard

old-pyserini = our PIRB fork + pyserini 0.21.0 (float query weights), the regime of the published polish-splade entry (docs/benchmark-vs-leaderboard.md §1). Base sanity re-run: tiny10 61.59 (reference 61.59). The same models in the quantized regime: results/summary_v35.md.


### Średnie per zestaw (old-pyserini, nDCG@10)

| model | tiny10 (10) | heldout8 (8) | heldout2 (5) | wszystkie wspólne (23) | Δ vs base (wspólne) |
|---|---:|---:|---:|---:|---:|
| published polish-splade | 61.53 | 41.06 | 67.80 | 55.77 |  |
| base-splade-wrapped | 61.59 | 41.03 | 67.53 | 55.73 |  |
| soup-v9b | 63.67 | 42.56 | 69.42 | 57.58 | +1.85 |

### tiny10 (old-pyserini, nDCG@10)

| task | published polish-splade | ours base-splade-wrapped | soup-v9b | Δ soup-v9b vs base | best standalone on PIRB |
|---|---:|---:|---:|---:|---|
| arguana-pl | 36.44 | 37.13 | 46.33 | +9.20 | 70.93 (1b-etap3) |
| gpt-exams | 98.62 | 98.64 | 98.71 | +0.07 | 99.60 (llama-embed-nemotron-8b) |
| maupqa-gpt3-cc | 18.41 | 18.43 | 18.53 | +0.10 | 19.28 (silver-retriever-base-v1) |
| maupqa-mkqa | 63.88 | 63.89 | 64.81 | +0.92 | 65.86 (silver-retriever-base-v1) |
| maupqa-poleval2021-pairs | 82.91 | 82.69 | 83.32 | +0.63 | 88.64 (llama-embed-nemotron-8b) |
| maupqa-templates | 75.23 | 75.01 | 75.81 | +0.80 | 79.56 (llama-embed-nemotron-8b) |
| nfcorpus-pl | 29.84 | 30.04 | 32.34 | +2.30 | 41.12 (bm25) |
| poleval-2022-test-A-allegro-faq | 73.37 | 73.75 | 76.87 | +3.12 | 89.36 (F2LLM-v2-1.7B) |
| poleval-2022-test-B-allegro-faq | 71.87 | 71.82 | 73.55 | +1.74 | 87.72 (F2LLM-v2-8B) |
| scifact-pl | 64.70 | 64.50 | 66.46 | +1.96 | 79.67 (stella-pl-retrieval-8k-ft) |
| **mean** | 61.53 | 61.59 | 63.67 | +2.08 |  |

### heldout8 (old-pyserini, nDCG@10)

| task | published polish-splade | ours base-splade-wrapped | soup-v9b | Δ soup-v9b vs base | best standalone on PIRB |
|---|---:|---:|---:|---:|---|
| fiqa-pl | 26.67 | 26.86 | 30.36 | +3.50 | 52.85 (stella_pl_2etap) |
| maupqa-1z10 | 47.39 | 47.32 | 48.48 | +1.16 | 51.81 (bge-multilingual-gemma2) |
| maupqa-gpt3.5-cc | 20.12 | 20.14 | 20.26 | +0.12 | 20.74 (silver-retriever-base-v1) |
| maupqa-multilingual-NLI | 14.15 | 14.21 | 14.54 | +0.33 | 15.87 (llama-embed-nemotron-8b) |
| maupqa-wiki-def | 64.96 | 64.74 | 66.36 | +1.62 | 75.62 (1b-etap3) |
| mfaq | 62.15 | 62.04 | 63.19 | +1.15 | 70.27 (F2LLM-v2-8B) |
| poleval-2022-test-A-legal-questions | 80.44 | 80.30 | 82.72 | +2.42 | 84.03 (bge-multilingual-gemma2) |
| scidocs-pl | 12.56 | 12.61 | 14.56 | +1.94 | 23.54 (stella_pl_1etap) |
| **mean** | 41.06 | 41.03 | 42.56 | +1.53 |  |

### heldout2 (old-pyserini, nDCG@10)

| task | published polish-splade | ours base-splade-wrapped | soup-v9b | Δ soup-v9b vs base | best standalone on PIRB |
|---|---:|---:|---:|---:|---|
| maupqa-czy-wiesz-v2 | 85.97 | 85.80 | 86.88 | +1.08 | 89.05 (llama-embed-nemotron-8b) |
| maupqa-gpt3.5-wiki | 91.22 | 91.17 | 91.96 | +0.79 | 95.70 (llama-embed-nemotron-8b) |
| maupqa-poquad | 19.57 | 19.49 | 19.24 | -0.24 | 20.44 (silver-retriever-base-v1) |
| quora-pl | 76.76 | 76.56 | 79.33 | +2.77 | 86.26 (stella_pl_1etap) |
| trec-covid-pl | 65.46 | 64.63 | 69.70 | +5.07 | 89.93 (Qwen3-Embedding-8B) |
| **mean** | 67.80 | 67.53 | 69.42 | +1.89 |  |

## Queue state (v36)

```
regime:old-pyserini:base-splade-wrapped:tiny10	ok	61.59 10	743s
regime:old-pyserini:soup-of-ft:tiny10	ok	62.94 10	691s
regime:old-pyserini:base-splade-wrapped:heldout8	ok	41.03 8	3515s
regime:old-pyserini:soup-of-ft:heldout8	ok	42.11 8	3273s
regime:old-pyserini:ck-warmup-s:tiny10	ok	62.16 10	785s
regime:old-pyserini:soup-v2b:tiny10	ok	63.0 10	666s
regime:old-pyserini:ck-warmup-s:heldout8	ok	41.72 8	4485s
regime:old-pyserini:ck-warmup-s-len256:tiny10	ok	62.95 10	714s
regime:old-pyserini:base-splade-wrapped:heldout2_czywiesz	ok	85.8 1	1322s
regime:old-pyserini:base-splade-wrapped:heldout2_gpt35wiki	ok	91.17 1	987s
regime:old-pyserini:base-splade-wrapped:heldout2_quora	ok	76.56 1	958s
regime:old-pyserini:base-splade-wrapped:heldout2_poquad	ok	19.49 1	2294s
regime:old-pyserini:base-splade-wrapped:heldout2_treccovid	ok	64.63 1	1386s
regime:old-pyserini:soup-of-ft:heldout2_czywiesz	ok	86.31 1	1184s
regime:old-pyserini:soup-of-ft:heldout2_gpt35wiki	ok	91.53 1	891s
regime:old-pyserini:soup-of-ft:heldout2_quora	ok	79.11 1	1016s
regime:old-pyserini:soup-of-ft:heldout2_poquad	ok	19.48 1	2109s
regime:old-pyserini:soup-of-ft:heldout2_treccovid	ok	68.57 1	1338s
regime:old-pyserini:soup-v2b:heldout8	ok	42.06 8	3010s
regime:old-pyserini:arm-cluster-like-fiqa:tiny10	ok	62.85 10	760s
regime:old-pyserini:arm-cluster-like-fiqa:heldout8	ok	41.76 8	3985s
regime:old-pyserini:arm-fiqa-plus-squadv2:tiny10	ok	62.61 10	720s
regime:old-pyserini:arm-fiqa-plus-squadv2:heldout8	ok	41.74 8	3791s
regime:old-pyserini:soup-v2b:heldout2_czywiesz	ok	86.65 1	1081s
regime:old-pyserini:soup-v2b:heldout2_gpt35wiki	ok	91.92 1	826s
regime:old-pyserini:soup-v2b:heldout2_quora	ok	79.04 1	978s
regime:old-pyserini:soup-v2b:heldout2_poquad	ok	19.63 1	1840s
regime:old-pyserini:soup-v2b:heldout2_treccovid	ok	67.79 1	1300s
regime:old-pyserini:base-splade-wrapped-recheck:tiny10	ok	61.59 10	748s
sanity:old-pyserini:base	ok	61.59	reference 61.59
regime:old-pyserini:soup-v9b:tiny10	ok	63.67 10	758s
regime:old-pyserini:soup-v9b:heldout8	ok	42.56 8	4006s
regime:old-pyserini:soup-v9b:heldout2_czywiesz	ok	86.88 1	1366s
regime:old-pyserini:soup-v9b:heldout2_gpt35wiki	ok	91.96 1	1064s
regime:old-pyserini:soup-v9b:heldout2_quora	ok	79.33 1	1265s
regime:old-pyserini:soup-v9b:heldout2_poquad	ok	19.24 1	2524s
regime:old-pyserini:soup-v9b:heldout2_treccovid	ok	69.7 1	1383s
```
