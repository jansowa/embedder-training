# Noc 4 — heldout8 nocy 3 i pełny publiczny PIRB

Aktualizacja: 2026-11-06T01:42:22+01:00

| Model s04306 | tiny10 | heldout8 | wym./dok (FP32, 512) |
|---|---:|---:|---:|
| night3-paired-a-s04306 | 56.73 | 39.64 | 459.13 |
| night3-paired-m-s04306 | 53.75 | 37.12 | 327.85 |

A s4306: **nieporównywalne przy tej gęstości** (pasmo 327–458).
Kotwice tiny10: baza 61.60; mlm-msmarco-ep1 59.29.

P1: quantized, batch 8; wiki-trivia 14 h timeout, pozostałe 10 h. Wymagane ≥60 GiB na cache.
Scoring pd400: surowe cosinusy, scale=1; sonda KD czeka na S i zgodę użytkownika.

Leaderboard polish-splade (eksport użytkownika): wiki-trivia 37.4 / 35.0 / 35.9; dbpedia 28.42; hotpotqa 55.17.

### wikitrivia

(no results)

### dbpediapl

(no results)

### hotpotqapl

(no results)



## Scoring PolDense-400M

```
squadv2-v2-all.jsonl: 1563126 texts / 6462s = 241.90 texts/s (wall time including model load). Remaining ETA: 2026-11-06T02:07:08+01:00
Median margin: 0.07597452402114868; rows with negative above positive: 11184

2026-11-06T01:00:48+01:00 squadv2-v2-all elapsed=6499s
130214 rows -> datasets/arms/squadv2-v2-all-tpd400.partial.jsonl; teacher OPI-PIB/PolDense-400M (cosine x 1.00, prefix '[query]: '/''); median margin min(pos)-max(neg) 0.08; rows with a negative above a positive 11184
msmarco-rejected-20000.jsonl: 240000 texts / 539s = 445.27 texts/s (wall time including model load). Remaining ETA: 2026-11-06T01:37:10+01:00
Median margin: 0.04997891187667847; rows with negative above positive: 2143

2026-11-06T01:10:07+01:00 msmarco-rejected-20000 elapsed=548s
20000 rows -> datasets/judged/msmarco-rejected-20000-tpd400.partial.jsonl; teacher OPI-PIB/PolDense-400M (cosine x 1.00, prefix '[query]: '/''); median margin min(pos)-max(neg) 0.05; rows with a negative above a positive 2143
msmarco-clean-20000.jsonl: 240000 texts / 529s = 453.69 texts/s (wall time including model load). Remaining ETA: 2026-11-06T01:37:01+01:00
Median margin: 0.06702286005020142; rows with negative above positive: 653

2026-11-06T01:19:16+01:00 msmarco-clean-20000 elapsed=537s
20000 rows -> datasets/judged/msmarco-clean-20000-tpd400.partial.jsonl; teacher OPI-PIB/PolDense-400M (cosine x 1.00, prefix '[query]: '/''); median margin min(pos)-max(neg) 0.07; rows with a negative above a positive 653
msmarco-plain-20000.jsonl: 240000 texts / 530s = 452.83 texts/s (wall time including model load). Remaining ETA: 2026-11-06T01:37:21+01:00
Median margin: 0.06355997920036316; rows with negative above positive: 966

2026-11-06T01:28:24+01:00 msmarco-plain-20000 elapsed=536s
20000 rows -> datasets/judged/msmarco-plain-20000-tpd400.partial.jsonl; teacher OPI-PIB/PolDense-400M (cosine x 1.00, prefix '[query]: '/''); median margin min(pos)-max(neg) 0.06; rows with a negative above a positive 966
squadv2-fresh-16000.jsonl: 192096 texts / 822s = 233.69 texts/s (wall time including model load). Remaining ETA: 2026-11-06T01:46:00+01:00
Median margin: 0.0975382924079895; rows with negative above positive: 423

2026-11-06T01:42:22+01:00 squadv2-fresh-16000 elapsed=826s
16000 rows -> datasets/arms/squadv2-fresh-16000-tpd400.partial.jsonl; teacher OPI-PIB/PolDense-400M (cosine x 1.00, prefix '[query]: '/''); median margin min(pos)-max(neg) 0.10; rows with a negative above a positive 423

```

## Stan kolejki

```
train:arm-n1-cluster-fiqa-plus-squadv2	FAILED	-	7200s
train:arm-n2-cluster-fiqa-2073	FAILED	-	7200s
train:arm-n3-cluster-fiqa-plus-squadv2-10k	FAILED	-	7200s
train:arm-n4-cluster-mix3-6219	FAILED	-	7200s
train:arm-n5-cluster-fiqa-plus-squadv2-b16	FAILED	-	47s
train:arm-n6-mmse-fiqa-plus-squadv2	FAILED	-	7200s
regime:quantized:soup-v2b:tiny10	ok	62.99 10	425s
train:arm-n1-cluster-fiqa-plus-squadv2	ok	-	13969s
bench:arm-n1-cluster-fiqa-plus-squadv2	ok	62.1	536s
train:arm-n2-cluster-fiqa-2073	ok	-	9572s
bench:arm-n2-cluster-fiqa-2073	ok	61.79	491s
train:arm-n3-cluster-fiqa-plus-squadv2-10k	ok	-	23084s
bench:arm-n3-cluster-fiqa-plus-squadv2-10k	ok	62.01	509s
train:arm-n4-cluster-mix3-6219	ok	-	19480s
regime:quantized:mlm-path-ep3:tiny10	ok	55.62 10	531s
regime:quantized:mlm-msmarco-ep1:tiny10	ok	59.3 10	597s
train:arm-mlm-probe-a	FAILED	-	7s
step:probe_train	FAILED	1	
sparsity:mlm-probe-a-s00625	ok	418.5	33s
step:probe_train	FAILED	1	
train:arm-mlm-probe-a	ok	-	16543s
regime:quantized:mlm-probe-a-s00625:tiny10	ok	54.71 10	496s
sparsity:mlm-probe-a-s01250	ok	434.9	30s
regime:quantized:mlm-probe-a-s01250:tiny10	ok	54.73 10	484s
sparsity:mlm-probe-a-s01875	ok	390.8	30s
regime:quantized:mlm-probe-a-s01875:tiny10	ok	55.7 10	461s
sparsity:mlm-probe-a-s02500	ok	297.2	30s
regime:quantized:mlm-probe-a-s02500:tiny10	ok	56.33 10	422s
sparsity:mlm-probe-a-s05000	ok	310.6	30s
regime:quantized:mlm-probe-a-s05000:tiny10	ok	55.37 10	415s
sparsity:mlm-probe-a-s10000	ok	249.2	30s
regime:quantized:mlm-probe-a-s10000:tiny10	ok	55.62 10	395s
sparsity:mlm-probe-a-s15000	ok	222.4	29s
regime:quantized:mlm-probe-a-s15000:tiny10	ok	52.94 10	383s
sparsity:mlm-probe-a-s18750	ok	221.0	30s
regime:quantized:mlm-probe-a-s18750:tiny10	ok	52.44 10	382s
mlm2:head	ok	stock	2.26s/step
train:modernbert-probe-m	ok	-	16950s
sparsity:modernbert-probe-m-s05000	ok	0.0	40s
regime:quantized:modernbert-probe-m-s05000:tiny10	ok	 0	32s
sparsity:modernbert-probe-m-s02500	ok	0.0	38s
regime:quantized:modernbert-probe-m-s02500:tiny10	ok	 0	30s
sparsity:modernbert-probe-m-s01250	ok	0.0	37s
regime:quantized:modernbert-probe-m-s01250:tiny10	ok	 0	29s
sparsity:modernbert-probe-m-s00625	ok	0.0	38s
regime:quantized:modernbert-probe-m-s00625:tiny10	ok	 0	30s
sparsity:modernbert-probe-m-s01875	ok	0.0	39s
regime:quantized:modernbert-probe-m-s01875:tiny10	ok	 0	29s
sparsity:modernbert-probe-m-s03750	ok	0.0	38s
regime:quantized:modernbert-probe-m-s03750:tiny10	ok	 0	30s
regime:quantized:modernbert-probe-m-s05000:tiny10	FAILED	-	96s
mlm-control:noreg-scale1	0	-	1057s
mlm-control:noreg-scale1e-4	1	-	1709s
mlm-controls:analysis	0/0	-	no further training scheduled
mlm-control:noreg-scale1e-4	139	-	43s
mlm-controls:analysis	0/0	-	no further training scheduled
mlm-control:noreg-scale1e-4	2	-	23s
mlm-controls:analysis	0/0	-	no further training scheduled
mlm-control:noreg-scale1e-4	0	-	1738s
mlm-controls:analysis	0/0	-	no further training scheduled
regime:quantized:modernbert-init-n2-e5-s01500:tiny10	ok	55.63 10	787s
regime:quantized:modernbert-init-n2-e5-s02000:tiny10	ok	56.61 10	851s
regime:quantized:night3-paired-a-s04306:heldout8	ok	39.64 8	1834s
regime:quantized:night3-paired-m-s04306:heldout8	ok	37.12 8	1964s
score:squadv2-v2-all	ok	datasets/arms/squadv2-v2-all-tpd400.jsonl	6462s
score:msmarco-rejected-20000	ok	datasets/judged/msmarco-rejected-20000-tpd400.jsonl	539s
score:msmarco-clean-20000	ok	datasets/judged/msmarco-clean-20000-tpd400.jsonl	529s
score:msmarco-plain-20000	ok	datasets/judged/msmarco-plain-20000-tpd400.jsonl	530s
score:squadv2-fresh-16000	ok	datasets/arms/squadv2-fresh-16000-tpd400.jsonl	822s

```

## FAILED

```
[22:08:08] [n4] FAILED publishing results: git push
[22:08:08] [n4] FAILED initial publishing
[22:39:04] [n4] FAILED publishing results: git push
[22:39:04] [n4] FAILED updating report/publishing
[23:12:09] [n4] FAILED publishing results: git push
[23:12:09] [n4] FAILED updating report/publishing
[01:00:49] [n4] FAILED publishing results: git push
[01:00:49] [score] FAILED updating n4 report/publishing
[01:10:09] [n4] FAILED publishing results: git push
[01:10:09] [score] FAILED updating n4 report/publishing
[01:19:18] [n4] FAILED publishing results: git push
[01:19:18] [score] FAILED updating n4 report/publishing
[01:28:26] [n4] FAILED publishing results: git push
[01:28:26] [score] FAILED updating n4 report/publishing
```
