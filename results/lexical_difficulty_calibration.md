# Lexical difficulty calibration

- backend: stanza (pl, tokenize+pos+lemma, CPU) model_dir=/home/jsowa/.cache/stanza/1.14.0/resources
- inputs: fiqa=datasets/arms/fiqa-2073.jsonl, squadv2=datasets/arms/squadv2-2073.jsonl, msmarco=datasets/arms/msmarco-clean-2073.jsonl, trivia=/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/doc2query/data/raw/trivia-mined-negatives/train_pl.jsonl, nq=/home/jsowa/work/speakleash/mine-hard-negatives/data/natural_questions_pl_mxbai_scores.jsonl
- sample: 300 rows per input, seed 42, first 20000 rows as candidates
- idf_pool: 4497 passages (4497 unique texts: all sampled positives + 3000 external), 73391 lemmas, rare = idf > 7.72 (90th df-weighted percentile)
- stopwords: 309 lemma-level stopwords
- wall_time: 2436s total, 2431s lemmatising

## IDF-weighted query coverage

| dataset | n | p10 | p25 | median | p75 | p90 | mean | >0.5 | >0.8 | >0.9 | has_rare_match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fiqa | 300 | 0.163 | 0.324 | 0.494 | 0.706 | 0.871 | 0.516 | 0.487 | 0.163 | 0.090 | 0.130 |
| squadv2 | 300 | 0.510 | 0.647 | 0.785 | 0.951 | 1.000 | 0.771 | 0.900 | 0.463 | 0.300 | 0.267 |
| msmarco | 300 | 0.270 | 0.500 | 0.738 | 1.000 | 1.000 | 0.681 | 0.740 | 0.413 | 0.313 | 0.400 |
| trivia | 300 | 0.236 | 0.365 | 0.531 | 0.739 | 0.838 | 0.547 | 0.553 | 0.170 | 0.040 | 0.347 |
| nq | 300 | 0.418 | 0.601 | 0.764 | 0.904 | 1.000 | 0.737 | 0.863 | 0.420 | 0.257 | 0.317 |

## coverage

| dataset | n | p10 | p25 | median | p75 | p90 | mean |
|---|---|---|---|---|---|---|---|
| fiqa | 300 | 0.200 | 0.333 | 0.500 | 0.675 | 0.857 | 0.525 |
| squadv2 | 300 | 0.500 | 0.667 | 0.778 | 0.906 | 1.000 | 0.764 |
| msmarco | 300 | 0.250 | 0.500 | 0.667 | 1.000 | 1.000 | 0.666 |
| trivia | 300 | 0.286 | 0.400 | 0.539 | 0.714 | 0.803 | 0.547 |
| nq | 300 | 0.400 | 0.593 | 0.750 | 0.857 | 1.000 | 0.712 |

## longest_copied_ngram

| dataset | n | p10 | p25 | median | p75 | p90 | mean |
|---|---|---|---|---|---|---|---|
| fiqa | 300 | 1.000 | 1.000 | 2.000 | 2.000 | 4.000 | 2.033 |
| squadv2 | 300 | 1.000 | 2.000 | 3.000 | 4.000 | 5.000 | 2.913 |
| msmarco | 300 | 1.000 | 1.000 | 1.000 | 2.000 | 3.000 | 1.657 |
| trivia | 300 | 1.000 | 1.000 | 2.000 | 3.000 | 4.000 | 2.270 |
| nq | 300 | 1.000 | 2.000 | 3.000 | 3.000 | 5.000 | 2.757 |

## copy_density

| dataset | n | p10 | p25 | median | p75 | p90 | mean |
|---|---|---|---|---|---|---|---|
| fiqa | 300 | 0.077 | 0.125 | 0.167 | 0.268 | 0.400 | 0.210 |
| squadv2 | 300 | 0.143 | 0.200 | 0.286 | 0.400 | 0.500 | 0.319 |
| msmarco | 300 | 0.143 | 0.200 | 0.286 | 0.400 | 0.500 | 0.324 |
| trivia | 300 | 0.083 | 0.125 | 0.182 | 0.250 | 0.333 | 0.201 |
| nq | 300 | 0.143 | 0.250 | 0.333 | 0.444 | 0.571 | 0.355 |

## entity_preservation

| dataset | n | p10 | p25 | median | p75 | p90 | mean |
|---|---|---|---|---|---|---|---|
| fiqa (queries with entity-like tokens: 0.380) | 114 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.664 |
| squadv2 (queries with entity-like tokens: 0.680) | 204 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.883 |
| msmarco (queries with entity-like tokens: 0.147) | 44 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.879 |
| trivia (queries with entity-like tokens: 0.833) | 250 | 0.000 | 0.500 | 0.800 | 1.000 | 1.000 | 0.697 |
| nq (queries with entity-like tokens: 0.750) | 225 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.884 |

## max_matched_idf

| dataset | n | p10 | p25 | median | p75 | p90 | mean |
|---|---|---|---|---|---|---|---|
| fiqa | 300 | 3.847 | 4.698 | 5.772 | 7.025 | 8.411 | 5.766 |
| squadv2 | 300 | 4.974 | 5.926 | 7.025 | 8.411 | 8.411 | 6.888 |
| msmarco | 300 | 4.070 | 6.013 | 7.718 | 8.411 | 8.411 | 6.737 |
| trivia | 300 | 4.715 | 5.926 | 7.313 | 8.411 | 8.411 | 6.915 |
| nq | 300 | 4.774 | 5.926 | 7.025 | 8.411 | 8.411 | 6.881 |

## q_len_content

| dataset | n | p10 | p25 | median | p75 | p90 | mean |
|---|---|---|---|---|---|---|---|
| fiqa | 300 | 3.900 | 5.000 | 6.000 | 8.000 | 10.000 | 6.493 |
| squadv2 | 300 | 3.000 | 5.000 | 6.000 | 7.000 | 9.000 | 6.013 |
| msmarco | 300 | 2.000 | 2.000 | 3.000 | 4.000 | 5.000 | 3.410 |
| trivia | 300 | 4.000 | 6.000 | 8.000 | 10.000 | 13.000 | 8.243 |
| nq | 300 | 3.000 | 4.000 | 5.000 | 6.000 | 7.000 | 5.160 |

## Separation (AUC = P(metric is higher in the second dataset), 0.5 = no separation)

| metric | fiqa vs trivia | fiqa vs nq | fiqa vs squadv2 | fiqa vs msmarco | msmarco vs trivia |
|---|---|---|---|---|---|
| idf_coverage | 0.538 | 0.743 | 0.780 | 0.676 | 0.339 |
| coverage | 0.528 | 0.715 | 0.774 | 0.656 | 0.353 |
| longest_copied_ngram | 0.578 | 0.665 | 0.678 | 0.412 | 0.664 |
| copy_density | 0.517 | 0.772 | 0.734 | 0.721 | 0.275 |
| entity_preservation | 0.487 | 0.620 | 0.623 | 0.625 | 0.333 |
| max_matched_idf | 0.697 | 0.694 | 0.693 | 0.699 | 0.481 |
| q_len_content | 0.640 | 0.331 | 0.442 | 0.132 | 0.924 |
| has_rare_match | 0.608 | 0.593 | 0.568 | 0.635 | 0.473 |
