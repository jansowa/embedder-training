# PIRB comparison (NDCG@10)

Tasks in the intersection: 10

Reference column: `./models/polish-splade`

Skipped tasks (missing in at least one column): abczdrowie, dbpedia-pl, eprawnik, fiqa-pl, gemini, hotpotqa-pl, maupqa-1z10, maupqa-czy-wiesz-v2, maupqa-gpt3.5-cc, maupqa-gpt3.5-wiki, maupqa-mqa, maupqa-multilingual-NLI, maupqa-poquad, maupqa-wiki-def, mfaq, msmarco-pl, nq-pl, odi, onet, poleval-2022-dev-0-wiki-trivia, poleval-2022-test-A-legal-questions, poleval-2022-test-A-wiki-trivia, poleval-2022-test-B-legal-questions, poleval-2022-test-B-wiki-trivia, pwn, quora-pl, scidocs-pl, specprawnik, techpedia, trec-covid-pl, zapytajfizyka

| row | group | tasks | ./models/polish-splade | ./models/mmlw/mmlw-retrieval-roberta-base | final | base-splade/final | ck-fiqa-l/final | ck-fiqa-s-ep1/final | ck-fiqa-s/final | ck-grid-s/final | ck-warmup-l/final | ck-warmup-s/final | judge-clean/final | judge-plain/final | judge-rejected/final | mlm-msmarco-ep1/final | soup-a0.05/final | soup-a0.10/final | soup-a0.15/final | soup-a0.25/final | soup-a0.35/final | soup-a0.5/final | soup-a0.75/final | soup-of-ft-wiseft-a0.5/final | soup-of-ft-wiseft-a0.7/final | soup-of-ft/final | sw-lr1e-5/final | sw-lr3e-5/final | sw-seed43/final | sw-seed44/final | wiseft-ck-fiqa-s-a0.15/final | wiseft-ck-warmup-l-a0.3/final | wiseft-ck-warmup-l-a0.5/final | wiseft-ck-warmup-l-a0.7/final |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maupqa-gpt3-cc | MaupQA | 1 | 18.41 | 17.14 | 18.18 | 6.75 | 16.96 | 17.75 | 17.66 | 18.56 | 17.83 | 18.07 | 18.21 | 18.12 | 17.88 | 17.85 | 18.30 | 18.31 | 18.37 | 18.36 | 18.39 | 18.45 | 18.23 | 18.26 | 18.26 | 18.20 | 18.32 | 17.90 | 18.08 | 17.96 | 18.26 | 18.20 | 18.13 | 18.02 |
| maupqa-mkqa | MaupQA | 1 | 63.88 | 61.76 | 59.88 | 7.07 | 58.36 | 59.62 | 60.16 | 61.28 | 60.86 | 60.68 | 59.45 | 59.14 | 57.43 | 55.04 | 60.24 | 60.30 | 60.04 | 59.11 | 58.36 | 57.44 | 56.27 | 61.65 | 61.81 | 61.78 | 61.20 | 60.66 | 60.73 | 60.93 | 61.00 | 60.85 | 61.15 | 61.22 |
| maupqa-poleval2021-pairs | MaupQA | 1 | 82.91 | 83.03 | 77.53 | 20.57 | 77.86 | 78.84 | 78.76 | 78.04 | 79.42 | 79.67 | 77.99 | 75.70 | 75.44 | 74.63 | 77.75 | 77.51 | 77.54 | 77.27 | 76.72 | 76.51 | 75.58 | 79.32 | 79.09 | 79.06 | 79.42 | 79.27 | 79.55 | 79.96 | 78.21 | 78.64 | 78.92 | 79.30 |
| maupqa-templates | MaupQA | 1 | 75.23 | 74.53 | 71.61 | 11.81 | 71.68 | 72.15 | 72.24 | 73.38 | 73.00 | 72.50 | 71.86 | 71.66 | 71.21 | 70.60 | 71.67 | 71.73 | 71.73 | 71.85 | 71.85 | 72.20 | 71.74 | 72.89 | 73.04 | 73.10 | 72.53 | 72.21 | 72.88 | 72.73 | 72.05 | 72.33 | 72.67 | 72.84 |
| poleval-2022-test-A-allegro-faq | PolEval-2022 | 1 | 73.37 | 74.86 | 73.05 | 28.73 | 73.70 | 73.97 | 75.20 | 72.74 | 75.24 | 75.04 | 70.40 | 70.59 | 70.05 | 67.12 | 72.92 | 72.60 | 73.09 | 72.43 | 71.29 | 70.76 | 68.28 | 73.92 | 74.99 | 75.55 | 74.25 | 74.85 | 74.56 | 75.10 | 74.52 | 73.66 | 74.49 | 74.45 |
| poleval-2022-test-B-allegro-faq | PolEval-2022 | 1 | 71.87 | 74.32 | 67.77 | 27.03 | 68.97 | 67.83 | 69.97 | 67.37 | 71.05 | 70.76 | 68.06 | 66.16 | 64.24 | 61.76 | 68.01 | 67.85 | 67.72 | 66.87 | 65.84 | 64.21 | 62.17 | 70.34 | 70.84 | 70.68 | 70.33 | 71.22 | 71.75 | 70.56 | 68.60 | 68.52 | 69.48 | 69.83 |
| arguana-pl | BEIR-PL | 1 | 36.44 | 55.55 | 41.72 | 30.10 | 45.54 | 41.78 | 42.74 | 38.13 | 44.99 | 42.60 | 38.75 | 39.84 | 37.23 | 38.11 | 41.80 | 41.99 | 41.95 | 42.37 | 42.07 | 42.85 | 41.35 | 43.22 | 43.75 | 44.15 | 42.41 | 43.23 | 43.70 | 41.75 | 42.43 | 42.97 | 43.79 | 44.46 |
| nfcorpus-pl | BEIR-PL | 1 | 29.84 | 34.96 | 30.79 | 5.92 | 31.13 | 32.23 | 31.93 | 32.40 | 31.90 | 31.79 | 32.11 | 30.99 | 29.41 | 31.32 | 31.86 | 31.91 | 31.82 | 31.52 | 31.05 | 31.90 | 31.33 | 32.41 | 32.50 | 32.52 | 32.08 | 32.05 | 32.20 | 31.99 | 31.57 | 31.30 | 31.66 | 31.89 |
| scifact-pl | BEIR-PL | 1 | 64.70 | 71.43 | 62.75 | 14.17 | 64.73 | 63.41 | 63.88 | 64.68 | 64.92 | 65.43 | 63.25 | 61.33 | 60.55 | 62.95 | 63.54 | 63.82 | 63.66 | 63.97 | 64.11 | 63.12 | 62.87 | 64.75 | 64.61 | 65.08 | 64.52 | 64.33 | 64.86 | 64.35 | 64.29 | 64.16 | 64.47 | 65.30 |
| gpt-exams | Other | 1 | 98.62 | 98.91 | 98.10 | 78.23 | 98.31 | 98.25 | 98.44 | 97.83 | 98.57 | 98.38 | 97.79 | 97.59 | 97.43 | 96.74 | 98.12 | 98.00 | 97.95 | 97.89 | 97.83 | 97.45 | 96.99 | 98.44 | 98.47 | 98.55 | 98.33 | 98.27 | 98.47 | 98.43 | 98.23 | 98.30 | 98.38 | 98.48 |
| mean |  | 10 | 61.53 | 64.65 | 60.14 | 23.04 | 60.72 | 60.58 | 61.10 | 60.44 | 61.78 | 61.49 | 59.79 | 59.11 | 58.09 | 57.61 | 60.42 | 60.40 | 60.39 | 60.16 | 59.75 | 59.49 | 58.48 | 61.52 | 61.73 | 61.87 | 61.34 | 61.40 | 61.68 | 61.38 | 60.91 | 60.89 | 61.31 | 61.58 |
| group mean | MaupQA | 4 | 60.10 | 59.12 | 56.80 | 11.55 | 56.22 | 57.09 | 57.20 | 57.82 | 57.78 | 57.73 | 56.88 | 56.15 | 55.49 | 54.53 | 56.99 | 56.96 | 56.92 | 56.65 | 56.33 | 56.15 | 55.46 | 58.03 | 58.05 | 58.04 | 57.87 | 57.51 | 57.81 | 57.89 | 57.38 | 57.51 | 57.72 | 57.84 |
| group mean | PolEval-2022 | 2 | 72.62 | 74.59 | 70.41 | 27.88 | 71.34 | 70.90 | 72.59 | 70.05 | 73.14 | 72.90 | 69.23 | 68.38 | 67.14 | 64.44 | 70.47 | 70.23 | 70.41 | 69.65 | 68.56 | 67.48 | 65.22 | 72.13 | 72.91 | 73.12 | 72.29 | 73.04 | 73.16 | 72.83 | 71.56 | 71.09 | 71.99 | 72.14 |
| group mean | BEIR-PL | 3 | 43.66 | 53.98 | 45.09 | 16.73 | 47.13 | 45.80 | 46.18 | 45.07 | 47.27 | 46.61 | 44.70 | 44.05 | 42.40 | 44.13 | 45.73 | 45.91 | 45.81 | 45.95 | 45.74 | 45.96 | 45.18 | 46.79 | 46.95 | 47.25 | 46.34 | 46.54 | 46.92 | 46.03 | 46.10 | 46.14 | 46.64 | 47.22 |
| group mean | Other | 1 | 98.62 | 98.91 | 98.10 | 78.23 | 98.31 | 98.25 | 98.44 | 97.83 | 98.57 | 98.38 | 97.79 | 97.59 | 97.43 | 96.74 | 98.12 | 98.00 | 97.95 | 97.89 | 97.83 | 97.45 | 96.99 | 98.44 | 98.47 | 98.55 | 98.33 | 98.27 | 98.47 | 98.43 | 98.23 | 98.30 | 98.38 | 98.48 |
| delta vs ./models/polish-splade |  | 10 | +0.00 | +3.12 | -1.39 | -38.49 | -0.80 | -0.94 | -0.43 | -1.08 | +0.25 | -0.03 | -1.74 | -2.41 | -3.44 | -3.91 | -1.10 | -1.12 | -1.14 | -1.36 | -1.78 | -2.04 | -3.05 | -0.01 | +0.21 | +0.34 | -0.19 | -0.13 | +0.15 | -0.15 | -0.61 | -0.63 | -0.21 | +0.05 |

## Columns

| column | model | source |
| --- | --- | --- |
| ./models/polish-splade | ./models/polish-splade | results/leaderboard_data.json (./models/polish-splade) |
| ./models/mmlw/mmlw-retrieval-roberta-base | ./models/mmlw/mmlw-retrieval-roberta-base | results/leaderboard_data.json (./models/mmlw/mmlw-retrieval-roberta-base) |
| final | /home/jsowa/work/FlagEmbedding/models/base-splade-wrapped | results/base-splade-wrapped/final/metrics.json |
| base-splade/final | sdadas/polish-splade | results/base-splade/final/metrics.json |
| ck-fiqa-l/final | /home/jsowa/work/FlagEmbedding/models/ck-fiqa-l | results/ck-fiqa-l/final/metrics.json |
| ck-fiqa-s-ep1/final | /home/jsowa/work/FlagEmbedding/models/ck-fiqa-s-ep1 | results/ck-fiqa-s-ep1/final/metrics.json |
| ck-fiqa-s/final | /home/jsowa/work/FlagEmbedding/models/ck-fiqa-s | results/ck-fiqa-s/final/metrics.json |
| ck-grid-s/final | /home/jsowa/work/FlagEmbedding/models/ck-grid-s | results/ck-grid-s/final/metrics.json |
| ck-warmup-l/final | /home/jsowa/work/FlagEmbedding/models/ck-warmup-l | results/ck-warmup-l/final/metrics.json |
| ck-warmup-s/final | /home/jsowa/work/FlagEmbedding/models/ck-warmup-s | results/ck-warmup-s/final/metrics.json |
| judge-clean/final | /home/jsowa/work/FlagEmbedding/models/judge-clean | results/judge-clean/final/metrics.json |
| judge-plain/final | /home/jsowa/work/FlagEmbedding/models/judge-plain | results/judge-plain/final/metrics.json |
| judge-rejected/final | /home/jsowa/work/FlagEmbedding/models/judge-rejected | results/judge-rejected/final/metrics.json |
| mlm-msmarco-ep1/final | /home/jsowa/work/FlagEmbedding/models/mlm-msmarco-ep1 | results/mlm-msmarco-ep1/final/metrics.json |
| soup-a0.05/final | /home/jsowa/work/FlagEmbedding/models/soup-a0.05 | results/soup-a0.05/final/metrics.json |
| soup-a0.10/final | /home/jsowa/work/FlagEmbedding/models/soup-a0.10 | results/soup-a0.10/final/metrics.json |
| soup-a0.15/final | /home/jsowa/work/FlagEmbedding/models/soup-a0.15 | results/soup-a0.15/final/metrics.json |
| soup-a0.25/final | /home/jsowa/work/FlagEmbedding/models/soup-a0.25 | results/soup-a0.25/final/metrics.json |
| soup-a0.35/final | /home/jsowa/work/FlagEmbedding/models/soup-a0.35 | results/soup-a0.35/final/metrics.json |
| soup-a0.5/final | /home/jsowa/work/FlagEmbedding/models/soup-sdadas-mlm-a0.5 | results/soup-a0.5/final/metrics.json |
| soup-a0.75/final | /home/jsowa/work/FlagEmbedding/models/soup-a0.75 | results/soup-a0.75/final/metrics.json |
| soup-of-ft-wiseft-a0.5/final | /home/jsowa/work/FlagEmbedding/models/soup-of-ft-wiseft-a0.5 | results/soup-of-ft-wiseft-a0.5/final/metrics.json |
| soup-of-ft-wiseft-a0.7/final | /home/jsowa/work/FlagEmbedding/models/soup-of-ft-wiseft-a0.7 | results/soup-of-ft-wiseft-a0.7/final/metrics.json |
| soup-of-ft/final | /home/jsowa/work/FlagEmbedding/models/soup-of-ft | results/soup-of-ft/final/metrics.json |
| sw-lr1e-5/final | /home/jsowa/work/FlagEmbedding/models/sw-lr1e-5 | results/sw-lr1e-5/final/metrics.json |
| sw-lr3e-5/final | /home/jsowa/work/FlagEmbedding/models/sw-lr3e-5 | results/sw-lr3e-5/final/metrics.json |
| sw-seed43/final | /home/jsowa/work/FlagEmbedding/models/sw-seed43 | results/sw-seed43/final/metrics.json |
| sw-seed44/final | /home/jsowa/work/FlagEmbedding/models/sw-seed44 | results/sw-seed44/final/metrics.json |
| wiseft-ck-fiqa-s-a0.15/final | /home/jsowa/work/FlagEmbedding/models/wiseft-ck-fiqa-s-a0.15 | results/wiseft-ck-fiqa-s-a0.15/final/metrics.json |
| wiseft-ck-warmup-l-a0.3/final | /home/jsowa/work/FlagEmbedding/models/wiseft-ck-warmup-l-a0.3 | results/wiseft-ck-warmup-l-a0.3/final/metrics.json |
| wiseft-ck-warmup-l-a0.5/final | /home/jsowa/work/FlagEmbedding/models/wiseft-ck-warmup-l-a0.5 | results/wiseft-ck-warmup-l-a0.5/final/metrics.json |
| wiseft-ck-warmup-l-a0.7/final | /home/jsowa/work/FlagEmbedding/models/wiseft-ck-warmup-l-a0.7 | results/wiseft-ck-warmup-l-a0.7/final/metrics.json |
