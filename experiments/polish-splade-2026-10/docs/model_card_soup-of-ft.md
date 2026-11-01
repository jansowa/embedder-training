---
language:
  - pl
license: apache-2.0
base_model: sdadas/polish-splade
library_name: sentence-transformers
pipeline_tag: feature-extraction
tags:
  - splade
  - sparse-encoder
  - sentence-transformers
  - sparse-retrieval
  - information-retrieval
  - polish
datasets:
  - speakleash/fiqa_pl
  - speakleash/squadv2
  - speakleash/msmarco_pl
---

# polish-splade, dostrojony na FiQA-PL (soup sześciu fine-tune'ów)

Model rzadki (SPLADE) do wyszukiwania w języku polskim: `sdadas/polish-splade`
dostrojony kontrastywnie na polskim FiQA i uśredniony w przestrzeni wag z sześciu
wariantów tego samego treningu. W porównaniu z modelem bazowym daje
**+1,7 pkt nDCG@10 na 10 zadaniach PIRB** i **+2,3 pkt na 8 innych zadaniach
PIRB, na których nie podejmowano żadnych decyzji**, przy indeksie mniejszym o
ok. 22 % (średnio 353 aktywnych wymiarów na dokument zamiast 450).

Licencja: do potwierdzenia z kartą modelu bazowego `sdadas/polish-splade`.

## Jak powstał

### 1. Dane treningowe

- Zbiór: `speakleash/fiqa_pl`, plik `train_pl.jsonl` (polskie FiQA: pytania
  z forum finansowego + pasaże).
- Filtr: zostawiono tylko pozytywy, dla których score silniejszego rerankera
  zapisany w metadanych zbioru (`pos_scores_stronger_reranker`, w jego braku
  `pos_scores`) jest większy niż 23,50; pytania bez takiego pozytywu odpadają.
  Po filtrze: **2073 pytania**, każde z jednym pozytywem i twardymi negatywami
  ze zbioru (w treningu używano 6 pierwszych negatywów na pytanie).
- Jeden z sześciu składników (`ck-grid-s`) był trenowany na mieszance FiQA-PL,
  SQuAD-v2-PL i MS MARCO-PL z tym samym filtrem (1 epoka, 1866 kroków).

### 2. Przepis treningu (backend sentence-transformers 5.3, `SparseEncoder`)

| element | wartość |
|---|---|
| model startowy | `sdadas/polish-splade` (owinięty w `SparseEncoder` z `SpladePooling(max)`) |
| strata | `SparseMultipleNegativesRankingLoss` (in-batch + twarde negatywy, skala 1.0) w `SpladeLoss` |
| regularyzacja FLOPS | dokumenty λ_d = 0,001, zapytania λ_q = 0,00005, stałe (bez rozgrzewania λ) |
| negatywy na pytanie | 6 |
| efektywny batch | 32 (4 × 8 akumulacji, lub 2 × 16 dla wariantów `-l`) |
| sampler | `no_duplicates` |
| precyzja | fp16, gradient checkpointing |
| epoki | 2 (ok. 130 kroków optymalizatora na 2073 pytaniach) |
| harmonogram | liniowy |

Sześć składników soupu, każdy to osobny fine-tune bazy tym przepisem:

| składnik | różnica względem przepisu |
|---|---|
| `ck-fiqa-s` | lr 5e-5, bez warmupu, długość treningowa 256 |
| `ck-fiqa-s-ep1` | checkpoint po 1. epoce `ck-fiqa-s` |
| `ck-fiqa-l` | lr 5e-5, bez warmupu, długość 512 (batch 2 × 16) |
| `ck-warmup-s` | lr 2e-5, warmup 10 %, długość 256 |
| `ck-warmup-l` | lr 2e-5, warmup 10 %, długość 512 (batch 2 × 16) |
| `ck-grid-s` | lr 2e-5, warmup 10 %, 1 epoka na mieszance FiQA + SQuAD-v2-PL + MS MARCO-PL |

### 3. Uśrednienie wag

Wagi wszystkich sześciu modeli uśredniono jednostajnie (po 1/6), łącznie z głową
MLM; żadnego składnika nie wyrzucono, także dwóch najsłabszych. Merge jest zwykłą
interpolacją w przestrzeni parametrów (ten sam tokenizer i architektura, więc nie
wymaga dopasowywania). Model wynikowy jest lepszy od każdego ze składników
osobno (najlepszy składnik `ck-warmup-l`: 61,78 na tych samych 10 zadaniach).

## Wyniki (PIRB, nDCG@10, Anserini, długość 512)

Pomiar na dwóch rozłącznych grupach zadań PIRB. Na pierwszej („tiny”, 10 zadań)
porównywano kilkadziesiąt wariantów i na jej podstawie wybrano ten model; druga
(8 zadań) nie była używana do żadnej decyzji i mierzy, ile z poprawy zostaje
poza zbiorem selekcyjnym.

Szum pomiaru: trzy treningi tego samego przepisu różniące się tylko ziarnem dały
na „tiny” 61,49 / 61,68 / 61,38 (SD 0,15), więc różnice poniżej ok. 0,3 pkt nie
mają znaczenia.

### Grupa selekcyjna (10 zadań): 61,87 vs 60,14 (+1,73; 95 % CI bootstrapu [+1,18; +2,25]; 10/10 zadań w górę)

| zadanie | baza | ten model | Δ |
|---|---:|---:|---:|
| maupqa-gpt3-cc | 18,18 | 18,20 | +0,02 |
| maupqa-mkqa | 59,88 | 61,78 | +1,90 |
| maupqa-poleval2021-pairs | 77,53 | 79,06 | +1,54 |
| maupqa-templates | 71,61 | 73,10 | +1,49 |
| poleval-2022 allegro-faq test-A | 73,05 | 75,55 | +2,50 |
| poleval-2022 allegro-faq test-B | 67,77 | 70,68 | +2,91 |
| arguana-pl | 41,72 | 44,15 | +2,43 |
| nfcorpus-pl | 30,79 | 32,52 | +1,73 |
| scifact-pl | 62,75 | 65,08 | +2,33 |
| gpt-exams | 98,10 | 98,55 | +0,45 |

### Grupa kontrolna (8 zadań nieużywanych do wyboru modelu): 40,37 vs 38,09 (+2,28; 8/8 zadań w górę)

| zadanie | baza | ten model | Δ |
|---|---:|---:|---:|
| maupqa-1z10 | 43,55 | 45,95 | +2,40 |
| maupqa-gpt3.5-cc | 19,69 | 19,90 | +0,21 |
| maupqa-multilingual-NLI | 13,76 | 14,10 | +0,34 |
| maupqa-wiki-def | 55,08 | 60,34 | +5,26 |
| poleval-2022 legal-questions test-A | 78,71 | 80,70 | +1,99 |
| fiqa-pl | 23,06 | 27,24 | +4,18 |
| scidocs-pl | 10,58 | 12,71 | +2,13 |
| mfaq | 60,26 | 61,99 | +1,73 |

Zysk na `fiqa-pl` jest zgodny z domeną treningu, ale poprawa pojawia się też na
Wikipedii, pytaniach prawnych, FAQ i tekstach naukowych.

### Koszt indeksu (ta sama próbka pasaży i zapytań z PIRB, kwantyzacja round(w·100) > 0)

| | baza | ten model | stosunek |
|---|---:|---:|---:|
| aktywnych wymiarów na dokument | 450 | 353 | 0,78 |
| aktywnych wymiarów na zapytanie | 139 | 126 | 0,91 |
| postingów odwiedzanych na zapytanie | 16 415 | 14 995 | 0,91 |

Indeks (Anserini, Qdrant sparse) jest więc o ok. 1/5 mniejszy, a zapytanie
odwiedza o ok. 9 % mniej postingów niż w modelu bazowym.

## Czego nie zmierzono

- Pozostałe zadania PIRB (m.in. `msmarco-pl`, `nq-pl`, `hotpotqa-pl`,
  `dbpedia-pl`, `maupqa-mqa`, `quora-pl`, `trec-covid-pl`, poleval wiki-trivia)
  nie były liczone; druga grupa kontrolna (5 kolejnych zadań) jest w trakcie.
- Długie dokumenty: trening szedł na długości 256 (składniki `-l` na 512), ocena
  na 512.
- Model nie był kalibrowany ani testowany poza PIRB (np. na własnych danych
  produkcyjnych).

## Użycie

```python
from sentence_transformers import SparseEncoder

model = SparseEncoder("speakleash/<nazwa-repo>")
model.max_seq_length = 512

q = model.encode_query(["jak rozliczyć stratę z inwestycji w PIT"])
d = model.encode_document(["Stratę z odpłatnego zbycia papierów wartościowych ..."])
print(model.similarity(q, d))
```

Wektory są rzadkie (wymiar = słownik tokenizera, ok. 50 k); do indeksu Anserini
lub Qdrant zwykle wystarczy zaokrąglić wagi (`round(w * 100)`) i odrzucić zera.
Model był oceniany z pustym prefiksem zapytania (bez instrukcji).

## Pochodzenie i reprodukcja

- Repozytorium treningowe: fork FlagEmbedding z backendem `sentence-transformers`
  (`python -m training.train --backend sentence-transformers --training-type splade`).
- Konfiguracje składników: `configs/weekend/ft_fiqa_3060_*.yaml`,
  `configs/weekend/ft_grid_3060_lr2e-5.yaml`; merge:
  `scripts/merge_splade_weights.py --models <6 katalogów> --output models/soup-of-ft`.
- Benchmark: PIRB (Anserini, JDK 21), `run_experiments.py --mode benchmark
  --run_pirb --pirb_scope tiny --pirb-max-seq-length 512`; grupa kontrolna przez
  `--pirb-tasks ...`.
- Sprzęt: jedna karta RTX 3060 Ti (8 GB); jeden składnik trenuje się ok. 8 minut.
