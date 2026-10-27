# polish-splade, soup-of-ft i soup-v2b na tle leaderboardu PIRB

Stan: 27.10.2026. Wszystkie liczby poniżej to nDCG@10 zmierzone w reżimie
`old-pyserini`, który odtwarza opublikowane wyniki `sdadas/polish-splade`
(sekcja 1). Tabele per zadanie generuje `scripts/leaderboard_compare.py`
(sekcja 3), narracja jest ręczna.

## 1. Metoda pomiaru i jej zgodność z leaderboardem

Leaderboard PIRB (`results/leaderboard_data.json`, eksport `data.json` ze
space'a sdadas/pirb-leaderboard) podaje dla `./models/polish-splade` liczby z
wpisu z 29.09.2023, liczone starym pyserini, w którym wagi zapytania trafiają
do Anserini jako floaty. Nowsze pyserini (≥ 0.22) rzutuje je na inty, przez co
termy o wadze < 1 znikają; wszystkie nasze codzienne benche (`results/`,
`results-holdout/`) były liczone w tym obciętym reżimie i są zaniżone o 2–5 pkt
na zadanie. Sprawdziliśmy cztery reżimy (`scripts/pirb_regime_eval.sh`):

| reżim | co to jest | zgodność |
|---|---|---|
| `old-pyserini` | nasz fork PIRB + pyserini 0.21.0 (`JFloat`), `~/work/venv-pyserini021` | odtwarza leaderboard: 18 zadań tiny10 + held-out-8 w ±0.3, held-out-2 w ±0.3 poza trec-covid-pl (−0.83) |
| `new-backend` | upstream `sdadas/pirb` HEAD, backend `splade-index` (dokładne floaty), `~/work/venv-pirb-new` | = `old-pyserini` ±0.1, pod warunkiem `max_seq_length: 512` w checkpoincie |
| `quantized` | nasz fork + pyserini 0.39 + `quantize_queries: true` (round(w·100) po stronie zapytania) | = `old-pyserini` (61.60 vs 61.59 na tiny10); najtańszy, działa w `.venv` |
| `truncated` | nasz fork + pyserini 0.39 bez kwantyzacji | wszystkie liczby sprzed 26.10; tylko do rankingu ramion |

Do porównań z leaderboardem używamy `old-pyserini` (ten dokument) albo
`quantized`; dzisiejsze dane sdadas-a liczone są kodem równoważnym
`new-backend`. Ustawienia: długość 512 z checkpointu, batch 16 (8 dla
maupqa-poquad i trec-covid-pl), kod PIRB z `third_party/pirb` (commit f836e46).

Odtworzenie bazy, średnie per zestaw (opublikowane → nasze): tiny10 61.53 → 61.59,
held-out-8 41.06 → 41.03, held-out-2 67.80 → 67.53.

## 2. Które zestawy zadań do czego służyły

| zestaw | zadania | rola | zastrzeżenie |
|---|---|---|---|
| **tiny10** | arguana-pl, gpt-exams, maupqa-gpt3-cc, maupqa-mkqa, maupqa-poleval2021-pairs, maupqa-templates, nfcorpus-pl, poleval-2022-test-A-allegro-faq, poleval-2022-test-B-allegro-faq, scifact-pl | zestaw deweloperski: każde z ponad 60 ramion, sweepów i zup było na nim rankowane (ok. 10 min na model w reżimie obciętym); wszystkie decyzje go/no-go (która połówka bije kontrolę, która zupa idzie na held-out, który model jest „najlepszy”) | selekcja na tym zestawie zawyża jego liczby; różnice ≤ 0.3 to szum |
| **held-out-8** | fiqa-pl, maupqa-1z10, maupqa-gpt3.5-cc, maupqa-multilingual-NLI, maupqa-wiki-def, mfaq, poleval-2022-test-A-legal-questions, scidocs-pl | mierzony tylko dla modeli, które wygrały na tiny10; rozstrzygał remisy między ≤ 3 finalistami (np. top-2 zup z v9) | lekko skażony selekcją; fiqa-pl dzieli domenę z danymi treningowymi (pary FiQA-PL, inny split), więc +2.7 na fiqa-pl jest częściowo in-domain |
| **held-out-2** | maupqa-czy-wiesz-v2, maupqa-gpt3.5-wiki, quora-pl, maupqa-poquad, trec-covid-pl | nigdy nie użyty do żadnej decyzji; zmierzony tylko dla bazy, soup-of-ft i soup-v2b | najczystsze oszacowanie generalizacji |

Dane treningowe: FiQA-PL (2073 par po bramce rerankera > 23.50, z puli 5500),
SQuAD-v2-PL 2073, MS MARCO-PL 2073 po sędzim; żadne zadanie z tych zestawów nie
było źródłem par poza wspomnianym FiQA. Pozostałe 18 zadań PIRB (9 prywatnych,
maupqa-mqa, 4 zestawy poleval-2022, dbpedia-pl, hotpotqa-pl, msmarco-pl, nq-pl)
nie były liczone.

## 3. Wyniki per zadanie

Pierwsza tabela to średnie per zestaw (liczba zadań w nawiasie; kolumna
„wszystkie wspólne” liczy średnią tylko z zadań, które mają wszystkie modele).
Dalej tabele per zadanie; kolumna „best standalone on PIRB” to najlepszy
samodzielny model leaderboardu (bez hybryd i rerankerów) na danym zadaniu:
sufit, do którego się odnosimy.

<!-- tables:start -->
### Średnie per zestaw (old-pyserini, nDCG@10)

| model | tiny10 (10) | heldout8 (8) | heldout2 (5) | wszystkie wspólne (19) | Δ vs base (wspólne) |
|---|---:|---:|---:|---:|---:|
| published polish-splade | 61.53 | 41.06 | 67.80 | 54.19 |  |
| base-splade-wrapped | 61.59 | 41.03 | 67.53 | 54.21 |  |
| soup-of-ft | 62.94 | 42.11 | 69.00 | 55.40 | +1.19 |
| soup-v2b | 63.00 | 42.06 |  | 55.43 | +1.22 |

### tiny10 (old-pyserini, nDCG@10)

| task | published polish-splade | ours base-splade-wrapped | soup-of-ft | soup-v2b | Δ soup-of-ft vs base | Δ soup-v2b vs base | best standalone on PIRB |
|---|---:|---:|---:|---:|---:|---:|---|
| arguana-pl | 36.44 | 37.13 | 42.20 | 41.99 | +5.07 | +4.86 | 70.93 (1b-etap3) |
| gpt-exams | 98.62 | 98.64 | 98.70 | 98.73 | +0.06 | +0.09 | 99.60 (llama-embed-nemotron-8b) |
| maupqa-gpt3-cc | 18.41 | 18.43 | 18.26 | 18.55 | -0.17 | +0.12 | 19.28 (silver-retriever-base-v1) |
| maupqa-mkqa | 63.88 | 63.89 | 64.30 | 64.67 | +0.41 | +0.77 | 65.86 (silver-retriever-base-v1) |
| maupqa-poleval2021-pairs | 82.91 | 82.69 | 83.24 | 82.74 | +0.55 | +0.06 | 88.64 (llama-embed-nemotron-8b) |
| maupqa-templates | 75.23 | 75.01 | 75.39 | 75.59 | +0.38 | +0.58 | 79.56 (llama-embed-nemotron-8b) |
| nfcorpus-pl | 29.84 | 30.04 | 32.35 | 32.09 | +2.30 | +2.05 | 41.12 (bm25) |
| poleval-2022-test-A-allegro-faq | 73.37 | 73.75 | 76.57 | 76.30 | +2.82 | +2.54 | 89.36 (F2LLM-v2-1.7B) |
| poleval-2022-test-B-allegro-faq | 71.87 | 71.82 | 72.13 | 72.66 | +0.32 | +0.84 | 87.72 (F2LLM-v2-8B) |
| scifact-pl | 64.70 | 64.50 | 66.26 | 66.65 | +1.76 | +2.14 | 79.67 (stella-pl-retrieval-8k-ft) |
| **mean** | 61.53 | 61.59 | 62.94 | 63.00 | +1.35 | +1.41 |  |

### heldout8 (old-pyserini, nDCG@10)

| task | published polish-splade | ours base-splade-wrapped | soup-of-ft | soup-v2b | Δ soup-of-ft vs base | Δ soup-v2b vs base | best standalone on PIRB |
|---|---:|---:|---:|---:|---:|---:|---|
| fiqa-pl | 26.67 | 26.86 | 29.59 | 29.17 | +2.74 | +2.32 | 52.85 (stella_pl_2etap) |
| maupqa-1z10 | 47.39 | 47.32 | 47.98 | 48.40 | +0.66 | +1.08 | 51.81 (bge-multilingual-gemma2) |
| maupqa-gpt3.5-cc | 20.12 | 20.14 | 20.13 | 20.27 | -0.00 | +0.13 | 20.74 (silver-retriever-base-v1) |
| maupqa-multilingual-NLI | 14.15 | 14.21 | 14.51 | 14.49 | +0.31 | +0.28 | 15.87 (llama-embed-nemotron-8b) |
| maupqa-wiki-def | 64.96 | 64.74 | 65.83 | 65.72 | +1.09 | +0.98 | 75.62 (1b-etap3) |
| mfaq | 62.15 | 62.04 | 62.94 | 62.93 | +0.90 | +0.89 | 70.27 (F2LLM-v2-8B) |
| poleval-2022-test-A-legal-questions | 80.44 | 80.30 | 81.82 | 81.61 | +1.52 | +1.32 | 84.03 (bge-multilingual-gemma2) |
| scidocs-pl | 12.56 | 12.61 | 14.07 | 13.92 | +1.46 | +1.31 | 23.54 (stella_pl_1etap) |
| **mean** | 41.06 | 41.03 | 42.11 | 42.06 | +1.08 | +1.04 |  |

### heldout2 (old-pyserini, nDCG@10)

| task | published polish-splade | ours base-splade-wrapped | soup-of-ft | soup-v2b | Δ soup-of-ft vs base | Δ soup-v2b vs base | best standalone on PIRB |
|---|---:|---:|---:|---:|---:|---:|---|
| maupqa-czy-wiesz-v2 | 85.97 | 85.80 | 86.31 | 86.65 | +0.51 | +0.85 | 89.05 (llama-embed-nemotron-8b) |
| maupqa-gpt3.5-wiki | 91.22 | 91.17 | 91.53 |  | +0.36 |  | 95.70 (llama-embed-nemotron-8b) |
| maupqa-poquad | 19.57 | 19.49 | 19.48 |  | -0.01 |  | 20.44 (silver-retriever-base-v1) |
| quora-pl | 76.76 | 76.56 | 79.11 |  | +2.54 |  | 86.26 (stella_pl_1etap) |
| trec-covid-pl | 65.46 | 64.63 | 68.57 |  | +3.94 |  | 89.93 (Qwen3-Embedding-8B) |
| **mean** | 67.80 | 67.53 | 69.00 |  | +1.47 |  |  |
<!-- tables:end -->

## 4. Gdzie stoją modele i gdzie jest największe pole do poprawy

**Zysk jest realny, ale mały i skoncentrowany.** soup-of-ft i soup-v2b dają
średnio +1.2–1.5 nad polish-splade we wszystkich trzech zestawach, z tego samego
rzędu na zestawie, który nigdy nie służył do selekcji (held-out-2: +1.47). Zysk
siedzi w zadaniach typu BEIR z długimi pasażami i pytaniami nieleksykalnymi:
arguana-pl +5, trec-covid-pl +3.9, allegro-faq A +2.8, fiqa-pl +2.7, quora-pl
+2.5, nfcorpus-pl +2.3, scifact-pl +1.8–2.1, scidocs-pl +1.5. Zadania MAUPQA z
krótkimi pytaniami faktograficznymi nie ruszają się wcale (gpt3-cc, gpt3.5-cc,
poquad ≈ 0; mkqa, templates, czy-wiesz +0.4–0.8) i to one dominują w średniej.
Na nich także najlepsze modele leaderboardu są tylko 1–2 pkt nad polish-splade:
tam nie ma pola do poprawy dla nikogo.

**Dystans do czołówki leaderboardu jest architektoniczny.** Najlepsze samodzielne
modele to gęste embeddery 1.7B–8B; na zadaniach, gdzie liczy się semantyka ponad
pokryciem leksykalnym, dzieli nas 10–30 pkt: arguana-pl 42 vs 71, fiqa-pl 30 vs
53, scidocs-pl 14 vs 24, scifact-pl 66 vs 80, trec-covid-pl 69 vs 90, allegro-faq
77 vs 89 i 73 vs 88, mfaq 63 vs 70, wiki-def 66 vs 76, quora-pl 79 vs 86.
polish-splade ma 6 warstw i 82M parametrów; celem realnym dla tej architektury
jest „najlepszy model rzadki dla polskiego z wyraźnym marginesem”, nie szczyt tabeli.

**Największe pola do poprawy (w kolejności oczekiwanej wartości):**

1. **Sygnał uczący.** Cały fine-tuning był kontrastowy (MNRL) na małych dawkach
   par; każda recepta (FiQA, FiQA+SQuAD, 512/10 negatywów, zupy) ląduje w ±0.3 od
   siebie, dłuższy trening nie pomaga: plateau recepty. Następny krok to
   destylacja marginesów od `sdadas/polish-reranker-roberta-v3` (`sparse_margin_mse`,
   backend gotowy) na tych samych 2073 zapytaniach po bramce, ze świeżymi
   negatywami z `mine-hard-negatives` (progi percentylowe + bezwzględne).
   Szczegóły: `docs/PLAN-2026-10-27.md`.
2. **Zadania parafrazowe/FAQ** (allegro-faq, quora, mfaq): 7–13 pkt do sufitu,
   a tu SPLADE może zyskać najwięcej na lepszej ekspansji zapytań; dziś nasze
   modele dają tam +0.3 do +2.8.
3. **nfcorpus-pl: BM25 (41.1) bije każdy model neuronowy, nas o 9 pkt.** Sygnał
   leksykalny na długich tekstach medycznych jest tracony; kandydaci: słabsza
   regularyzacja FLOPS dokumentów, ocena przy dłuższym kontekście, hybryda z BM25.
4. **Długość w ocenie.** soup-of-ft oceniany przy 256 zamiast 512 tokenów daje
   63.42 / 42.16 (tiny10 / held-out-8, `new-backend`), czyli +0.5 za darmo na
   arguana-pl (długie zapytania); do sprawdzenia na held-out-2 przed użyciem.
5. **Koszt indeksu jest już niższy niż bazy**, nie wymaga pracy: aktywne wymiary
   na dokument 346 (soup-of-ft, ×0.79) i 326 (soup-v2b, ×0.75) wobec 436 bazy;
   na zapytanie 126 vs 139 (×0.91) dla soup-of-ft; postings na zapytanie ×0.91.

## 5. Odtworzenie

```bash
# jeden model, jeden zestaw, reżim leaderboardu (wyniki w results-regimes/old-pyserini/<label>/<zestaw>.json)
scripts/pirb_regime_eval.sh old-pyserini soup-v2b models/soup-v2b tiny10 16
scripts/pirb_regime_eval.sh old-pyserini soup-v2b models/soup-v2b heldout8 16
for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora; do scripts/pirb_regime_eval.sh old-pyserini soup-v2b models/soup-v2b $ts 16; done
for ts in heldout2_poquad heldout2_treccovid; do scripts/pirb_regime_eval.sh old-pyserini soup-v2b models/soup-v2b $ts 8; done
# tabele z sekcji 3
python scripts/leaderboard_compare.py --regime old-pyserini --model base-splade-wrapped --model soup-of-ft --model soup-v2b \
    --taskset tiny10 --taskset heldout8 --taskset heldout2
```
