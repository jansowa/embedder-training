# Zatwierdzone A/B kontynuacji po trzech epokach KL

Plan z 8 listopada zastępuje oczekujące A/B ze startem od MLM. Główny trening
KL trwa dalej bez zmian. Stary nadzorca A/B został zatrzymany wyłącznie w stanie
oczekiwania, bez dzieci i bez pracy na GPU.

## Kolejność

1. Ukończenie trzeciej epoki, tiny10 i profile wszystkich trzech checkpointów.
2. Wybór najwyższego tiny10 spośród epok 1, 2 i 3, bez użycia heldout8 ani
   bramkowania gęstością. Przy remisie zostaje wcześniejsza epoka.
3. Smoke KL+MNRL: 20 kroków ze wskazanego checkpointu.
4. Dwa niezależne treningi ze wspólnych wag, ze świeżymi optymalizatorami:

| Ramię | Loss | Dane | Długość | LR |
|---|---|---|---|---|
| KL | KL, T=2 | 40 000 zapytań relevance>=2 | 1 epoka | 5e-6 |
| KL+MNRL | KL, T=2 + MNRL z wagą 1, scale=1 | te same 40 000 | 1 epoka | 5e-6 |

Oba ramiona mają identyczne: dane i kolejność, seed=42, fizyczny batch=8,
akumulację=4, 10 negatywów, max_seq_length=256, FP16, gradient checkpointing,
pooling chunk=8, rozgrzewkę LR=50 kroków i stały LR po rozgrzewce.
Nie ma ramienia z wagą MNRL=3 ani startów od MLM.

## Wagi i regularizacja

Szablony `mlm_p1_continue_ab_{kl,kl_mnrl,smoke}.yaml` mają znacznik
`__P1_BEST_CHECKPOINT__`. Nadzorca wiąże je dopiero po ukończeniu pomiarów fazy 1,
zapisując wybraną epokę, sumę wag modelu, sumy szablonów i konfiguracji wynikowych.
Każdy worker porównuje wszystkie zapisane tensory MLM z tym samym checkpointem
przed pierwszym krokiem. Smoke nie modyfikuje wspólnego źródła wag.

Nie uruchamiamy ponownej kalibracji biasu. Wagi regularyzacji dokumentów i zapytań
pochodzą z ostatniej kontroli nie późniejszej niż wybrany checkpoint; są zamrożone
i wspólne dla obu ramion. Nie ma ponownej rozgrzewki λ ani niezależnej adaptacji
kontrolera w poszczególnych ramionach. Usuwa to dodatkową różnicę między stratami.

Zapis: pierwszy krok kontrolny, następnie co 500 kroków; dwa ostatnie zwykłe
checkpointy oraz osobna kopia końca epoki. Po zapisie sprawdzane są wagi modelu,
stan optymalizatora, schedulera, RNG i Trenera. Alias biasu RoBERTy jest zachowany
również w modelu finalnym. Rezerwa dysku: 10 GB na C: i dysku wyjściowym.

## Ocena i dalsza decyzja

Każde pełne ramię otrzymuje tiny10 quantized, FP16, 512 tokenów, batch 16 oraz
profil wym./dok, wym./zapytanie i postings/zapytanie, także min_weight=0.05.
Raport pokazuje zmianę tiny10 względem wspólnego startu oraz kontrolnego KL,
a także stosunki każdego z trzech kosztów indeksu KL+MNRL/KL.

Najwyższy tiny10 spośród startu i obu kontynuacji wskazuje wariant do heldout8.
Zachowanie startu jako kandydata chroni przed wyborem pogorszonej kontynuacji.
Heldout8 nie uczestniczy w selekcji. Istniejąca kolejka fazy 1 ma również ocenę
heldout8 wybranego checkpointu KL; jeśli pozostaje on najlepszy, wynik jest
wykorzystywany ponownie po sprawdzeniu modelu i reżimu.

Po A/B wybieramy na podstawie jakości i kosztu indeksu: szerszą walidację MNRL
na puli 301 042 przykładów albo dogrzanie n5+kj (20 292 + ustalony dodatek
dobrego MS MARCO). Te dwa warianty nie są automatycznie uruchamiane przez
nadzorcę A/B: wybór zależy od wyników, a wielkość dodatku MS MARCO nie jest
jeszcze ustalona. Nie stosujemy reguły doboru wagi wyłącznie z wartości lossów.

## Artefakty

- Nadzorca: `queues/queue_5070_p1_continue_ab.py`.
- Stan: `logs/mlm-p1-ab-queue.json`, pole `mode=continuation`.
- Zgoda i zamrożone szablony: `logs/mlm-p1-continue-ab-approval.json`.
- Wybrany start i związane konfiguracje: `logs/mlm-p1-continue-ab-{start,bound}.json`.
- Logi ramion: `logs/mlm-p1-continue-ab-{smoke,kl,kl-mnrl}.log`.
- Wyjścia: `cache/mlm-p1-v2-continuation-ab-runs` na dysku WSL, przez symlinki z `runs/`.
- Wyniki: `results/data/mlm-p1-v2-continuation-ab.json` oraz
  `results/summary_5070_mlm_p1_continuation_ab.md`.
