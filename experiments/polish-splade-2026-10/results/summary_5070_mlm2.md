# Probe M (04.11.2026 06:26): SPLADE from pl-ModernBERT-512-base vs probe A (distilroberta) at equal steps

Training head: `stock` (config `experiments/polish-splade-2026-10/configs/next/5070/modernbert_probe_m.yaml`; SPEEDUP-SPLADE-3060.md §6). Leaderboard regime (quantized, 512 tokens), tiny10 nDCG@10; dims = active document dimensions (base 436).

| step | examples | M tiny10 | M dims | A tiny10 | A dims | M − A |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 625 | 20000 | – | 0.00 | 54.71 | 418.50 | – |
| 1250 | 40000 | – | 0.00 | 54.73 | 434.90 | – |
| 1875 | 60000 | – | 0.00 | 55.70 | 390.80 | – |
| 2500 | 80000 | – | 0.00 | 56.33 | 297.20 | – |
| 3750 | 120000 | – | 0.00 | – | – | – |
| 5000 | 160000 | – | 0.00 | 55.37 | 310.60 | – |

**Gate GM (s05000):** FAIL techniczny — reprezentacje modelu zapadły się do zera. Brak poprawnego pomiaru nDCG; nie jest to zmierzony wynik tiny10 równy 0.

GM: PASS when M − A ≥ +2.0 and M ≥ 57.0 and dims in 300–600; MAYBE when the advantage is in [+0.8, +2.0); FAIL otherwise.
Next: PASS → ModernBERT stage 1 on fresh data (NEXT-SESSION-5070-MLM.md, night 3); MAYBE → one day of stage 1 with a hard ≥ 59.5 threshold; FAIL → branch A or B by G1.

## Diagnostyka po zakończeniu kolejki (4.11)

Oryginalny MLM daje niezerowe reprezentacje, lecz checkpointy 625 i 5000 dają
zerowe wektory zarówno w FP32, jak i BF16. Wszystkie sprawdzone wagi są skończone.
W kroku 250 średnia gęstość próbki treningowej wynosiła już tylko 3.29, a w kroku
500 spadła do zera wraz z normą gradientu. To załamanie treningu, nie uszkodzony
plik checkpointu ani wyłącznie błąd precyzji ewaluacji.

Osobno naprawiono nieobsługiwaną konwersję BF16 do NumPy dla dokumentów i zapytań
oraz błędne uznawanie 0 ukończonych zadań za sukces. Po poprawce ponowna ewaluacja
s05000 utworzyła 10 176 pustych wektorów i pusty indeks; zatrzymała się przy jego
otwieraniu w Pyserini. Brak wiarygodnego porównania jakości M z A. Tej porażki
przepisu nie należy traktować jako dowodu na sufit architektury ModernBERT.

Szczegóły: [diagnosis_5070_mlm2.md](diagnosis_5070_mlm2.md).
