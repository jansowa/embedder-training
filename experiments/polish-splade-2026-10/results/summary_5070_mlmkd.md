# Sonda A od MLM z destylacją PolDense-400M

Aktualizacja: 2026-11-06T15:49:07+01:00

Dawka docelowa: te same 137 773 przykłady po bramce i rozwinięciu pozytywów co w nocy 3. KL, T=2, bez normalizacji wyników; surowe cosinusy z siedmiu plików -tpd400.

S=121.2 jest wartością tymczasową do smoke. Pełny trening wymaga podania S i osobnego „tak” użytkownika.

| Model | krok | tiny10 przy 512 | wym./dok FP32 przy 512 |
|---|---:|---:|---:|
| A MNRL, noc 3 | 2150 | 55.14 | 339.06 |
| A MNRL, noc 3 | 4306 | 56.73 | 459.13 |
| ModernBERT MNRL, noc 3 | 4306 | 53.75 | 327.85 |
| baza | — | 61.60 | — |
| A KD | 2150 | — | — |
| A KD | 4306 | — | — |

Werdykt: Oczekuje na zatwierdzony pełny trening i pomiary.

Progi pozostają stałe: pasmo 327–458; ≥59.5 / <58.0 / pomiędzy. Punkt kontrolny ewentualnego dalszego biegu: ≥61.6 przy około 500 k przykładów.

## Smoke 20 kroków

S smoke: 121.2; kroki: 20; czas kroku: 2.60 s; peak CUDA: 9593 MiB; peak NVIDIA sampled: 11141 MiB.

Run smoke usunięty po zachowaniu telemetrii. save_strategy=no, report_to=[], WANDB_MODE=disabled.

Run sondy i smoke na WSL. Limit nowego zużycia przestrzeni podczas treningu: 25 GiB (smoke: 4 GiB), w ramach odzyskanych 30 GiB. P1 pozostaje wstrzymane z braku 60 GiB fizycznego miejsca i modelu soup-v9b.
