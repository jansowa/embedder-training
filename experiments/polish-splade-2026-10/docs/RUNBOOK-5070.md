# Runbook: nocna kolejka na 5070 Ti (`scripts/queue_5070.sh`)

Instrukcja dla agenta (Codex) na maszynie z RTX 5070 Ti 16 GB. Cel: uruchomić
`scripts/queue_5070.sh` tak, żeby przez noc przeliczyła sześć ramion fine-tuningu
polish-SPLADE, soupy, held-out i reżim leaderboardowy, a rano były wyniki w
`results/summary_5070.md`. Co kolejka robi i dlaczego: nagłówek
`scripts/queue_5070.sh` oraz `docs/PLAN-2026-10-26.md`.

Zasady, których nie łamiemy (nauczka z maszyny 3060):

- jeden job GPU naraz; kolejka sama pilnuje wolnej karty, RAM i dysku,
- nie edytujemy skryptu bash, który właśnie działa (`queue_5070.sh`); poprawka =
  zabić i uruchomić ponownie (kolejka pomija kroki, które mają już wyniki),
- nie używamy `pkill -f <wzorzec>` z wzorcem, który występuje we własnej linii
  poleceń,
- nie commitujemy `models/`, `datasets/`, `logs/`, `runs/`, `cache/`, `wandb/`.

## 0. Windows z WSL: gdzie i jak uruchamiać

Komputer to Windows, projekt działa wyłącznie w WSL (dystrybucja Linux, np.
Ubuntu). Wszystkie polecenia z tego runbooka wykonujemy w shellu WSL (Windows
Terminal z profilem Ubuntu albo `wsl.exe`), nie w PowerShell ani cmd.

- Repo, `.venv`, JDK i dane trzymamy w systemie plików Linuksa (np.
  `~/work/embedder-training`), nie na `/mnt/c/...`: na dyskach Windows I/O
  jest wielokrotnie wolniejsze, a uprawnienia i symlinki nie działają
  poprawnie. Jeśli repo leży na `/mnt/c`, sklonować je ponownie w `~`.
- Klonować gitem z WSL, nie gitem z Windows: skrypty bash z końcówkami CRLF nie
  działają (`$'\r': command not found`). Kontrola:
  `grep -c $'\r' scripts/queue_lib.sh scripts/queue_5070.sh` musi dać 0 i 0,
  `git config --get core.autocrlf` puste albo `false`.
- GPU: sterownik NVIDIA instaluje się tylko po stronie Windows; w WSL
  `nvidia-smi` ma działać od razu, bez instalowania sterowników w Linuksie.
  Torch z locka (`+cu128`) działa w WSL bez zmian.
- RAM: WSL2 domyślnie dostaje połowę RAM hosta. Kolejka wymaga co najmniej
  6 GB wolnego (`free -g` w WSL). Jeśli jest mniej, ustawić w Windows plik
  `C:\Users\<user>\.wslconfig`:

  ```
  [wsl2]
  memory=24GB
  swap=8GB
  ```

  i wykonać `wsl --shutdown` z PowerShell przed ponownym wejściem do WSL.
  Tylko przed startem kolejki, nigdy w trakcie: `wsl --shutdown` zabija
  wszystkie procesy w WSL.
- Dysk: `df -h .` w WSL pokazuje wirtualny dysk ext4 (plik `ext4.vhdx` na
  dysku Windows), więc wolne miejsce na tym dysku Windows musi być co najmniej
  30 GB.
- Noc: procesy uruchomione przez `setsid nohup` żyją tylko tak długo, jak
  instancja WSL, a Windows potrafi ją zamknąć po zamknięciu ostatniego
  terminala. Dlatego: zostawić otwarte okno terminala WSL z
  `tail -f logs/queue.log`, wyłączyć na noc usypianie i hibernację Windows
  (opcje zasilania, uśpienie „nigdy" przy zasilaniu sieciowym), nie
  wylogowywać się, nie uruchamiać `wsl --shutdown`.
- Archiwum z Google Drive pobrane w Windows jest w WSL widoczne jako
  `/mnt/c/Users/<user>/Downloads/transfer-5070-20261026.tar.gz`; rozpakować
  je do repo w systemie plików Linuksa (sekcja 1, punkt 2).

## 1. Co musi być na maszynie

1. Repo `jansowa/embedder-training`, gałąź `master`, co najmniej commit
   `f1ceecc` (kolejki, konfigi `configs/next/5070/`, wyniki referencyjne) i
   submoduł `third_party/pirb` na gałęzi `agent/splade-perf` (łatka
   `quantize_queries`):

   ```shell
   git pull
   git submodule update --init --recursive
   git -C third_party/pirb log --oneline -1   # oczekiwane: f836e46 Let the SPLADE query encoder quantize weights ...
   ```

2. Archiwum z Google Drive `transfer-5070-20261026.tar.gz` (392 MB,
   sha256 zaczyna się od `4a972ef8ef2b37aa`), rozpakowane w katalogu repo:

   ```shell
   tar xzf /sciezka/do/transfer-5070-20261026.tar.gz -C /sciezka/do/repo
   ```

   Po rozpakowaniu muszą istnieć (ścieżki względem katalogu repo):

   | plik / katalog | rozmiar | po co |
   |---|---|---|
   | `datasets/arms/fiqa-2073.jsonl` | 21 MB | ramię N2 |
   | `datasets/arms/fiqa-plus-squadv2.jsonl` | 41 MB | ramiona N1, N5, N6 |
   | `datasets/arms/fiqa-plus-squadv2-10k.jsonl` | 102 MB | ramię N3 |
   | `datasets/arms/mix3-6219.jsonl` | 50 MB | ramię N4 |
   | `models/soup-v2b/` | 316 MB | soupy b/c i reżim leaderboardowy |

   Wyniki referencyjne (`results/ck-warmup-s`, `results/soup-v2b`,
   `results-holdout/soup-of-ft`, `results-holdout/soup-v2b`,
   `results-holdout/arm-fiqa-plus-squadv2`) przychodzą z gita.

3. Środowisko Pythona wg `README.md`, ale lock ma `torch==2.7.1+cu128`
   (`sm_120`, czyli Blackwell jest wspierany), więc indeks PyTorch to `cu128`,
   nie `cu126` z README:

   ```shell
   uv venv --python 3.10.4 .venv
   uv pip sync requirements.lock \
       --extra-index-url https://download.pytorch.org/whl/cu128 \
       --index-url https://pypi.org/simple \
       --index-strategy unsafe-best-match
   ```

4. JDK 21 dla Anserini (pyserini 0.39). `scripts/queue_lib.sh` bierze
   `JAVA_HOME`, jeśli wskazuje na JDK, a inaczej szuka `~/opt/jdk-21*` i
   `/usr/lib/jvm/*21*`. Instalacja bez ruszania systemu (z `README-env`):

   ```shell
   mkdir -p ~/opt && cd ~/opt
   curl -fsSL -o jdk21.tar.gz \
       "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.5%2B11/OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz"
   tar xzf jdk21.tar.gz && rm jdk21.tar.gz
   ```

5. Weights & Biases. Konfigi mają `report_to: [wandb]`; bez zalogowania
   trening w `nohup` (brak tty) pada na starcie. Albo `wandb login` przed
   startem, albo start kolejki z `WANDB_MODE=offline` (patrz punkt 4).

6. Dysk: minimum 30 GB wolnego w katalogu repo. Kolejka pobiera dane PIRB do
   `third_party/pirb/data` (18 zadań, ok. 6 GB, z czego 5,7 GB to
   `poleval-2022`), buduje indeksy per model (kasowane po benchmarku), zapisuje
   6 checkpointów po 320 MB w `models/` i `runs/`. Modele i zbiory z HF są
   publiczne, token nie jest potrzebny.

## 2. Sprawdzenie przed startem

Wszystko z katalogu repo, `.venv` aktywowane lub przez `.venv/bin/python`.

```shell
git log --oneline -1 && git -C third_party/pirb status -sb | head -1
grep -c $'\r' scripts/queue_lib.sh scripts/queue_5070.sh   # WSL: oba 0, inaczej repo sklonowane gitem z Windows
pwd | grep -q "^/mnt/" && echo "UWAGA: repo na dysku Windows, przenieś do ~" || echo "fs ok"
ls -la datasets/arms/{fiqa-2073,fiqa-plus-squadv2,fiqa-plus-squadv2-10k,mix3-6219}.jsonl models/soup-v2b/model.safetensors
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -c "import sentence_transformers, pyserini; print(sentence_transformers.__version__)"
bash -c 'QUEUE_TAG=probe source scripts/queue_lib.sh; echo JAVA_HOME=$JAVA_HOME; java -version'
JAVA_HOME=$(bash -c 'QUEUE_TAG=probe source scripts/queue_lib.sh; echo $JAVA_HOME') .venv/bin/python -c "from pyserini.search.lucene import LuceneImpactSearcher; print('jvm ok')"
bash -n scripts/queue_5070.sh scripts/queue_lib.sh
.venv/bin/python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('configs/next/5070/*.yaml')]; print('yaml ok')"
df -h .
```

Oczekiwane: `java -version` pokazuje 21.x, `torch.cuda.is_available()` True,
import `LuceneImpactSearcher` nie rzuca `jnius.JavaException`.

## 3. Smoke test treningu (10 minut, zanim pójdzie noc)

Kopia konfigu N2 ograniczona do trzech kroków, bez wandb, uruchomiona
dokładnie tą samą ścieżką co kolejka (`run_experiments.py` z gridem):

```shell
sed -e 's|^runs_dir: .*|runs_dir: runs/smoke-5070|' \
    -e 's|^    num_train_epochs: .*|    num_train_epochs: 1\n    max_steps: 3|' \
    configs/next/5070/n2_cluster_fiqa_2073.yaml > configs/next/5070/smoke_5070.yaml
WANDB_MODE=disabled timeout 900 .venv/bin/python run_experiments.py \
    --grid-configuration-file configs/next/5070/smoke_5070.yaml \
    --backend sentence-transformers --training-type splade 2>&1 | tail -30
ls runs/smoke-5070/sentence-transformers/splade/*/final
```

Sukces = katalog `final/` z `model.safetensors`. Pierwsze uruchomienie
pobiera `sdadas/polish-splade` i NanoBEIR z HF (chwila). Po teście usunąć
`runs/smoke-5070` i `configs/next/5070/smoke_5070.yaml` (nie commitować).
Jeśli trening pada na OOM już tutaj, coś jest nie tak ze środowiskiem (ten
konfig trenował na 8 GB).

## 4. Start kolejki

```shell
mkdir -p logs
setsid nohup bash scripts/queue_5070.sh > logs/queue_5070.out 2>&1 &
sleep 20; tail -5 logs/queue.log
```

Bez zalogowanego wandb: `WANDB_MODE=offline setsid nohup bash scripts/queue_5070.sh ...`.

W WSL po starcie zostawić w tym oknie `tail -f logs/queue.log` na całą noc
(sekcja 0: zamknięcie ostatniego terminala może zamknąć instancję WSL razem
z kolejką).

Kolejka zapisuje PID w `logs/queue_5070.pid`, postęp w `logs/queue.log`
(jedna linia na krok, `->` = wynik, `FAILED` = krok pominięty) i tabelę w
`logs/queue_state.tsv`. Log każdego treningu: `logs/arm_<nazwa>.log`, każdego
benchmarku: `logs/bench_<nazwa>.log`, `logs/holdout_<nazwa>.log`,
`logs/regime_*.out`.

Orientacyjne czasy na 16 GB: trening ramienia 15–40 min (N3 ok. 2× dłużej:
10 073 wierszy), tiny 10–15 min, held-out 30–40 min, reżim quantized ok.
15 min. Całość 6–8 godzin.

## 5. Monitorowanie i awarie

- `tail -f logs/queue.log`, `nvidia-smi`, `free -g`, `df -h .`.
- Krok, który padł, jest logowany i pomijany; kolejka idzie dalej. Po nocy
  wystarczy sprawdzić linie `FAILED` i `no checkpoint`.
- N5 (batch fizyczny 16) może dostać OOM; to jest dopuszczalne, nic od niego
  nie zależy. Jeśli OOM dotknie N1, przerwać kolejkę (patrz niżej), zmniejszyć
  `splade_pooling_chunk_size` do 8 w konfigach `n1`–`n4`, `n6` i uruchomić
  ponownie.
- Zatrzymanie kolejki: `kill $(cat logs/queue_5070.pid)`, potem zabić proces
  `run_experiments.py` i jego dziecko `python -m training.train` /
  `run_benchmark.py` po PID z `ps -eo pid,args | grep "[r]un_experiments"`.
  `scripts/orphan_guard.py` (uruchamiany przez kolejkę) dobija osierocone
  procesy GPU po 30 s.
- Jeśli `need_space` czeka na 10 GB wolnego: zwolnić miejsce (najpierw
  `third_party/pirb/cache/*/_*models_*` po nieudanych benchmarkach, potem
  `runs/`), kolejka sama ruszy.
- Ponowny start po poprawce: ten sam `setsid nohup ...`; wytrenowane ramiona
  (`models/arm-*`), benchmarki (`results/*/final/metrics.json`) i held-outy są
  pomijane.

## 6. Co raportować rano

1. `results/summary_5070.md` i tabela: dla każdego z N1–N6 i soupów
   `results/<nazwa>/final/metrics.json` → `epoch0/pirb_average_ndcg@10` (tiny),
   `results-holdout/<nazwa>/final/metrics.json` (held-out-8) dla dwóch
   najlepszych, `results-regimes/quantized/` dla najlepszego i soup-v2b.
2. Punkty odniesienia (tiny / held-out-8, reżim truncated, ten sam co w
   kolejce; odczyt: `epoch0/pirb_average_ndcg@10` w `results/<m>/final/metrics.json`
   i `results-holdout/<m>/final/metrics.json`):

   | model | tiny | held-out-8 |
   |---|---|---|
   | polish-splade bazowy (`results/base-splade-wrapped`, `results-holdout/base`) | 60.14 | 38.09 |
   | ck-warmup-s | 61.49 | 39.97 |
   | arm-fiqa-plus-squadv2 | 61.84 | 40.43 |
   | soup-of-ft | 61.87 | 40.37 |
   | soup-v2b | 62.07 | 40.32 |

   Nowy model jest ciekawy, gdy bije soup-v2b na obu kolumnach o więcej niż
   0.3; różnice do 0.3 to szum (trzy seedy tej samej receptury na 3060 dały
   61.38–61.68 na tiny).
3. Linie `FAILED` z `logs/queue.log` z jednozdaniową przyczyną z odpowiedniego
   logu.
4. Nic nie commitować poza ewentualnymi poprawkami konfigów/skryptów; wyniki i
   modele zostają lokalnie do przejrzenia.
