#!/usr/bin/env bash
# Monday daytime queue: takes over from queue_v5 as soon as v5 has finished the
# old-pyserini held-out run of ck-warmup-s that was in flight on Monday 06:09
# (or when v5 reaches its block D'/E on its own), so that the decision-relevant
# work below runs before v5's lower-value tail. Serial, one GPU job at a time,
# same guards as queue_lib.sh.
#
#   R3  evaluation max_seq_length 256 vs 512 in the leaderboard regimes
#   H2  FiQA judged by Opus: answerable-only vs rejected pairs
#   H3  loss / coverage halves on the judge-accepted FiQA pairs only
#   M   MLM path, 3 epochs on judged-clean MS MARCO (epochs 1/2/3 benchmarked)
#   H   FiQA pool arms: random / hardest-by-base-loss / gate+hard, no reranker gate
#   C   cluster-like inputs (512, 10 negatives) on FiQA
#   E   second held-out set (5 unused tasks), old-pyserini regime, base + soup-of-ft
#   V1  old-pyserini regime for the runners-up (+ held-out-8 for ck-warmup-s, soup-v2b)
#   D'  trivia halves (answerability hypothesis), moved here from v5
#   F   SQuAD-v2 halves, C2 cluster-like FiQA+SQuAD, G wikisynth halves
#   V2  second held-out set in the new-backend regime
#   I   index cost, HF metric files, summary
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v6 MIN_RAM_GB=6
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v6.pid"
V5_PID=$(cat "$LOGS/queue_v5.pid" 2>/dev/null || echo 0)

kill_tree() {  # kill_tree PID: terminate a process and all its descendants
    local pid="$1" kids
    kids=$(ps -o pid= --ppid "$pid" 2>/dev/null)
    for k in $kids; do kill_tree "$k"; done
    kill "$pid" 2>/dev/null
}

# ---- take over from v5 ---------------------------------------------------------
V5_DONE_MARK="regime old-pyserini/ck-warmup-s/heldout8 (->|FAILED)|=== v5 block (D'|E)"
if kill -0 "$V5_PID" 2>/dev/null; then
    log "waiting for queue_v5 (pid $V5_PID) to finish the ck-warmup-s held-out run (or reach block D'/E)"
    while kill -0 "$V5_PID" 2>/dev/null && ! grep -q -E "$V5_DONE_MARK" logs/queue.log; do sleep 60; done
    if kill -0 "$V5_PID" 2>/dev/null; then
        kill_tree "$V5_PID"; sleep 5
        for p in $(ps -eo pid,args | awk '/run_benchmark\.py|run_experiments\.py/ && !/awk/ {print $1}'); do kill "$p" 2>/dev/null; done
        sleep 10; rm -rf cache/regimes/*
        log "took over from queue_v5"
    else
        log "queue_v5 finished on its own"
    fi
fi
guard_alive; wait_gpu_free
log "=== v6 start: $(free_gb) GB free on /, $(free -g | awk '/^Mem:/ {print $7}') GB RAM available ==="

# ---- R3: evaluation sequence length ---------------------------------------------
# Checkpoints trained at 256 carry max_seq_length=256; the upstream backend
# evaluates them at 256 (ck-warmup-s: arguana-pl 45.15 vs 37.14 at 512), our fork
# always at 512. Does 256 lift the soup too, and does it reproduce in the old regime?
log "=== v6 block R3: evaluation max_seq_length 256 vs 512 in the leaderboard regimes ==="
regime new-backend soup-of-ft-len256 models/soup-of-ft tiny10 16 256
regime old-pyserini ck-warmup-s-len256 models/ck-warmup-s tiny10 16 256
regime new-backend ck-warmup-s-len512 models/ck-warmup-s tiny10 16 512
a=$(regime_mean results-regimes/new-backend/soup-of-ft-len256/tiny10.json 2>/dev/null | awk '{print $1}')
b=$(regime_mean results-regimes/new-backend/soup-of-ft/tiny10.json 2>/dev/null | awk '{print $1}')
if [ -n "$a" ] && [ -n "$b" ] && [ "$(ge "$a" "$(fsub "$b" -0.3)")" = 1 ]; then
    log "R3: soup-of-ft at 256 beats 512 on tiny10 ($a vs $b): held-out-8 at 256"
    regime new-backend soup-of-ft-len256 models/soup-of-ft heldout8 16 256
else
    log "R3: soup-of-ft at 256 does not beat 512 by 0.3 on tiny10 (${a:-?} vs ${b:-?}); no held-out"
fi

# ---- H2: FiQA with the Opus judge applied -------------------------------------
log "=== v6 block H2: FiQA judged by Opus (answerable-only vs rejected) ==="
for name in fiqa-2073-clean fiqa-2073-rejected; do
    cfg="configs/weekend/judged/arm_${name//-/_}.yaml"
    [ -f "$cfg" ] && train "$name" "$cfg" && bench "arm-$name" "models/arm-$name"
done
[ -d models/arm-fiqa-2073-clean ] && holdout arm-fiqa-2073-clean models/arm-fiqa-2073-clean

# ---- H3: loss / coverage halves on judge-accepted FiQA pairs only --------------
log "=== v6 block H3: halves (rand/hard/easy/lowcov/highcov) on the 1898 answerable FiQA pairs ==="
split_block fiqaans rand hard easy lowcov highcov

# ---- M: MLM path ------------------------------------------------------------
log "=== v6 block M: MLM path, 3 epochs on judged-clean MS MARCO ==="
if [ ! -d models/mlm-path-ep3 ]; then
    need_space && { wait_gpu_free; wait_ram
    log "train mlm-path-3060"; s=$SECONDS
    timeout -k 60 14400 "$PY" run_experiments.py --grid-configuration-file configs/weekend/mlm_path_3060.yaml \
        --backend sentence-transformers --training-type splade > "$LOGS/mlm_path_3060.log" 2>&1
    run=$(ls -1dt runs/mlm-path-3060/sentence-transformers/splade/*/ 2>/dev/null | head -1)
    if [ -n "$run" ] && [ -d "$run/final" ]; then
        cp -r "$run/final" models/mlm-path-ep3
        for ep in 1 2; do ck=$(ls -1d "$run"/epoch-checkpoints/epoch-000$ep-* 2>/dev/null | head -1); [ -n "$ck" ] && cp -r "$ck" "models/mlm-path-ep$ep"; done
        log "train mlm-path-3060 done in $((SECONDS - s))s (checkpoints: $(ls -d models/mlm-path-ep* | xargs -n1 basename | tr '\n' ' '))"
        record "train:mlm-path-3060" ok "-" "$((SECONDS - s))s"
    else log "train mlm-path-3060: no final checkpoint after $((SECONDS - s))s"; record "train:mlm-path-3060" FAILED "-" "$((SECONDS - s))s"; fi; }
fi
for ep in 1 2 3; do [ -d "models/mlm-path-ep$ep" ] && bench "mlm-path-ep$ep" "models/mlm-path-ep$ep"; done
# dropped 26.10 (55.45 on tiny): [ -d models/mlm-path-ep3 ] && holdout mlm-path-ep3 models/mlm-path-ep3

# ---- H: FiQA pool arms --------------------------------------------------------
log "=== v6 block H: FiQA pool arms (gate vs base-loss selection) ==="
mkdir -p datasets/arms/pool
[ -s datasets/arms/pool/fiqa-pool-5500.jsonl ] || cp cache/huggingface_datasets/speakleash-fiqa_pl-main-train_pl-a18e389dfc0ce58c/dataset.jsonl datasets/arms/pool/fiqa-pool-5500.jsonl
if [ ! -s results/base_loss/fiqa-pool-5500.jsonl ]; then
    wait_gpu_free; wait_ram
    timeout -k 60 3600 "$PY" scripts/splade_pair_loss.py --model models/base-splade-wrapped --n 0 \
        --input datasets/arms/pool/fiqa-pool-5500.jsonl --per-row-dir results/base_loss > "$LOGS/base_loss_pool.log" 2>&1 \
        && log "per-row base loss for the FiQA pool written" || log "per-row base loss for the pool FAILED"
fi
"$PY" scripts/build_pool_arms.py 2>&1 | while read -r l; do log "  pool arms: $l"; done
for name in fiqa-pool-rand-2073 fiqa-pool-hard-2073 fiqa-pool-gate-hard-2073; do
    cfg="configs/weekend/pool/arm_${name//-/_}.yaml"
    [ -f "$cfg" ] && train "$name" "$cfg" && bench "arm-$name" "models/arm-$name"
done
for name in fiqa-pool-rand-2073 fiqa-pool-hard-2073 fiqa-pool-gate-hard-2073; do
    v=$(tiny "arm-$name"); ref=$(tiny ck-warmup-s)
    [ -n "$v" ] && [ "$(ge "$v" "$(fsub "$ref" 0.3)")" = 1 ] && holdout "arm-$name" "models/arm-$name"
done

# ---- C: cluster-like inputs on FiQA ------------------------------------------
log "=== v6 block C: cluster-like inputs (512, 10 negatives) on FiQA ==="
train cluster-like-fiqa configs/weekend/cluster_like_fiqa_3060.yaml && { bench arm-cluster-like-fiqa models/arm-cluster-like-fiqa; holdout arm-cluster-like-fiqa models/arm-cluster-like-fiqa; }

# ---- E: second held-out set, old-pyserini regime ------------------------------
log "=== v6 block E: second held-out set (5 unused tasks), old-pyserini regime ==="
for m in base-splade-wrapped soup-of-ft; do
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do
        bs=16; case "$ts" in heldout2_poquad|heldout2_treccovid) bs=8;; esac
        regime old-pyserini "$m" "models/$m" "$ts" "$bs"
    done
done

# ---- V1: old-pyserini regime for the runners-up --------------------------------
log "=== v6 block V1: old-pyserini regime for the runners-up ==="
regime old-pyserini soup-v2b models/soup-v2b heldout8 16
regime new-backend ck-warmup-s-len512 models/ck-warmup-s heldout8 16 512
for m in arm-squadv2-2073 ck-warmup-l arm-fiqa-rand arm-fiqa-hard arm-cluster-like-fiqa arm-fiqa-plus-squadv2; do
    [ -d "models/$m" ] || continue
    for ts in tiny10 heldout8; do regime old-pyserini "$m" "models/$m" "$ts" 16; done
done

# ---- D': trivia halves (from v5) ------------------------------------------------
log "=== v6 block D': trivia halves (answerability hypothesis) ==="
split_block trivia rand easy hard lowcov highcov

# ---- F, C2, G --------------------------------------------------------------------
log "=== v6 block F: SQuAD-v2 halves ==="
split_block squadv2 rand hard easy lowcov highcov
log "=== v6 block C2: cluster-like inputs on FiQA+SQuAD ==="
train cluster-like-fiqa-plus-squadv2 configs/weekend/cluster_like_fiqa_plus_squadv2_3060.yaml && { bench arm-cluster-like-fiqa-plus-squadv2 models/arm-cluster-like-fiqa-plus-squadv2; holdout arm-cluster-like-fiqa-plus-squadv2 models/arm-cluster-like-fiqa-plus-squadv2; }
log "=== v6 block G: wikisynth halves ==="
split_block wikisynth rand hard easy lowcov highcov

# ---- V2: second held-out set in the new-backend regime --------------------------
log "=== v6 block V2: second held-out set, new-backend regime ==="
for m in base-splade-wrapped soup-of-ft; do
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do
        bs=16; case "$ts" in heldout2_poquad|heldout2_treccovid) bs=8;; esac
        regime new-backend "$m" "models/$m" "$ts" "$bs"
    done
done

# ---- I ---------------------------------------------------------------------------
log "=== v6 block I: index cost, HF metric files, summary ==="
wait_gpu_free; wait_ram
models_args=(--model base=models/base-splade-wrapped --model soup-of-ft=models/soup-of-ft --model ck-warmup-s=models/ck-warmup-s)
for d in models/soup-v2a models/soup-v2b models/mlm-path-ep3 models/arm-*; do [ -d "$d" ] && models_args+=(--model "$(basename "$d")=$d"); done
timeout -k 60 7200 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${models_args[@]}" \
    --output-json results/index_cost_v6.json > "$LOGS/index_cost_v6.log" 2>&1 && log "index-cost profile written (results/index_cost_v6.json)" || log "index-cost profile FAILED"
"$PY" scripts/build_hf_metric_files.py >> "$LOGS/build_hf_metric_files.log" 2>&1 && log "HF metric files written"
"$PY" scripts/queue_summary.py > results/summary_v6.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v6.md)"
log "=== v6 queue finished ==="
