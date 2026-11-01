#!/usr/bin/env bash
# Overnight queue after the 22:37 reboot (Sunday 25/26.10). Serial, one GPU job
# at a time, no side jobs. Blocks:
#   0  wait for the validation chain (logs/pirb_ab/ab.log "[val] ALL DONE")
#   R1 the finalist in the two leaderboard-comparable regimes:
#      old-pyserini (fork + pyserini 0.21, float query weights) and
#      new-backend (upstream HEAD + splade-index); base + soup-of-ft; tiny10 + heldout8
#   S  fiqa+squadv2 arm (died at the v2->v3 takeover), soup v2 candidates
#   R2 regimes for ck-warmup-s and the soup v2 winner
#   D' trivia halves: train hard, tiny for rand/easy/hard, conditional held-out
#   E  second held-out set (5 unused tasks) in the old-pyserini regime, base + soup-of-ft
#   F  squadv2 halves, G wikisynth halves
#   I  index cost, HF metric files, summary table
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v5 MIN_RAM_GB=6
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v5.pid"
guard_alive

log "=== v5 start: $(free_gb) GB free on /, $(free -g | awk '/^Mem:/ {print $7}') GB RAM available ==="
waited=0
# the validation chain logs "[val] ALL DONE" at its end; the first (new-backend only)
# chain already left one such line, the A/B chain adds the second
until [ "$(grep -c '\[val\] ALL DONE' logs/pirb_ab/ab.log 2>/dev/null)" -ge 2 ] || [ "$waited" -ge 5400 ]; do sleep 30; waited=$((waited + 30)); done
log "validation chain: $(grep -c "finished" logs/pirb_ab/ab.log) runs logged; starting"
wait_gpu_free

log "=== v5 block R1: finalist in the leaderboard-comparable regimes ==="
for ts in tiny10 heldout8; do
    for m in base-splade-wrapped soup-of-ft; do
        regime old-pyserini "$m" "models/$m" "$ts" 16
        regime new-backend  "$m" "models/$m" "$ts" 16
    done
done
regime quantized base-splade-wrapped models/base-splade-wrapped tiny10 16   # sanity: should match old-pyserini

log "=== v5 block S: fiqa+squadv2 arm and soup v2 ==="
train fiqa-plus-squadv2 configs/weekend/arm_fiqa_plus_squadv2.yaml && { bench arm-fiqa-plus-squadv2 models/arm-fiqa-plus-squadv2; holdout arm-fiqa-plus-squadv2 models/arm-fiqa-plus-squadv2; }
SOUP_V2=""
if [ -d models/arm-fiqa-plus-squadv2 ] && [ -d models/arm-squadv2-2073 ]; then
    [ -d models/soup-v2a ] || "$PY" scripts/merge_splade_weights.py --models models/soup-of-ft models/arm-fiqa-plus-squadv2 --alpha 0.5 --output models/soup-v2a > "$LOGS/merge_soup-v2a.log" 2>&1 || log "merge soup-v2a FAILED"
    [ -d models/soup-v2b ] || "$PY" scripts/merge_splade_weights.py --models models/soup-of-ft models/arm-squadv2-2073 models/arm-fiqa-plus-squadv2 --weights 0.5 0.25 0.25 --output models/soup-v2b > "$LOGS/merge_soup-v2b.log" 2>&1 || log "merge soup-v2b FAILED"
    bench soup-v2a models/soup-v2a; bench soup-v2b models/soup-v2b
    SOUP_REF=$(tiny soup-of-ft); SOUP_V2_V=""
    for n in soup-v2a soup-v2b; do v=$(tiny "$n"); [ -n "$v" ] || continue; if [ -z "$SOUP_V2_V" ] || [ "$(ge "$v" "$SOUP_V2_V")" = 1 ]; then SOUP_V2="$n"; SOUP_V2_V="$v"; fi; done
    if [ -n "$SOUP_V2" ] && [ "$(ge "$SOUP_V2_V" "$(fsub "$SOUP_REF" 0.2)")" = 1 ]; then
        log "soup v2 winner $SOUP_V2 ($SOUP_V2_V vs soup-of-ft $SOUP_REF): held-out"; holdout "$SOUP_V2" "models/$SOUP_V2"
    else log "soup v2: best $SOUP_V2 ($SOUP_V2_V) not within 0.2 of soup-of-ft ($SOUP_REF)"; SOUP_V2=""; fi
else log "soup v2 skipped: SQuAD arms missing"; fi

log "=== v5 block R2: regimes for the runners-up ==="
for ts in tiny10 heldout8; do
    regime old-pyserini ck-warmup-s models/ck-warmup-s "$ts" 16
    regime new-backend  ck-warmup-s models/ck-warmup-s "$ts" 16
    [ -n "$SOUP_V2" ] && { regime old-pyserini "$SOUP_V2" "models/$SOUP_V2" "$ts" 16; regime new-backend "$SOUP_V2" "models/$SOUP_V2" "$ts" 16; }
done

log "=== v5 block D': trivia halves (answerability hypothesis) ==="
split_block trivia rand easy hard lowcov highcov

log "=== v5 block E: second held-out set (5 unused tasks), old-pyserini regime ==="
for m in base-splade-wrapped soup-of-ft; do
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do
        bs=16; case "$ts" in heldout2_poquad|heldout2_treccovid) bs=8;; esac
        regime old-pyserini "$m" "models/$m" "$ts" "$bs"
    done
done

log "=== v5 block F: SQuAD-v2 halves ==="
split_block squadv2 rand hard easy lowcov highcov
log "=== v5 block G: wikisynth halves ==="
split_block wikisynth rand hard easy lowcov highcov

log "=== v5 block I: index cost, HF metric files, summary ==="
wait_gpu_free; wait_ram
models_args=(--model base=models/base-splade-wrapped --model soup-of-ft=models/soup-of-ft --model ck-warmup-s=models/ck-warmup-s)
for d in models/soup-v2a models/soup-v2b models/arm-*; do [ -d "$d" ] && models_args+=(--model "$(basename "$d")=$d"); done
timeout -k 60 7200 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${models_args[@]}" \
    --output-json results/index_cost_v5.json > "$LOGS/index_cost_v5.log" 2>&1 && log "index-cost profile written (results/index_cost_v5.json)" || log "index-cost profile FAILED"
"$PY" scripts/build_hf_metric_files.py >> "$LOGS/build_hf_metric_files.log" 2>&1 && log "HF metric files written (results/lexical_difficulty/hf_upload)"
"$PY" scripts/queue_summary.py > results/summary_v5.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v5.md)"
log "=== v5 queue finished ==="
