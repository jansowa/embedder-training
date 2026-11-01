#!/usr/bin/env bash
# Night queue for the 5070 Ti (16 GB): six arms of polish-splade fine-tuning with
# the cluster-like recipe (512 tokens, 10 hard negatives, effective batch 32),
# then soups, then held-out for the two best, then the leaderboard-comparable
# (quantized-query) regime for the best. Serial, one GPU job at a time, with the
# guards of scripts/queue_lib.sh (idle GPU, >= 6 GB RAM, >= 10 GB disk, timeouts).
#
#   N1  FiQA 2073 + SQuAD-v2 2073                 configs/next/5070/n1_*.yaml
#   N2  FiQA 2073 (control)                       n2
#   N3  FiQA 2073 + 8000 fresh gated SQuAD-v2     n3   (data scale)
#   N4  FiQA + SQuAD-v2 + judged-clean MS MARCO   n4   (source diversity)
#   N5  N1 with physical batch 16 x 2             n5   (in-batch negatives; may OOM)
#   N6  N1 with margin-MSE distillation           n6   (exploratory)
#   S   soup-5070-a = equal soup of the MNRL arms that exist (N1..N5)
#       soup-5070-b = 0.5 soup-v2b + 0.5 soup-5070-a      (needs models/soup-v2b)
#       soup-5070-c = 0.5 soup-v2b + 0.5 best single arm
#   HO  held-out-8 for the two best new models by tiny
#   Q   quantized-query regime (= leaderboard numbers) for the best new model and soup-v2b
#
# Needs on this machine: the repo with .venv, JDK for Anserini (JAVA_HOME in
# queue_lib.sh), datasets/arms/{fiqa-2073,fiqa-plus-squadv2,fiqa-plus-squadv2-10k,mix3-6219}.jsonl,
# models/soup-v2b (for the S/Q steps; optional), results/ck-warmup-s + results/soup-v2b
# metrics for the comparison lines (optional). third_party/pirb/data fills itself.
#
#   setsid nohup bash scripts/queue_5070.sh > logs/queue_5070.out 2>&1 &
#   tail -f logs/queue.log
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=5070 MIN_RAM_GB=6
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_5070.pid"
guard_alive; wait_gpu_free
log "=== 5070 start: $(free_gb) GB free, $(free -g | awk '/^Mem:/ {print $7}') GB RAM, GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="

ARMS=(n1-cluster-fiqa-plus-squadv2 n2-cluster-fiqa-2073 n3-cluster-fiqa-plus-squadv2-10k n4-cluster-mix3-6219 n5-cluster-fiqa-plus-squadv2-b16 n6-mmse-fiqa-plus-squadv2)
cfg_of() { echo "configs/next/5070/${1%%-*}_$(echo "${1#*-}" | tr '-' '_').yaml"; }

# ---- arms: train, then tiny --------------------------------------------------------
for a in "${ARMS[@]}"; do
    cfg=$(cfg_of "$a")
    log "=== arm $a ($cfg) ==="
    train "$a" "$cfg" && bench "arm-$a" "models/arm-$a"
done

# ---- soups --------------------------------------------------------------------------
log "=== soups ==="
mnrl=(); for a in "${ARMS[@]:0:5}"; do [ -d "models/arm-$a" ] && mnrl+=("models/arm-$a"); done
if [ "${#mnrl[@]}" -ge 2 ] && [ ! -d models/soup-5070-a ]; then
    w=$("$PY" -c "n=${#mnrl[@]}; print(' '.join([str(round(1/n,6))]*n))")
    "$PY" scripts/merge_splade_weights.py --models "${mnrl[@]}" --weights $w --output models/soup-5070-a > "$LOGS/merge_soup-5070-a.log" 2>&1 \
        && log "soup-5070-a = equal soup of ${#mnrl[@]} arms" || log "merge soup-5070-a FAILED"
fi
[ -d models/soup-5070-a ] && bench soup-5070-a models/soup-5070-a
best=""; best_v=""
for a in "${ARMS[@]}"; do v=$(tiny "arm-$a"); [ -n "$v" ] && { [ -z "$best_v" ] || [ "$(ge "$v" "$best_v")" = 1 ]; } && { best="arm-$a"; best_v="$v"; }; done
log "best single arm on tiny: ${best:-none} (${best_v:-?})"
if [ -d models/soup-v2b ]; then
    [ -d models/soup-5070-a ] && [ ! -d models/soup-5070-b ] && "$PY" scripts/merge_splade_weights.py --models models/soup-v2b models/soup-5070-a --alpha 0.5 --output models/soup-5070-b > "$LOGS/merge_soup-5070-b.log" 2>&1
    [ -n "$best" ] && [ ! -d models/soup-5070-c ] && "$PY" scripts/merge_splade_weights.py --models models/soup-v2b "models/$best" --alpha 0.5 --output models/soup-5070-c > "$LOGS/merge_soup-5070-c.log" 2>&1
    for s in soup-5070-b soup-5070-c; do [ -d "models/$s" ] && bench "$s" "models/$s"; done
else
    log "models/soup-v2b not on this machine: soups b/c skipped"
fi

# ---- held-out-8 for the two best new models -------------------------------------------
log "=== held-out for the two best new models ==="
ranked=$(for m in soup-5070-a soup-5070-b soup-5070-c "${ARMS[@]/#/arm-}"; do v=$(tiny "$m"); [ -n "$v" ] && echo "$v $m"; done | sort -rn | head -2 | awk '{print $2}')
for m in $ranked; do holdout "$m" "models/$m"; done
ref=$(hold soup-of-ft); [ -n "$ref" ] && log "reference held-out: soup-of-ft $ref, soup-v2b $(hold soup-v2b), arm-fiqa-plus-squadv2 $(hold arm-fiqa-plus-squadv2)"

# ---- leaderboard-comparable regime for the best -------------------------------------------
log "=== quantized-query regime (leaderboard numbers) ==="
top=$(echo "$ranked" | head -1)
[ -n "$top" ] && regime quantized "$top" "models/$top" tiny10 16
[ -d models/soup-v2b ] && regime quantized soup-v2b models/soup-v2b tiny10 16

"$PY" scripts/queue_summary.py > results/summary_5070.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_5070.md)"
log "=== 5070 queue finished ==="
