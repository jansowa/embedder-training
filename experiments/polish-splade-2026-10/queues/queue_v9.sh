#!/usr/bin/env bash
# Follow-up to the cluster-like finding (26.10 evening). arm-cluster-like-fiqa
# (FiQA 2073, 512 tokens, 10 negatives, batch 2 x 16) scored 62.27 on tiny, the
# best single arm so far (ck-warmup-s 61.49 on the same rows at 256 / 6,
# soup-v2b 62.07). The 5070 Ti night builds on that recipe, so this machine
# answers the questions it leaves open:
#   A1  arm-fiqa-512-6neg          512 tokens, 6 negatives      (is it the length?)
#   A2  arm-fiqa-256-10neg         256 tokens, 10 negatives     (is it the negatives?)
#   A3  arm-cluster-like-fiqa-s43  same recipe, seed 43          (does it replicate?)
#   S   soup-v3a = 0.5 soup-v2b + 0.5 arm-cluster-like-fiqa
#       soup-v3b = 0.5 arm-cluster-like-fiqa + 0.5 arm-cluster-like-fiqa-s43
#       soup-v3c = soup-v2b recipe with the new arm in place of arm-fiqa-plus-squadv2
#                  (0.5 soup-of-ft + 0.25 arm-squadv2-2073 + 0.25 arm-cluster-like-fiqa)
#   HO  held-out-8 for A1 and A2 (the ablation needs it either way) and for the
#       two best of A3 / soups that are within 0.3 of arm-cluster-like-fiqa on tiny
#   Q   old-pyserini regime (leaderboard numbers) tiny10 + heldout8 for the best
#       new model if it beats arm-cluster-like-fiqa on tiny
# Runs the held-out of arm-cluster-like-fiqa first (queue_v8 was stopped at
# 21:20 while its bench hung on the wandb upload; v8's pool arms move to
# queue_v6 block H), and relaunches queue_v6 at the end (E, V1, H, D', F, G,
# V2, I; C2 dropped = arm N1 on the 5070).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v9 MIN_RAM_GB=6
# 26.10 21:15: api.wandb.ai is unreachable from this machine (connect timeout);
# the bench of arm-cluster-like-fiqa hung 10+ min on "uploading data" after
# writing its metrics. Every job tonight logs offline; sync in the morning with
#   .venv/bin/wandb sync wandb/offline-run-*
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v9.pid"
V8_PID=$(cat "$LOGS/queue_v8.pid" 2>/dev/null || echo 0)

kill_tree() {  # kill_tree PID: terminate a process and all its descendants
    local pid="$1" kids
    kids=$(ps -o pid= --ppid "$pid" 2>/dev/null)
    for k in $kids; do kill_tree "$k"; done
    kill "$pid" 2>/dev/null
}

# ---- take over from v8 once its held-out of arm-cluster-like-fiqa is logged ----
if kill -0 "$V8_PID" 2>/dev/null; then
    log "waiting for queue_v8 (pid $V8_PID) to log the held-out of arm-cluster-like-fiqa"
    while kill -0 "$V8_PID" 2>/dev/null && ! grep -q "^\[.*\] \[v8\] holdout arm-cluster-like-fiqa.* \(->\|FAILED\|already present\)" logs/queue.log; do sleep 30; done
    if kill -0 "$V8_PID" 2>/dev/null; then
        kill_tree "$V8_PID"; sleep 5
        for p in $(ps -eo pid,args | awk '/run_benchmark\.py|run_experiments\.py/ && !/awk/ {print $1}'); do kill "$p" 2>/dev/null; done
        sleep 10
        log "took over from queue_v8 (its pool arms are left to queue_v6 block H)"
    else
        log "queue_v8 finished on its own"
    fi
fi
guard_alive; wait_gpu_free
log "=== v9 start: $(free_gb) GB free on /, $(free -g | awk '/^Mem:/ {print $7}') GB RAM available ==="
REF=$(tiny arm-cluster-like-fiqa); log "reference: arm-cluster-like-fiqa tiny ${REF:-?}, soup-v2b $(tiny soup-v2b), ck-warmup-s $(tiny ck-warmup-s)"

# ---- C: the held-out v8 did not get to (stopped while its bench hung on wandb) ----
log "=== v9 block C: held-out-8 for arm-cluster-like-fiqa ==="
holdout arm-cluster-like-fiqa models/arm-cluster-like-fiqa

# ---- A: ablation + replication --------------------------------------------------
log "=== v9 block A: which lever (512 vs 10 negatives) and does it replicate ==="
train fiqa-512-6neg configs/weekend/ablation/fiqa_512_6neg.yaml && bench arm-fiqa-512-6neg models/arm-fiqa-512-6neg
train fiqa-256-10neg configs/weekend/ablation/fiqa_256_10neg.yaml && bench arm-fiqa-256-10neg models/arm-fiqa-256-10neg
train cluster-like-fiqa-s43 configs/weekend/ablation/cluster_like_fiqa_seed43.yaml && bench arm-cluster-like-fiqa-s43 models/arm-cluster-like-fiqa-s43

# ---- S: soups with the new arm ------------------------------------------------------
log "=== v9 block S: soups with arm-cluster-like-fiqa ==="
merge() {  # merge NAME --models ... (--weights ...|--alpha a)
    local name="$1"; shift
    [ -d "models/$name" ] && return 0
    "$PY" scripts/merge_splade_weights.py "$@" --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
        && log "merged $name" || { log "merge $name FAILED"; return 1; }
}
[ -d models/soup-v2b ] && merge soup-v3a --models models/soup-v2b models/arm-cluster-like-fiqa --alpha 0.5
[ -d models/arm-cluster-like-fiqa-s43 ] && merge soup-v3b --models models/arm-cluster-like-fiqa models/arm-cluster-like-fiqa-s43 --alpha 0.5
[ -d models/soup-of-ft ] && [ -d models/arm-squadv2-2073 ] && merge soup-v3c --models models/soup-of-ft models/arm-squadv2-2073 models/arm-cluster-like-fiqa --weights 0.5 0.25 0.25
for s in soup-v3a soup-v3b soup-v3c; do [ -d "models/$s" ] && bench "$s" "models/$s"; done

# ---- HO: held-out ---------------------------------------------------------------------
log "=== v9 block HO: held-out-8 ==="
for m in arm-fiqa-512-6neg arm-fiqa-256-10neg; do [ -d "models/$m" ] && holdout "$m" "models/$m"; done
thr=$(fsub "${REF:-0}" 0.3)
ranked=$(for m in arm-cluster-like-fiqa-s43 soup-v3a soup-v3b soup-v3c; do v=$(tiny "$m"); [ -n "$v" ] && [ "$(ge "$v" "$thr")" = 1 ] && echo "$v $m"; done | sort -rn | head -2 | awk '{print $2}')
for m in $ranked; do holdout "$m" "models/$m"; done
log "held-out reference: arm-cluster-like-fiqa $(hold arm-cluster-like-fiqa), soup-v2b $(hold soup-v2b), soup-of-ft $(hold soup-of-ft)"

# ---- Q: leaderboard regime for a new best ------------------------------------------------
best=""; best_v="${REF:-0}"
for m in arm-fiqa-512-6neg arm-fiqa-256-10neg arm-cluster-like-fiqa-s43 soup-v3a soup-v3b soup-v3c; do
    v=$(tiny "$m"); [ -n "$v" ] && [ "$(ge "$v" "$best_v")" = 1 ] && [ "$v" != "$best_v" ] && { best="$m"; best_v="$v"; }
done
if [ -n "$best" ]; then
    log "=== v9 block Q: old-pyserini regime for $best (tiny $best_v > arm-cluster-like-fiqa $REF) ==="
    for ts in tiny10 heldout8; do regime old-pyserini "$best" "models/$best" "$ts" 16; done
else
    log "v9 block Q skipped: nothing beats arm-cluster-like-fiqa on tiny"
fi

"$PY" scripts/queue_summary.py > results/summary_v9.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v9.md)"
log "=== v9 done; relaunching queue_v6 for its remaining blocks (E, V1, H, D', F, G, V2, I) ==="
setsid nohup bash scripts/queue_v6.sh > "$LOGS/queue_v6.out" 2>&1 &
sleep 5; log "queue_v6 relaunched (pid $(cat "$LOGS/queue_v6.pid"))"
