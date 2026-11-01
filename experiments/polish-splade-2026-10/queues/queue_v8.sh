#!/usr/bin/env bash
# Second pass at the queue_v6 steps lost on Monday 26.10. queue_v7 ran them at
# 19:15-19:48 but both failed: the three FiQA pool arms crashed at load time on
# pool rows whose first positive is an empty string (build_pool_arms.py now
# drops such rows), and the cluster-like FiQA training finished (twice) but its
# config saved under runs/cluster-like-fiqa, not runs/arm-cluster-like-fiqa
# where train() looks (configs fixed; the checkpoint is recovered below).
# Takes over from queue_v6 at the end of its current regime run, benches and
# holds out the cluster-like arm, rebuilds and trains the pool arms, then
# relaunches queue_v6, which skips what is done and continues E, V1, D', F,
# C2, G, V2, I. Serial, one GPU job at a time, same guards as queue_lib.sh.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v8 MIN_RAM_GB=6
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v8.pid"
V6_PID=$(cat "$LOGS/queue_v6.pid" 2>/dev/null || echo 0)

kill_tree() {  # kill_tree PID: terminate a process and all its descendants
    local pid="$1" kids
    kids=$(ps -o pid= --ppid "$pid" 2>/dev/null)
    for k in $kids; do kill_tree "$k"; done
    kill "$pid" 2>/dev/null
}

# ---- take over from v6 at the end of its current regime run -------------------
if kill -0 "$V6_PID" 2>/dev/null; then
    n0=$(grep -c "^\[.*\] \[v6\] regime .* \(->\|FAILED\)" logs/queue.log)
    log "waiting for queue_v6 (pid $V6_PID) to finish its current regime run"
    while kill -0 "$V6_PID" 2>/dev/null && [ "$(grep -c "^\[.*\] \[v6\] regime .* \(->\|FAILED\)" logs/queue.log)" -le "$n0" ]; do sleep 30; done
    if kill -0 "$V6_PID" 2>/dev/null; then
        kill_tree "$V6_PID"; sleep 5
        for p in $(ps -eo pid,args | awk '/run_benchmark\.py|run_experiments\.py/ && !/awk/ {print $1}'); do kill "$p" 2>/dev/null; done
        sleep 10; rm -rf cache/regimes/*
        log "took over from queue_v6"
    else
        log "queue_v6 finished on its own"
    fi
fi
guard_alive; wait_gpu_free
log "=== v8 start: $(free_gb) GB free on /, $(free -g | awk '/^Mem:/ {print $7}') GB RAM available ==="

# ---- C: cluster-like inputs on FiQA (trained by v7, saved under the wrong runs_dir) ----
log "=== v8 block C: cluster-like inputs (512, 10 negatives) on FiQA ==="
if [ ! -d models/arm-cluster-like-fiqa ]; then
    ck=$(ls -1dt runs/cluster-like-fiqa/sentence-transformers/splade/*/final 2>/dev/null | head -1)
    [ -n "$ck" ] && cp -r "$ck" models/arm-cluster-like-fiqa && log "recovered $ck -> models/arm-cluster-like-fiqa"
fi
train cluster-like-fiqa configs/weekend/cluster_like_fiqa_3060.yaml && { bench arm-cluster-like-fiqa models/arm-cluster-like-fiqa; holdout arm-cluster-like-fiqa models/arm-cluster-like-fiqa; }

# ---- H: FiQA pool arms (rows with empty texts dropped) ----------------------------
log "=== v8 block H: FiQA pool arms (gate vs base-loss selection) ==="
"$PY" scripts/build_pool_arms.py 2>&1 | while read -r l; do log "  pool arms: $l"; done
for name in fiqa-pool-rand-2073 fiqa-pool-hard-2073 fiqa-pool-gate-hard-2073; do
    cfg="configs/weekend/pool/arm_${name//-/_}.yaml"
    [ -f "$cfg" ] && train "$name" "$cfg" && bench "arm-$name" "models/arm-$name"
done
for name in fiqa-pool-rand-2073 fiqa-pool-hard-2073 fiqa-pool-gate-hard-2073; do
    v=$(tiny "arm-$name"); ref=$(tiny ck-warmup-s)
    [ -n "$v" ] && [ "$(ge "$v" "$(fsub "$ref" 0.3)")" = 1 ] && holdout "arm-$name" "models/arm-$name"
done

"$PY" scripts/queue_summary.py > results/summary_v8.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v8.md)"
log "=== v8 done; relaunching queue_v6 for its remaining blocks (E, V1, D', F, C2, G, V2, I) ==="
setsid nohup bash scripts/queue_v6.sh > "$LOGS/queue_v6.out" 2>&1 &
sleep 5; log "queue_v6 relaunched (pid $(cat "$LOGS/queue_v6.pid"))"
