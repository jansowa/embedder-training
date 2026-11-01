#!/usr/bin/env bash
# Runs the queue_v6 steps that were skipped while / was full on Monday 26.10
# (13:52–18:50) as soon as the regime run v6 is in right now finishes: the FiQA
# pool arms (block H) and the cluster-like FiQA arm (block C). Then relaunches
# queue_v6, which skips everything already done and continues with E, V1, D',
# F, C2, G, V2, I. Serial, one GPU job at a time, same guards as queue_lib.sh.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v7 MIN_RAM_GB=6
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v7.pid"
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
# v6 will be relaunched below; its hold-out of mlm-path-ep3 (55.45 on tiny) is not worth 35 min
sed -i 's|^\[ -d models/mlm-path-ep3 \] && holdout mlm-path-ep3 models/mlm-path-ep3$|# dropped 26.10 (55.45 on tiny): [ -d models/mlm-path-ep3 ] \&\& holdout mlm-path-ep3 models/mlm-path-ep3|' scripts/queue_v6.sh
guard_alive; wait_gpu_free
log "=== v7 start: $(free_gb) GB free on /, $(free -g | awk '/^Mem:/ {print $7}') GB RAM available ==="

# ---- H: FiQA pool arms (skipped in v6: disk full) ------------------------------
log "=== v7 block H: FiQA pool arms (gate vs base-loss selection) ==="
"$PY" scripts/build_pool_arms.py 2>&1 | while read -r l; do log "  pool arms: $l"; done
for name in fiqa-pool-rand-2073 fiqa-pool-hard-2073 fiqa-pool-gate-hard-2073; do
    cfg="configs/weekend/pool/arm_${name//-/_}.yaml"
    [ -f "$cfg" ] && train "$name" "$cfg" && bench "arm-$name" "models/arm-$name"
done
for name in fiqa-pool-rand-2073 fiqa-pool-hard-2073 fiqa-pool-gate-hard-2073; do
    v=$(tiny "arm-$name"); ref=$(tiny ck-warmup-s)
    [ -n "$v" ] && [ "$(ge "$v" "$(fsub "$ref" 0.3)")" = 1 ] && holdout "arm-$name" "models/arm-$name"
done

# ---- C: cluster-like inputs on FiQA (skipped in v6: disk full) -------------------
log "=== v7 block C: cluster-like inputs (512, 10 negatives) on FiQA ==="
train cluster-like-fiqa configs/weekend/cluster_like_fiqa_3060.yaml && { bench arm-cluster-like-fiqa models/arm-cluster-like-fiqa; holdout arm-cluster-like-fiqa models/arm-cluster-like-fiqa; }

"$PY" scripts/queue_summary.py > results/summary_v7.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v7.md)"
log "=== v7 done; relaunching queue_v6 for its remaining blocks (E, V1, D', F, C2, G, V2, I) ==="
setsid nohup bash scripts/queue_v6.sh > "$LOGS/queue_v6.out" 2>&1 &
sleep 5; log "queue_v6 relaunched (pid $(cat "$LOGS/queue_v6.pid"))"
