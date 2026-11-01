#!/usr/bin/env bash
# Morning of 31.10 on the 3060 Ti, unattended: waits for queue v23 to exit, fills the gap until
# 10:00 (no training starts after 09:30, no held-out after 09:20).
#   1  GradCache speed probes: cluster-like recipe at a true batch of 32, three steps, mini-batch
#      4 / 8 / 16 with and without gradient checkpointing; step time and peak memory into
#      results/data/gradcache-smoke.json (the 30.10 smoke: mini-batch 4 + checkpointing, 3.3 GB).
#   2  seed3 of the reference recipe: tiny, sparsity, held-out-8.
#   3  seed4 of the reference recipe: tiny, sparsity; held-out-8 when it starts before 09:20.
#   4  results/summary_v24.md: the per-task noise table over seeds 42, 1, 2, 3, 4 and the probe
#      table; commit; push.
# Start: setsid nohup bash scripts/queue_v24.sh > logs/queue_v24.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v24 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v24.pid"
guard_alive
DEADLINE=$(date -d "2026-10-31 09:30" +%s)
HOLD_DEADLINE=$(date -d "2026-10-31 09:20" +%s)
before_deadline() { [ "$(date +%s)" -lt "$DEADLINE" ]; }
step() { before_deadline || { log "deadline reached; skipping: $*"; return 1; }; "$@"; }
C=configs/next/3060

if [ -f "$LOGS/queue_v23.pid" ] && kill -0 "$(cat "$LOGS/queue_v23.pid")" 2>/dev/null; then
    log "waiting for queue v23 (pid $(cat "$LOGS/queue_v23.pid")) to exit"
    while kill -0 "$(cat "$LOGS/queue_v23.pid")" 2>/dev/null; do sleep 60; done
    sleep 30
fi

dims() { sparsity_mean "results/sparsity/$1.json" "$1"; }
train2() {  # train2 TAG CFG [FALLBACK]: train; when the 2 x 16 run fails, retry once with the cached-loss config
    local tag="$1" cfg="$2" fb="${3:-}"
    [ -d "models/arm-$tag" ] && return 0
    train "$tag" "$cfg" && return 0
    [ -d "models/arm-$tag" ] && return 0
    { [ -n "$fb" ] && [ -f "$fb" ]; } || return 1
    mv -f "$LOGS/arm_$tag.log" "$LOGS/arm_$tag.first.log" 2>/dev/null
    log "train arm-$tag: first attempt failed ($(grep -m1 -oE 'OutOfMemoryError|CUDA out of memory' "$LOGS/arm_$tag.first.log" 2>/dev/null || echo 'see logs/arm_'"$tag"'.first.log')); retry with $fb"
    train "$tag" "$fb"
}
probe() {  # probe TAG CFG: one GradCache speed probe (3 steps); appends a line to the probe log
    local tag="$1" cfg="$2" out mem s e peak t
    out="$LOGS/smoke_$tag.log"; mem="$LOGS/smoke_$tag.mem"
    grep -q "^$tag " "$PROBES" 2>/dev/null && return 0
    wait_gpu_free; wait_ram
    rm -rf "runs/smoke-$tag"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$mem" 2>/dev/null & local mempid=$!
    s=$SECONDS
    timeout -k 60 900 "$PY" run_experiments.py --grid-configuration-file "$cfg" --backend sentence-transformers --training-type splade > "$out" 2>&1
    e=$((SECONDS - s))
    kill "$mempid" 2>/dev/null; wait "$mempid" 2>/dev/null
    peak=$(sort -n "$mem" 2>/dev/null | tail -1); peak=${peak:-0}
    t=$(grep -oE "3/3 \[[0-9:]+<[0-9:]+, *[0-9.]+s/it" "$out" | tail -1 | grep -oE "[0-9.]+s/it" | tr -d 's/it')
    if grep -qE "OutOfMemoryError|CUDA out of memory" "$out"; then
        echo "$tag OOM ${e} ${peak} -" >> "$PROBES"; log "probe $tag: CUDA OOM after ${e}s (peak ${peak} MiB)"
    elif [ -n "$t" ]; then
        echo "$tag ok ${e} ${peak} ${t}" >> "$PROBES"; log "probe $tag: ${t} s/step, peak ${peak} MiB, ${e}s total"
    else
        echo "$tag FAILED ${e} ${peak} -" >> "$PROBES"; log "probe $tag FAILED (logs/smoke_$tag.log)"
    fi
    rm -rf "runs/smoke-$tag"
}
PROBES=results/data/gradcache-smoke.txt
mkdir -p results/data

REF=$(tiny arm-cluster-like-fiqa)
log "=== v24 start: reference $REF, $(free_gb) GB free ==="

log "=== v24 step 1: GradCache speed probes (true batch 32, 3 steps) ==="
for mb in 4 8 16; do for gc in gc nogc; do step probe "cb32-mb$mb-$gc" "$C/smoke_cb32_mb${mb}_$gc.yaml"; done; done
"$PY" - <<'EOF' 2>> "$LOGS/queue_summary.log" && log "probe table written (results/data/gradcache-smoke.json)"
import json
rows = []
for line in open("results/data/gradcache-smoke.txt"):
    tag, status, total, peak, step = line.split()
    rows.append({"probe": tag, "status": status, "total_s": int(total), "peak_mib": int(peak), "s_per_step": None if step == "-" else float(step)})
json.dump({"recipe": "cluster-like, true batch 32, 512 tokens, 10 negatives, fp16, pooling chunk 8, 3 optimizer steps", "probes": rows},
          open("results/data/gradcache-smoke.json", "w"), indent=2)
EOF

log "=== v24 step 2: seed3 of the reference recipe ==="
step train2 mnrl-fiqa-seed3 "$C/mnrl_fiqa_seed3.yaml" "$C/mnrl_fiqa_seed3_cb32.yaml" \
    && { step bench arm-mnrl-fiqa-seed3 models/arm-mnrl-fiqa-seed3; step sparsity arm-mnrl-fiqa-seed3 models/arm-mnrl-fiqa-seed3
         [ "$(date +%s)" -lt "$HOLD_DEADLINE" ] && holdout arm-mnrl-fiqa-seed3 models/arm-mnrl-fiqa-seed3
         log "  arm-mnrl-fiqa-seed3: tiny $(tiny arm-mnrl-fiqa-seed3) held-out $(hold arm-mnrl-fiqa-seed3) (seed 42: $REF / $(hold arm-cluster-like-fiqa)), $(dims arm-mnrl-fiqa-seed3) dims/doc"; }

log "=== v24 step 3: seed4 of the reference recipe ==="
step train2 mnrl-fiqa-seed4 "$C/mnrl_fiqa_seed4.yaml" "$C/mnrl_fiqa_seed4_cb32.yaml" \
    && { step bench arm-mnrl-fiqa-seed4 models/arm-mnrl-fiqa-seed4; step sparsity arm-mnrl-fiqa-seed4 models/arm-mnrl-fiqa-seed4
         if [ "$(date +%s)" -lt "$HOLD_DEADLINE" ]; then holdout arm-mnrl-fiqa-seed4 models/arm-mnrl-fiqa-seed4; else log "  seed4 held-out skipped (after 09:20)"; fi
         log "  arm-mnrl-fiqa-seed4: tiny $(tiny arm-mnrl-fiqa-seed4) held-out $(hold arm-mnrl-fiqa-seed4) (seed 42: $REF / $(hold arm-cluster-like-fiqa)), $(dims arm-mnrl-fiqa-seed4) dims/doc"; }

log "=== v24 step 4: summary, commit, push ==="
{
    echo "# Queue v24 (morning of 31.10): seed noise floor and GradCache speed probes"; echo
    echo "## Per-task noise of the reference recipe (truncated regime)"; echo
    "$PY" scripts/seed_noise_table.py --models arm-cluster-like-fiqa arm-mnrl-fiqa-seed1 arm-mnrl-fiqa-seed2 arm-mnrl-fiqa-seed3 arm-mnrl-fiqa-seed4 2>> "$LOGS/queue_summary.log"
    echo; echo "## GradCache speed probes (cluster-like recipe, true batch 32, 3 steps; the 2 x 16 reference: 14 s per 32 queries)"; echo
    echo "| probe | status | s/step | peak MiB | total s |"; echo "|---|---|---:|---:|---:|"
    while read -r tag status total peak stp; do echo "| $tag | $status | $stp | $peak | $total |"; done < "$PROBES"
    echo; echo "## Queue state"; echo; echo '```'; grep -E "^(train|bench|holdout|sparsity):" "$STATE" | tail -30; echo '```'
} > results/summary_v24.md
for p in results/arm-mnrl-fiqa-seed3 results/arm-mnrl-fiqa-seed4 results-holdout/arm-mnrl-fiqa-seed3 results-holdout/arm-mnrl-fiqa-seed4 results/sparsity results/data/gradcache-smoke.json results/data/gradcache-smoke.txt results/summary_v24.md; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v24.out"
done
if git commit -q -m "Add the v24 results: seeds 3 and 4 of the reference recipe and the GradCache speed probes" 2>> "$LOGS/queue_v24.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v24.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v24.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v24.log)"
else
    log "nothing to commit or commit failed (see logs/queue_v24.out)"
fi
log "=== v24 done ($(date '+%d.%m %H:%M')) ==="
