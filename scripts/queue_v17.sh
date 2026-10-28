#!/usr/bin/env bash
# 28.10, after queue_v16: the 5070 Ti run N3 (mix2 at 2 x 16: 2 student + 8 old negatives per
# query; tiny 62.01 there against the 62.27 reference, within seed noise) measured here where
# tiny cannot decide: held-out-8, the leaderboard regime and soups. Waits for "=== v16 done ==="
# and for models/n3-5070 (a directory, or a zip dropped inside it, unpacked here). Blocks:
#   A   adopt models/n3-5070 (unzip when needed), tiny + sparsity in our regime
#   H   held-out-8 for n3-5070; leaderboard regime (quantized tiny10 + heldout8) when its
#       held-out is within 0.2 of arm-cluster-like-fiqa
#   S   soup-v5c = n3-5070 + soup-v2b + n1-5070, soup-v5d = n3-5070 + n1-5070: tiny, and held-out +
#       leaderboard regime when tiny reaches arm-cluster-like-fiqa
# Start with: setsid nohup bash scripts/queue_v17.sh > logs/queue_v17.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v17 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v17.pid"

adopt() {  # adopt NAME: unpack a zip dropped in models/NAME into a SparseEncoder directory
    local dir="models/$1" z tmp inner
    [ -f "$dir/config.json" ] && return 0
    z=$(ls -1 "$dir"/*.zip 2>/dev/null | head -1); [ -n "$z" ] || return 1
    tmp=$(mktemp -d "$dir/unpack.XXXX")
    unzip -q "$z" -d "$tmp" || { log "adopt $1: unzip failed"; return 1; }
    inner=$(find "$tmp" -name config.json -path '*' | grep -vE '1_SpladePooling|/0_' | head -1)
    [ -n "$inner" ] || { log "adopt $1: no config.json in the zip"; return 1; }
    inner=$(dirname "$inner")
    mv "$inner"/* "$dir"/ && rm -rf "$tmp" && rm -f "$z" && log "adopt $1: unpacked $(basename "$z") ($(du -sh "$dir" | cut -f1))"
}
reaches() { [ -n "$(tiny "$1")" ] && [ -n "$2" ] && [ "$(ge "$(tiny "$1")" "$2")" = 1 ]; }
leaderboard() { regime quantized "$1" "models/$1" tiny10 16; regime quantized "$1" "models/$1" heldout8 16; }
soup() {  # soup NAME MODEL... : equal-weight soup into models/NAME, then tiny + sparsity
    local name="$1"; shift
    if [ ! -d "models/$name" ]; then
        need_space || return 1; wait_gpu_free; wait_ram
        "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
            && log "soup $name = $*" || { log "soup $name FAILED (logs/merge_$name.log)"; return 1; }
    fi
    bench "$name" "models/$name"; sparsity "$name" "models/$name"
}

waited=0
until grep -qE '=== v16 done ===' logs/queue_v16.out 2>/dev/null && { [ -f models/n3-5070/config.json ] || ls models/n3-5070/*.zip >/dev/null 2>&1; }; do
    [ "$waited" -ge 43200 ] && { log "v17: v16 not finished or models/n3-5070 absent after 12 h; exiting"; exit 1; }
    sleep 120; waited=$((waited + 120))
done
wait_gpu_free
REF=$(tiny arm-cluster-like-fiqa); HREF=$(hold arm-cluster-like-fiqa)
log "=== v17 start: reference tiny ${REF:-?}, held-out ${HREF:-?}, $(free_gb) GB free on / ==="

log "=== v17 block A: n3-5070 in our regime ==="
adopt n3-5070 || { log "models/n3-5070 unusable"; exit 1; }
bench n3-5070 models/n3-5070; sparsity n3-5070 models/n3-5070
log "  n3-5070: tiny $(tiny n3-5070) (5070 Ti reported 62.01)"

log "=== v17 block H: held-out and leaderboard regime for n3-5070 ==="
holdout n3-5070 models/n3-5070
h=$(hold n3-5070)
if [ -n "$h" ] && [ -n "$HREF" ] && [ "$(ge "$h" "$(fsub "$HREF" 0.2)")" = 1 ]; then leaderboard n3-5070
else log "leaderboard regime skipped: held-out ${h:-?} is more than 0.2 below the reference ${HREF:-?}"; fi

log "=== v17 block S: soups with n3-5070 ==="
soup soup-v5c models/n3-5070 models/soup-v2b models/n1-5070
soup soup-v5d models/n3-5070 models/n1-5070
for s in soup-v5c soup-v5d; do
    [ -f "results/$s/final/metrics.json" ] || continue
    if reaches "$s" "$REF"; then holdout "$s" "models/$s"; leaderboard "$s"; else log "  $s: tiny $(tiny "$s") below the reference $REF; no held-out"; fi
done

"$PY" scripts/queue_summary.py > results/summary_v17.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v17.md)"
log "=== v17 done ==="
