#!/usr/bin/env bash
# 28/29.10: the 3060 until 29.10 20:00 (the user's deadline), after queue_v19. One job at a
# time; no training (the student-negative arms at 2 x 16 run on the 16 GB machine). A loop
# with two sources of work:
#   priority: any model zip the user drops into models/ (models/<name>.zip, or a zip inside
#             models/<name>/) is unpacked and measured: tiny + sparsity, held-out-8 when it
#             reaches the base on tiny, the leaderboard regime (quantized tiny10 + heldout8)
#             when within 0.3 of arm-cluster-like-fiqa, and a soup with soup-v2b + n1-5070
#             (tiny) when it reaches the reference
#   pending steps, in order, while nothing new arrived:
#     D   length-bias diagnostic (scripts/fiqa_splade_mine.py + fiqa_splade_length_bias.py,
#         ~2 min each, no teacher): how often the positive ranks first by its length bucket,
#         for base, cluster-like, n1-5070, n4-5070, soup-v5e
#     S   soups: v6a = n1 + n3 + n4, v6b = n1 + n4, v6c = n4 + soup-v2b,
#         v6d = soup-v5e + cluster-like, v6e = soup-v5e + the KL v3 arm; then held-out-8 and
#         the leaderboard regime for the two best that reach the reference
#     H2  held-out-2 in the leaderboard regime (5 tasks, ~100 min per model) for n1-5070,
#         n4-5070 and the soup with the best quantized heldout8: the third leaderboard number
# No step starts after 19:15 so that the machine is free by 20:00.
# Start with: setsid nohup bash scripts/queue_v20.sh > logs/queue_v20.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v20 MIN_RAM_GB=6
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v20.pid"
DEADLINE=$(date -d "2026-10-29 19:15" +%s)
before_deadline() { [ "$(date +%s)" -lt "$DEADLINE" ]; }

waited=0
until grep -qE '=== v19 done ===' logs/queue_v19.out 2>/dev/null; do
    [ "$waited" -ge 28800 ] && { log "v20: v19 not finished after 8 h; exiting"; exit 1; }
    sleep 120; waited=$((waited + 120))
done
wait_gpu_free

reaches() { [ -n "$(tiny "$1")" ] && [ -n "$2" ] && [ "$(ge "$(tiny "$1")" "$2")" = 1 ]; }
leaderboard() { regime quantized "$1" "models/$1" tiny10 16; regime quantized "$1" "models/$1" heldout8 16; }
lb_heldout8() { "$PY" -c "import json; print(round(json.load(open('results-regimes/quantized/$1/heldout8.json'))['mean_ndcg@10'], 2))" 2>/dev/null; }
lb_tiny10() { "$PY" -c "import json; print(round(json.load(open('results-regimes/quantized/$1/tiny10.json'))['mean_ndcg@10'], 2))" 2>/dev/null; }
soup() {  # soup NAME MODEL... : equal-weight soup into models/NAME, then tiny + sparsity
    local name="$1"; shift
    if [ ! -d "models/$name" ]; then
        need_space || return 1; wait_gpu_free; wait_ram
        "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
            && log "soup $name = $*" || { log "soup $name FAILED (logs/merge_$name.log)"; return 1; }
    fi
    bench "$name" "models/$name"; sparsity "$name" "models/$name"
    log "  $name: tiny $(tiny "$name")"
}
diag() {  # diag NAME: the length-bias diagnostic of models/NAME on the own FiQA corpus
    local m="$1" out="datasets/arms/diag-$1"
    [ -f "results/data/length-bias-$1.json" ] && return 0
    [ -d "models/$1" ] || { log "diag $1: model missing"; return 1; }
    wait_gpu_free; wait_ram; local s=$SECONDS
    timeout -k 60 1800 "$PY" scripts/fiqa_splade_mine.py --source-dir datasets/arms/fiqa-v3-mine --out-dir "$out" --model "models/$1" --top-k 100 > "$LOGS/diag_$1.log" 2>&1 \
        && "$PY" scripts/fiqa_splade_length_bias.py --mine-dir "$out" --model models/base-splade-wrapped --report "results/data/length-bias-$1.json" >> "$LOGS/diag_$1.log" 2>&1 \
        && log "diag $1 ($((SECONDS - s))s): $("$PY" -c "
import json; r = json.load(open('results/data/length-bias-$1.json')); b = r['positives_by_token_bucket']
print('positive first by length ' + ', '.join(f\"{k} {v['positive_first']:.2f}\" for k, v in b.items()) + f\"; top-1 over 512 tokens {r['student_top1']['share_over_512']:.0%}\")")" \
        || log "diag $1 FAILED after $((SECONDS - s))s (logs/diag_$1.log)"
}
adopt() {  # adopt NAME: unpack the zip in models/NAME into a SparseEncoder directory
    local dir="models/$1" z tmp inner
    [ -f "$dir/config.json" ] && return 0
    z=$(ls -1 "$dir"/*.zip 2>/dev/null | head -1); [ -n "$z" ] || return 1
    tmp=$(mktemp -d "$dir/unpack.XXXX")
    unzip -q "$z" -d "$tmp" || { log "adopt $1: unzip failed"; rm -rf "$tmp"; return 1; }
    inner=$(find "$tmp" -name modules.json | head -1)
    [ -n "$inner" ] || { log "adopt $1: no modules.json in the zip"; rm -rf "$tmp"; return 1; }
    inner=$(dirname "$inner")
    mv "$inner"/* "$dir"/ && rm -rf "$tmp" && rm -f "$z" && log "adopt $1: unpacked $(basename "$z") ($(du -sh "$dir" | cut -f1))"
}
new_models() {  # bare zips in models/ get a directory; print the directories that hold a zip and no model yet
    local z n d
    for z in models/*.zip; do
        [ -f "$z" ] || continue
        n=$(basename "$z" .zip); n=$(echo "$n" | sed -E 's/-20[0-9]{6}T[0-9]{6}Z-[0-9]-[0-9]{3}$//')
        mkdir -p "models/$n" && mv "$z" "models/$n/" && log "zip $(basename "$z") -> models/$n/"
    done
    for d in models/*/; do
        d=${d%/}; [ -f "$d/config.json" ] && continue
        ls "$d"/*.zip >/dev/null 2>&1 && basename "$d"
    done
}
evaluate() {  # evaluate NAME: tiny, sparsity, held-out and leaderboard regime by thresholds, a soup when it reaches the reference
    local m="$1"
    bench "$m" "models/$m"; sparsity "$m" "models/$m"
    log "  $m: tiny $(tiny "$m") (reference $REF, base $BASE)"
    [ -n "$(tiny "$m")" ] || return 1
    reaches "$m" "$BASE" && holdout "$m" "models/$m" && log "  $m: held-out $(hold "$m") (cluster-like $(hold arm-cluster-like-fiqa))"
    reaches "$m" "$NEAR" && leaderboard "$m" && log "  $m: leaderboard $(lb_tiny10 "$m") / $(lb_heldout8 "$m")"
    reaches "$m" "$REF" && soup "soup-x-$m" "models/$m" models/soup-v2b models/n1-5070
    return 0
}
best_soups_eval() {  # held-out + leaderboard regime for the two best v6 soups that reach the reference
    local s v best=() ; local -a ranked
    ranked=($(for s in soup-v6a soup-v6b soup-v6c soup-v6d soup-v6e; do v=$(tiny "$s"); [ -n "$v" ] && echo "$v $s"; done | sort -rn | head -2 | awk '{print $2}'))
    for s in "${ranked[@]}"; do
        if reaches "$s" "$REF"; then holdout "$s" "models/$s"; leaderboard "$s"; log "  $s: tiny $(tiny "$s") held-out $(hold "$s") leaderboard heldout8 $(lb_heldout8 "$s")"
        else log "  $s: tiny $(tiny "$s") below the reference; no held-out"; fi
    done
}
best_soup_lb() {  # the soup with the best quantized heldout8 (fallback: the best tiny that reaches the reference)
    local s v best="" best_v=""
    for s in soup-v5c soup-v5d soup-v5e soup-v6a soup-v6b soup-v6c soup-v6d soup-v6e; do
        v=$(lb_heldout8 "$s"); [ -n "$v" ] || continue
        if [ -z "$best_v" ] || [ "$(ge "$v" "$best_v")" = 1 ]; then best="$s"; best_v="$v"; fi
    done
    [ -n "$best" ] || for s in soup-v5e soup-v5c soup-v5d; do reaches "$s" "$REF" && { best="$s"; break; }; done
    echo "$best"
}
heldout2() {  # heldout2 NAME: the five held-out-2 tasks in the leaderboard regime
    local t; [ -d "models/$1" ] || { log "heldout2 $1: model missing"; return 1; }
    for t in czywiesz gpt35wiki poquad quora treccovid; do before_deadline || return 0; regime quantized "$1" "models/$1" "heldout2_$t" 16; done
    log "  $1 held-out-2: $("$PY" -c "
import json, glob
v = [json.load(open(f))['mean_ndcg@10'] for f in sorted(glob.glob('results-regimes/quantized/$1/heldout2_*.json'))]
print(f'{sum(v)/len(v):.2f} over {len(v)} tasks' if v else 'no result')")"
}

REF=$(tiny arm-cluster-like-fiqa); BASE=$(tiny base-splade-wrapped); NEAR=$(fsub "$REF" 0.3)
log "=== v20 start: reference ${REF:-?}, base ${BASE:-?}, $(free_gb) GB free on /, deadline $(date -d @"$DEADLINE" '+%d.%m %H:%M') ==="

STEPS=(
    "diag base-splade-wrapped" "diag arm-cluster-like-fiqa" "diag n1-5070" "diag n4-5070" "diag soup-v5e"
    "soup soup-v6a models/n1-5070 models/n3-5070 models/n4-5070"
    "soup soup-v6b models/n1-5070 models/n4-5070"
    "soup soup-v6c models/n4-5070 models/soup-v2b"
    "soup soup-v6d models/soup-v5e models/arm-cluster-like-fiqa"
    "soup soup-v6e models/soup-v5e models/arm-kl-fiqa-v3-scaleauto"
    "best_soups_eval"
    "heldout2 n1-5070"
    "heldout2 n4-5070"
    "heldout2 \$(best_soup_lb)"
)
i=0
while before_deadline; do
    found=0
    for m in $(new_models); do
        found=1
        log "=== v20: new model $m ==="
        adopt "$m" && evaluate "$m" || log "  $m: not usable (zip or layout)"
    done
    [ "$found" = 1 ] && continue
    if [ "$i" -lt "${#STEPS[@]}" ]; then
        step="${STEPS[$i]}"; i=$((i + 1))
        log "=== v20 step $i/${#STEPS[@]}: $step ==="
        eval "$step"
    else
        sleep 300
    fi
done
"$PY" scripts/queue_summary.py > results/summary_v20.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v20.md)"
log "=== v20 done ($(date '+%d.%m %H:%M')) ==="
