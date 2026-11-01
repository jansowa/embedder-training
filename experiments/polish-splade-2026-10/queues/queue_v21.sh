#!/usr/bin/env bash
# 29.10 day, replaces queue_v20 after its current step (soup-v6d heldout8 in the leaderboard
# regime). The user's 5070 Ti arms arrived as one zip with five models
# (datasets/transfer/in/fiqa-selected-models-20261029.zip): long-easy, short-easy, v3-pp,
# splade-mix2-pp, band-far. Tiny there: long-easy 62.43 (best single arm), v3-pp 62.25,
# band-far 62.23, short-easy 61.24, mix2-pp unknown (its benchmark ran out of memory).
# Order (deadline: no step starts after 19:15):
#   1  unpack into models/<name>-5070
#   2  long-easy: tiny, sparsity, length diagnostic, held-out-8, leaderboard regime;
#      soup-v7a = n1 + n3 + n4 + long-easy (soup-v6a plus long-easy): tiny, held-out, leaderboard
#   3  short-easy: tiny, length diagnostic, held-out-8 (the length pair of long-easy)
#   4  v3-pp: tiny, held-out-8 (the old-negative control of long-easy), leaderboard when near
#   5  mix2-pp: tiny, sparsity; band-far: tiny, held-out-8
#   6  held-out-2 in the leaderboard regime: soup-v6a, then soup-v7a when it beats soup-v6a
#      on the leaderboard heldout8, then n4-5070
# Start with: setsid nohup bash scripts/queue_v21.sh > logs/queue_v21.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v21 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v21.pid"
DEADLINE=$(date -d "2026-10-29 19:15" +%s)
before_deadline() { [ "$(date +%s)" -lt "$DEADLINE" ]; }
ZIP=datasets/transfer/in/fiqa-selected-models-20261029.zip

# take over from v20 once its current regime step has written its file
waited=0
until [ -f results-regimes/quantized/soup-v6d/heldout8.json ] || ! kill -0 "$(cat "$LOGS/queue_v20.pid" 2>/dev/null)" 2>/dev/null; do
    [ "$waited" -ge 5400 ] && break
    sleep 30; waited=$((waited + 30))
done
qp=$(cat "$LOGS/queue_v20.pid" 2>/dev/null)
if [ -n "$qp" ] && kill -0 "$qp" 2>/dev/null; then
    kids=$(pgrep -P "$qp"); kill -TERM "$qp" 2>/dev/null
    for k in $kids; do kill -TERM -- "-$k" 2>/dev/null || kill -TERM "$k" 2>/dev/null; done
    log "v20 (pid $qp) stopped after soup-v6d; children: ${kids:-none}"
    for n in $(seq 1 12); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break; sleep 10; done
fi
wait_gpu_free

reaches() { [ -n "$(tiny "$1")" ] && [ -n "$2" ] && [ "$(ge "$(tiny "$1")" "$2")" = 1 ]; }
leaderboard() { regime quantized "$1" "models/$1" tiny10 16; regime quantized "$1" "models/$1" heldout8 16; }
lb() { "$PY" -c "import json; print(round(json.load(open('results-regimes/quantized/$1/$2.json'))['mean_ndcg@10'], 2))" 2>/dev/null; }
soup() {  # soup NAME MODEL... : equal-weight soup into models/NAME, then tiny + sparsity
    local name="$1"; shift
    if [ ! -d "models/$name" ]; then
        need_space || return 1; wait_gpu_free; wait_ram
        "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
            && log "soup $name = $*" || { log "soup $name FAILED (logs/merge_$name.log)"; return 1; }
    fi
    bench "$name" "models/$name"; sparsity "$name" "models/$name"
}
diag() {  # diag NAME: the length-bias diagnostic of models/NAME on the own FiQA corpus
    local out="datasets/arms/diag-$1"
    [ -f "results/data/length-bias-$1.json" ] && return 0
    wait_gpu_free; wait_ram
    timeout -k 60 1800 "$PY" scripts/fiqa_splade_mine.py --source-dir datasets/arms/fiqa-v3-mine --out-dir "$out" --model "models/$1" --top-k 100 > "$LOGS/diag_$1.log" 2>&1 \
        && "$PY" scripts/fiqa_splade_length_bias.py --mine-dir "$out" --model models/base-splade-wrapped --report "results/data/length-bias-$1.json" >> "$LOGS/diag_$1.log" 2>&1 \
        && log "diag $1: $("$PY" -c "
import json; r = json.load(open('results/data/length-bias-$1.json')); b = r['positives_by_token_bucket']
print('positive first by length ' + ', '.join(f\"{k} {v['positive_first']:.2f}\" for k, v in b.items()) + f\"; top-1 over 512 tokens {r['student_top1']['share_over_512']:.0%}\")")" \
        || log "diag $1 FAILED (logs/diag_$1.log)"
    rm -rf "$out"
}
heldout2() {
    local t; for t in czywiesz gpt35wiki poquad quora treccovid; do before_deadline || return 0; regime quantized "$1" "models/$1" "heldout2_$t" 16; done
    log "  $1 held-out-2: $("$PY" -c "
import json, glob
v = [json.load(open(f))['mean_ndcg@10'] for f in sorted(glob.glob('results-regimes/quantized/$1/heldout2_*.json'))]
print(f'{sum(v)/len(v):.2f} over {len(v)} tasks' if v else 'no result')")"
}
step() { before_deadline || { log "deadline reached; skipping: $*"; return 1; }; "$@"; }

REF=$(tiny arm-cluster-like-fiqa); BASE=$(tiny base-splade-wrapped); NEAR=$(fsub "$REF" 0.3)
log "=== v21 start: reference ${REF:-?}, base ${BASE:-?}, soup-v6a tiny $(tiny soup-v6a) leaderboard $(lb soup-v6a tiny10) / $(lb soup-v6a heldout8), $(free_gb) GB free ==="

log "=== v21 step 1: unpack ==="
tmp=models/.unpack-v21; rm -rf "$tmp"; mkdir -p "$tmp"
if unzip -q "$ZIP" -d "$tmp"; then
    for n in long-easy short-easy v3-pp splade-mix2-pp band-far; do
        src=$(dirname "$(find "$tmp" -path "*/$n/modules.json" | head -1)")
        [ -n "$src" ] && [ -f "$src/modules.json" ] && [ ! -e "models/$n-5070" ] && mv "$src" "models/$n-5070" && log "  models/$n-5070 ($(du -sh "models/$n-5070" | cut -f1))"
    done
    rm -rf "$tmp" && rm -f "$ZIP" && log "zip removed; $(free_gb) GB free"
else log "unzip FAILED"; rm -rf "$tmp"; fi

log "=== v21 step 2: long-easy ==="
m=long-easy-5070
if [ -d "models/$m" ]; then
    step bench "$m" "models/$m"; step sparsity "$m" "models/$m"; step diag "$m"
    step holdout "$m" "models/$m"; reaches "$m" "$NEAR" && step leaderboard "$m"
    log "  $m: tiny $(tiny $m) held-out $(hold $m) (cluster-like $(hold arm-cluster-like-fiqa), soup-v6a $(hold soup-v6a)) leaderboard $(lb $m tiny10) / $(lb $m heldout8)"
    step soup soup-v7a models/n1-5070 models/n3-5070 models/n4-5070 "models/$m"
    if [ -n "$(tiny soup-v7a)" ]; then
        step holdout soup-v7a models/soup-v7a; step leaderboard soup-v7a
        log "  soup-v7a: tiny $(tiny soup-v7a) held-out $(hold soup-v7a) leaderboard $(lb soup-v7a tiny10) / $(lb soup-v7a heldout8) vs soup-v6a $(tiny soup-v6a) / $(hold soup-v6a) / $(lb soup-v6a tiny10) / $(lb soup-v6a heldout8)"
    fi
fi

log "=== v21 step 3: short-easy ==="
m=short-easy-5070
[ -d "models/$m" ] && { step bench "$m" "models/$m"; step sparsity "$m" "models/$m"; step diag "$m"; reaches "$m" "$BASE" && step holdout "$m" "models/$m"; log "  $m: tiny $(tiny $m) held-out $(hold $m)"; }

log "=== v21 step 4: v3-pp ==="
m=v3-pp-5070
[ -d "models/$m" ] && { step bench "$m" "models/$m"; step sparsity "$m" "models/$m"; reaches "$m" "$BASE" && step holdout "$m" "models/$m"; log "  $m: tiny $(tiny $m) held-out $(hold $m)"; }

log "=== v21 step 5: mix2-pp, band-far ==="
m=splade-mix2-pp-5070
[ -d "models/$m" ] && { step bench "$m" "models/$m"; step sparsity "$m" "models/$m"; log "  $m: tiny $(tiny $m)"; }
m=band-far-5070
[ -d "models/$m" ] && { step bench "$m" "models/$m"; step sparsity "$m" "models/$m"; reaches "$m" "$BASE" && step holdout "$m" "models/$m"; log "  $m: tiny $(tiny $m) held-out $(hold $m)"; }

log "=== v21 step 6: held-out-2 in the leaderboard regime ==="
step heldout2 soup-v6a
a=$(lb soup-v7a heldout8); b=$(lb soup-v6a heldout8)
[ -n "$a" ] && [ -n "$b" ] && [ "$(ge "$a" "$b")" = 1 ] && step heldout2 soup-v7a
step heldout2 n4-5070

"$PY" scripts/queue_summary.py > results/summary_v21.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v21.md)"
log "=== v21 done ($(date '+%d.%m %H:%M')) ==="
