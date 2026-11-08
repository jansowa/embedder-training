#!/usr/bin/env bash
# Queue s8 (3060 Ti, night 8/9.11, ~2.5 h, before v44): the static query table after s7, document side frozen (r2v12-bias005).
# s7: credit-k8-kd 59.76 -> phr2-k8-kd 60.01 -> phr2-k8-kd-r2 60.20 (tiny10), each step 0.1-0.2, i.e. near the noise.
#   1  heldout8 for the two tables that matter (s6 credit-k8-kd, s7 phr2-k8-kd-r2): is the +0.44 real?
#   2  phr2m3: phrase rows from 3 queries instead of 5 (more words and pairs), the s7 path (s6 pool 2 epochs, round-2 pool 3 epochs)
#   3  round 3: a third mined pool (new sample, seed 11) with phr2-k8-kd-r2 as the student, phr2-k8-kd-r2 trained on it
#   4  heldout8 for the best of steps 2-3 when it beats phr2-k8-kd-r2 on tiny10 by >= 0.15
#   5  summary_s8.md, commit, push
# No CPU timing here (the user's rule): the speed of the phrase tables is measured by hand on the user's go.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=s8 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_s8.pid"
guard_alive
DOC_MODEL=${DOC_MODEL:-models/arm-dense-kl-n5kj-ens400-r2v12-bias005}
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
POOL=cache/static_pools/kd-r2v12b
POOL2=cache/static_pools/kd-r2v12b-r2
POOL3=cache/static_pools/kd-r2v12b-r3
SM="models/static-r2v12b"
RS="$RESULTS_ROOT/static-r2v12b"
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
gte() { "$PY" -c "import sys;a,b,d=sys.argv[1:4];print(int(bool(a) and bool(b) and float(a)>=float(b)+float(d)))" "$1" "$2" "$3"; }
mkdir -p "$RS"
log "=== s8 start: static query tables over $DOC_MODEL; $(free_gb) GB free on / ==="

strain() {  # strain NAME INIT POOL EPOCHS
    local name="$1" init="$2" pool="$3" ep="$4"
    [ -f "$SM/$name/static_query.json" ] && return 0
    [ -f "$SM/$init/static_query.json" ] || { log "train $name: init $init missing"; return 1; }
    wait_gpu_free; wait_ram; local s=$SECONDS; log "train $name from $init on $(basename "$pool"), $ep epochs"
    timeout -k 60 3600 "$PY" -m distill.train_static --pool "$pool" --init "$SM/$init" --out "runs/static-r2v12b-$name" --lr 5e-3 \
        --target rows --learn-scale --epochs "$ep" --batch-size 64 --warmup-steps 200 --schedule linear --val-rows 2000 --eval-every 500 \
        --log-every 100 --device cuda --seed 42 > "$LOGS/static_train_r2v12b_$name.log" 2>&1
    [ -f "runs/static-r2v12b-$name/final/static_query.json" ] || { log "train $name FAILED"; record "static_train:r2v12b:$name" FAILED "-" "$((SECONDS - s))s"; return 1; }
    cp -r "runs/static-r2v12b-$name/final" "$SM/$name"; cp "runs/static-r2v12b-$name/static_summary.json" "$RS/$name.json"
    log "train $name done in $((SECONDS - s))s"; record "static_train:r2v12b:$name" ok "-" "$((SECONDS - s))s"
}
asym() {  # asym NAME TASKSET
    [ -f "$SM/$1/static_query.json" ] || return 1
    PIRB_QUERY_MODEL="$SM/$1" regime quantized "asym-r2v12b-q-static-$1" "$DOC_MODEL" "$2" 16
    log "  static $1 over r2v12-bias005: $2 $(lbv "asym-r2v12b-q-static-$1" "$2")"
}
mine() {  # mine OUT SEED STUDENT...
    local out="$1" seed="$2"; shift 2
    [ -f "$out/pool.json" ] && return 0
    local studs=(); for st in "$@"; do studs+=(--student "$SM/$st"); done
    need_space || return 1; wait_gpu_free; wait_ram; local s=$SECONDS; log "mine $out (seed $seed, students $*)"
    timeout -k 60 7200 "$PY" -m distill.static_mine --doc-model "$DOC_MODEL" --corpus "$CORPUS" --out "$out" "${studs[@]}" \
        --device cuda --dtype fp16 --n-docs 250000 --n-queries 120000 --teacher-top 32 --student-top 16 --max-candidates 64 \
        --batch-size 16 --search-batch 128 --seed "$seed" > "$LOGS/static_mine_$(basename "$out").log" 2>&1
    [ -f "$out/pool.json" ] && { log "mine done in $((SECONDS - s))s"; record "static_mine:$(basename "$out")" ok "-" "$((SECONDS - s))s"; } \
        || { log "mine FAILED"; record "static_mine:$(basename "$out")" FAILED "-" "$((SECONDS - s))s"; return 1; }
}

log "=== s8 step 1: heldout8 for credit-k8-kd (s6) and phr2-k8-kd-r2 (s7) ==="
asym credit-k8-kd heldout8
asym phr2-k8-kd-r2 heldout8

log "=== s8 step 2: phrase rows from 3 queries (phr2m3) ==="
if [ ! -f "$SM/phr2m3-k32/static_query.json" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS; log "build phr2m3-k32"
    timeout -k 60 5400 "$PY" -m distill.static_query build --out "$SM/phr2m3-k32" --init credit --teacher "$DOC_MODEL" --texts "$CORPUS" \
        --kinds query --top-k 32 --min-count 3 --batch-size 64 --device cuda --dtype fp16 --phrases 2 --phrase-min-count 3 \
        --max-phrases 400000 > "$LOGS/static_build_r2v12b_phr2m3.log" 2>&1
    [ -f "$SM/phr2m3-k32/static_query.json" ] && log "build phr2m3 done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$SM/phr2m3-k32/static_query.json'));print('phrases', d['phrases'])")" \
        || log "build phr2m3 FAILED"
fi
[ -f "$SM/phr2m3-k8/static_query.json" ] || { [ -f "$SM/phr2m3-k32/static_query.json" ] && "$PY" -m distill.static_query prune --model "$SM/phr2m3-k32" --out "$SM/phr2m3-k8" --top-k 8 > "$LOGS/static_prune_r2v12b_phr2m3.log" 2>&1; }
strain phr2m3-k8-kd phr2m3-k8 "$POOL" 2
strain phr2m3-k8-kd-r2 phr2m3-k8-kd "$POOL2" 3
asym phr2m3-k8-kd-r2 tiny10

log "=== s8 step 3: round 3 ==="
mine "$POOL3" 11 phr2-k8-kd-r2 && strain phr2-k8-kd-r3 phr2-k8-kd-r2 "$POOL3" 3 && asym phr2-k8-kd-r3 tiny10

log "=== s8 step 4: heldout8 for a new best ==="
REF=$(lbv asym-r2v12b-q-static-phr2-k8-kd-r2 tiny10)
for name in phr2m3-k8-kd-r2 phr2-k8-kd-r3; do
    v=$(lbv "asym-r2v12b-q-static-$name" tiny10)
    if [ "$(gte "$v" "$REF" 0.15)" = 1 ]; then log "$name $v beats phr2-k8-kd-r2 $REF by >= 0.15: heldout8"; asym "$name" heldout8
    else log "$name ${v:-?} vs phr2-k8-kd-r2 $REF: no heldout8"; fi
done

log "=== s8 step 5: summary ==="
SUM="$RESULTS_ROOT/summary_s8.md"
"$PY" - "$RS" "$REGIMES_ROOT/quantized" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
rs, reg = Path(sys.argv[1]), Path(sys.argv[2])
def nd(name, ts):
    f = reg / f"asym-r2v12b-q-static-{name}" / f"{ts}.json"
    return f"{json.load(open(f))['mean_ndcg@10']:.2f}" if f.exists() else "-"
print("# s8 (night 8/9.11): static query tables over r2v12-bias005, heldout8 check and two more steps (quantized regime, nDCG@10)\n")
print("References: r2v12-bias005 symmetric 64.21 / 43.08; BM25 pyserini (leaderboard) 50.07 / 33.34, bm25s 53.75 / 36.17; polish-splade 61.53 / 41.06.\n")
print("| table | what | tiny10 | heldout8 | val top1 | dims/q | postings fraction |\n|---|---|---:|---:|---:|---:|---:|")
desc = {"credit-k8-kd": "s6: token rows", "phr2-k8-kd-r2": "s7: + words and pairs (min 5), round-2 pool",
        "phr2m3-k8-kd-r2": "phrases from 3 queries, round-2 pool", "phr2-k8-kd-r3": "s7 best + round-3 pool"}
for name, what in desc.items():
    f = rs / f"{name}.json"
    v = json.load(open(f)).get("val_final", {}) if f.exists() else {}
    print(f"| {name} | {what} | {nd(name, 'tiny10')} | {nd(name, 'heldout8')} | {v.get('top1_agreement', 0):.3f} | {v.get('dims_per_query', 0):.1f} | {v.get('postings_fraction', 0):.3f} |")
print("\nval columns: 2 000 held-out kd_v1 queries of the pool each table last trained on (not comparable across pools).")
EOF
log "summary written to $SUM"
git add "$SUM" "$RS" "$REGIMES_ROOT"/quantized/asym-r2v12b-q-static-* "$EXP_ROOT/queues/queue_s8.sh" 2>/dev/null
git commit -q -m "Add the s8 results: heldout8 of the static query tables and two more training steps" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_s8.log" 2>&1 && { echo pushed >> "$LOGS/push_s8.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== s8 done ($(date '+%d.%m %H:%M')) ==="
