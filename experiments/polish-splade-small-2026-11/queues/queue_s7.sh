#!/usr/bin/env bash
# Queue s7 (3060 Ti, 8.11 evening, ~75 min): better static query tables over the r2v12-bias005 index, document side frozen.
# s6: credit-k8-kd 59.76 tiny10 (r2v12-bias005 symmetric 64.21). Three ideas, each against that number:
#   phrases  rows for whole multi-piece words (phr1) and also adjacent word pairs (phr2), min 5 queries, at most 200 k,
#            credited from the teacher like the token rows, cut to k8, trained with the s6 recipe on the s6 pool
#   round 2  a second mined pool whose student candidates come from the trained tables (their own false positives),
#            the s6 table and the phr2 table trained further on it
#   longer   the round-2 runs train 3 epochs
# Reading: phr1 / phr2 >= 60.3 = phrase rows pay (> 0.5); r2 runs >= +0.2 over their start = mining again pays.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=s7 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_s7.pid"
guard_alive
DOC_MODEL=${DOC_MODEL:-models/arm-dense-kl-n5kj-ens400-r2v12-bias005}
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
POOL=cache/static_pools/kd-r2v12b
POOL2=cache/static_pools/kd-r2v12b-r2
SM="models/static-r2v12b"
RS="$RESULTS_ROOT/static-r2v12b"
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done
mkdir -p "$RS"
log "=== s7 start: phrase rows and round-2 mining for the static query table over $DOC_MODEL; $(free_gb) GB free on / ==="

log "=== s7 step 0: unit tests, code commit ==="
wait_ram
if ! timeout -k 30 900 "$PY" -m pytest -q tests/test_distill_static.py tests/test_distill_static_mine.py tests/test_distill_static_phrases.py tests/test_st_static.py > "$LOGS/s7_pytest.log" 2>&1 \
   || ! (cd third_party/pirb && timeout -k 30 600 "$PY" -m pytest -q tests/test_splade_static_query.py) >> "$LOGS/s7_pytest.log" 2>&1; then
    log "unit tests FAILED (logs/s7_pytest.log); stopping"; record "pytest:s7" FAILED "-"; exit 1
fi
log "unit tests passed: $(grep -E 'passed' "$LOGS/s7_pytest.log" | tr '\n' ' ')"
if [ -n "$(git -C third_party/pirb status --short search/splade.py tests/test_splade_static_query.py)" ]; then
    git -C third_party/pirb add search/splade.py tests/test_splade_static_query.py \
        && git -C third_party/pirb commit -q -m "Add phrase rows to the static query encoder" && log "submodule committed $(git -C third_party/pirb log --oneline -1)"
fi
git add third_party/pirb distill/static_query.py release/st_static.py tests/test_distill_static_phrases.py tests/test_st_static.py "$EXP_ROOT/queues/queue_s7.sh" 2>/dev/null
git commit -q -m "Add phrase rows (whole words and word pairs) to static query tables" 2>/dev/null && log "code committed $(git log --oneline -1)"

build() {  # build NAME PHRASES
    local name="$1" ph="$2"
    [ -f "$SM/$name-k32/static_query.json" ] || {
        wait_gpu_free; wait_ram; local s=$SECONDS; log "build $name-k32 (phrases $ph)"
        timeout -k 60 5400 "$PY" -m distill.static_query build --out "$SM/$name-k32" --init credit --teacher "$DOC_MODEL" --texts "$CORPUS" \
            --kinds query --top-k 32 --min-count 3 --batch-size 64 --device cuda --dtype fp16 --phrases "$ph" --phrase-min-count 5 \
            --max-phrases 200000 > "$LOGS/static_build_r2v12b_$name.log" 2>&1
        [ -f "$SM/$name-k32/static_query.json" ] || { log "build $name FAILED"; record "static_build:r2v12b:$name" FAILED "-" "$((SECONDS - s))s"; return 1; }
        log "build $name-k32 done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$SM/$name-k32/static_query.json'));print('rows', d['rows'], 'phrases', d['phrases'], 'credited', d['meta'].get('phrases_credited'))")"
        record "static_build:r2v12b:$name" ok "-" "$((SECONDS - s))s"
    }
    [ -f "$SM/$name-k8/static_query.json" ] || "$PY" -m distill.static_query prune --model "$SM/$name-k32" --out "$SM/$name-k8" --top-k 8 > "$LOGS/static_prune_r2v12b_$name.log" 2>&1
}
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
    log "train $name done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$RS/$name.json'));i,f=d['val_init'],d['val_final'];print('val top1 %.3f -> %.3f, dims/q %.1f -> %.1f' % (i['top1_agreement'], f['top1_agreement'], i['dims_per_query'], f['dims_per_query']))")"
    record "static_train:r2v12b:$name" ok "-" "$((SECONDS - s))s"
}
asym() {
    [ -f "$SM/$1/static_query.json" ] || return 1
    PIRB_QUERY_MODEL="$SM/$1" regime quantized "asym-r2v12b-q-static-$1" "$DOC_MODEL" tiny10 16
    log "  static $1 over r2v12-bias005: tiny10 $(lbv "asym-r2v12b-q-static-$1" tiny10)"
}

log "=== s7 step 1: phrase tables ==="
build phr1 1
build phr2 2

log "=== s7 step 2: training on the s6 pool ==="
strain phr1-k8-kd phr1-k8 "$POOL" 2
strain phr2-k8-kd phr2-k8 "$POOL" 2

log "=== s7 step 3: tiny10 of the phrase tables ==="
asym phr1-k8-kd
asym phr2-k8-kd

log "=== s7 step 4: round-2 pool (students = the trained tables) and longer training ==="
if [ ! -f "$POOL2/pool.json" ]; then
    need_space || exit 1; wait_gpu_free; wait_ram; s=$SECONDS; log "mine $POOL2"
    timeout -k 60 7200 "$PY" -m distill.static_mine --doc-model "$DOC_MODEL" --corpus "$CORPUS" --out "$POOL2" \
        --student "$SM/credit-k8-kd" --student "$SM/phr2-k8-kd" --device cuda --dtype fp16 --n-docs 250000 --n-queries 120000 \
        --teacher-top 32 --student-top 16 --max-candidates 64 --batch-size 16 --search-batch 128 --seed 7 > "$LOGS/static_mine_kd-r2v12b-r2.log" 2>&1
    [ -f "$POOL2/pool.json" ] && { log "mine r2 done in $((SECONDS - s))s"; record "static_mine:kd-r2v12b-r2" ok "-" "$((SECONDS - s))s"; } \
        || { log "mine r2 FAILED"; record "static_mine:kd-r2v12b-r2" FAILED "-" "$((SECONDS - s))s"; }
fi
if [ -f "$POOL2/pool.json" ]; then
    strain credit-k8-kd-r2 credit-k8-kd "$POOL2" 3
    strain phr2-k8-kd-r2 phr2-k8-kd "$POOL2" 3
    asym credit-k8-kd-r2
    asym phr2-k8-kd-r2
fi

log "=== s7 step 5: summary ==="
SUM="$RESULTS_ROOT/summary_s7.md"
"$PY" - "$RS" "$REGIMES_ROOT/quantized" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
rs, reg = Path(sys.argv[1]), Path(sys.argv[2])
def nd(name):
    f = reg / f"asym-r2v12b-q-static-{name}" / "tiny10.json"
    return f"{json.load(open(f))['mean_ndcg@10']:.2f}" if f.exists() else "-"
print("# s7 (8.11): phrase rows and round-2 mining for the static query table over r2v12-bias005 (quantized regime, tiny10 nDCG@10)\n")
print("| table | what | tiny10 | val top1 | dims/q | postings fraction |\n|---|---|---:|---:|---:|---:|")
print("| r2v12-bias005 symmetric | full SPLADE query side | 64.21 | 1.000 | | |")
desc = {"credit-k8-kd": "s6: token rows, k8, s6 pool", "phr1-k8-kd": "+ rows for multi-piece words", "phr2-k8-kd": "+ words and word pairs",
        "credit-k8-kd-r2": "s6 table + round-2 pool, 3 epochs", "phr2-k8-kd-r2": "phr2 + round-2 pool, 3 epochs"}
for name, what in desc.items():
    f = rs / f"{name}.json"
    v = json.load(open(f)).get("val_final", {}) if f.exists() else {}
    print(f"| {name} | {what} | {nd(name)} | {v.get('top1_agreement', 0):.3f} | {v.get('dims_per_query', 0):.1f} | {v.get('postings_fraction', 0):.3f} |")
print("\nval columns: 2 000 held-out kd_v1 queries of the pool each table trained on (the r2 rows on the round-2 pool, not comparable 1:1).")
EOF
log "summary written to $SUM"
git add "$SUM" "$RS" "$REGIMES_ROOT"/quantized/asym-r2v12b-q-static-* "$EXP_ROOT/queues/queue_s7.sh" 2>/dev/null
git commit -q -m "Add the s7 results: phrase rows and round-2 mining for the static query table" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_s7.log" 2>&1 && { echo pushed >> "$LOGS/push_s7.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== s7 done ($(date '+%d.%m %H:%M')) ==="
