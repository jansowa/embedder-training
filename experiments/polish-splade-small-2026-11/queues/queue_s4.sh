#!/usr/bin/env bash
# Queue s4 (3060 Ti, draft 7.11, ~3 h): static query encoder over the index of the best SPLADE (arm-dense-kl-n5kj-ens400,
# tiny10 64.14). The document side stays frozen; the query side is a lookup table (distill.static_query), no transformer.
#   0  unit tests of the static modules (distill + PIRB fork); a failure stops the queue
#   1  tables: S0 = BM25 idf weights on 300 k kd_v1 passages (k = 1); S2-init = expansion table credited from the
#      document model on the kd_v1 queries (k = 32, rare tokens from <s> t </s>)
#   2  candidate pool: the n5 + kj rows (ensemble scores, -tens400) encoded once by the document model, + its own queries
#   3  training (minutes, no transformer): S1 = idf weights on the row scores; S2 = credit-k32 on the row scores;
#      S2m = credit-k32 on 0.5 x symmetric SPLADE in-batch + 0.5 x row scores
#   4  tiny10 in asymmetric mode for S0, S1, S2-init, S2, S2m; heldout8 for the best static encoder
#   5  summary_s4.md (nDCG, dims/query, posting fraction on the pool, validation agreement), commit, push
# Reading (tiny10; symmetric n5kj 64.14, ALBERT hl-full-s3 as query encoder over v9b 63.35):
#   best static >= 62.6 (within 1.5 of symmetric) = the static query side is the deployment candidate; next = k sweep
#   (16 / 64) and a joint fine-tune of the document side only if the gap is > 1;
#   S2 - S1 < 0.5 = expansions do not pay at this document side, keep k = 1 (the SparseStaticEmbedding model);
#   best static < 60 = the frozen symmetric index needs a document side trained for static queries (joint training).
#   WAIT_FOR="v42 v43" setsid nohup bash experiments/polish-splade-small-2026-11/queues/queue_s4.sh > logs/queue_s4.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=s4 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_s4.pid"
guard_alive
DOC_MODEL=${DOC_MODEL:-models/arm-dense-kl-n5kj-ens400}
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
ROWS=${ROWS:-datasets/arms/n5-plus-kj-tens400.jsonl}
POOL=${POOL:-cache/static_pools/n5kj}
SM="models/static"

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v42 v43}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
wait_gpu_free
for f in "$CORPUS" "$ROWS"; do [ -f "$f" ] || { log "$f missing"; exit 1; }; done
[ -d "$DOC_MODEL" ] || { log "document model $DOC_MODEL missing"; exit 1; }
mkdir -p "$SM" "$RESULTS_ROOT/static"
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
log "=== s4 start: static query encoders over $DOC_MODEL; $(free_gb) GB free on / ==="

log "=== s4 step 0: unit tests ==="
wait_ram
if ! timeout -k 30 900 "$PY" -m pytest -q tests/test_distill_static.py > "$LOGS/s4_pytest.log" 2>&1 \
    || ! (cd third_party/pirb && timeout -k 30 600 "$PY" -m pytest -q tests/test_splade_static_query.py tests/test_splade_query_model.py) >> "$LOGS/s4_pytest.log" 2>&1; then
    log "unit tests FAILED (logs/s4_pytest.log); stopping"; record "pytest:s4" FAILED "-"; exit 1
fi
log "unit tests passed: $(grep -E 'passed' "$LOGS/s4_pytest.log" | tr '\n' ' ')"
# the code was written on 7.11 without a GPU window to test it; it is committed here, once the tests pass:
# first inside the PIRB submodule (its own repo), then the submodule pointer and the distill files in this repo
if [ -n "$(git -C third_party/pirb status --short search/splade.py tests/test_splade_static_query.py)" ]; then
    git -C third_party/pirb add search/splade.py tests/test_splade_static_query.py \
        && git -C third_party/pirb commit -q -m "Add a static lookup-table query encoder for the SPLADE index" \
        && log "submodule committed $(git -C third_party/pirb log --oneline -1)"
fi
git add third_party/pirb distill/static_query.py distill/static_pool.py distill/train_static.py distill/__init__.py tests/test_distill_static.py docs/distillation.md 2>/dev/null
git commit -q -m "Add the static query encoder: lookup table over the SPLADE vocabulary, pool builder and trainer" 2>/dev/null && log "code committed $(git log --oneline -1)"

build() {  # build NAME [static_query build args]: a table in models/static/NAME
    local name="$1"; shift
    [ -f "$SM/$name/static_query.json" ] && return 0
    wait_gpu_free; wait_ram; log "build $name ($*)"; local s=$SECONDS
    timeout -k 60 5400 "$PY" -m distill.static_query build --out "$SM/$name" "$@" > "$LOGS/static_build_$name.log" 2>&1
    if [ -f "$SM/$name/static_query.json" ]; then
        log "build $name done in $((SECONDS - s))s: $("$PY" -c "import json,sys;d=json.load(open('$SM/$name/static_query.json'));print('k', d['k'], 'active slots', d['active_slots'])")"
        record "static_build:$name" ok "-" "$((SECONDS - s))s"
    else log "build $name FAILED (logs/static_build_$name.log)"; record "static_build:$name" FAILED "-" "$((SECONDS - s))s"; return 1; fi
}
strain() {  # strain NAME INIT [train_static args]: runs/static-NAME, final copied to models/static/NAME
    local name="$1" init="$2"; shift 2
    [ -f "$SM/$name/static_query.json" ] && return 0
    [ -f "$SM/$init/static_query.json" ] || { log "train $name: init $init missing"; return 1; }
    wait_gpu_free; wait_ram; log "train $name from $init ($*)"; local s=$SECONDS
    timeout -k 60 3600 "$PY" -m distill.train_static --pool "$POOL" --init "$SM/$init" --out "runs/static-$name" "$@" > "$LOGS/static_train_$name.log" 2>&1
    if [ -f "runs/static-$name/final/static_query.json" ]; then
        cp -r "runs/static-$name/final" "$SM/$name"; cp "runs/static-$name/static_summary.json" "$RESULTS_ROOT/static/$name.json"
        log "train $name done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$RESULTS_ROOT/static/$name.json'));i,f=d.get('val_init',{}),d.get('val_final',{});print('val mrr %.4f -> %.4f, tau %.3f -> %.3f, dims/q %.1f' % (i.get('mrr',0), f.get('mrr',0), i.get('kendall_tau',0), f.get('kendall_tau',0), f.get('dims_per_query',0)))")"
        record "static_train:$name" ok "-" "$((SECONDS - s))s"
    else log "train $name FAILED (logs/static_train_$name.log)"; record "static_train:$name" FAILED "-" "$((SECONDS - s))s"; return 1; fi
}
asym() {  # asym NAME TASKSET: the static table NAME as the query encoder over DOC_MODEL's index
    [ -f "$SM/$1/static_query.json" ] || return 1
    PIRB_QUERY_MODEL="$SM/$1" regime quantized "asym-n5kj-q-static-$1" "$DOC_MODEL" "$2" 16
    log "  static $1 over n5kj: $2 $(lbv "asym-n5kj-q-static-$1" "$2")"
}

log "=== s4 step 1: tables ==="
build idf --init idf --teacher "$DOC_MODEL" --texts "$CORPUS" --kinds passage --n 300000 --device cpu --dtype fp32
build credit-k32 --init credit --teacher "$DOC_MODEL" --texts "$CORPUS" --kinds query --top-k 32 --min-count 3 --batch-size 64 --device cuda --dtype fp16

log "=== s4 step 2: candidate pool ==="
if [ ! -f "$POOL/pool.json" ]; then
    need_space || exit 1; wait_gpu_free; wait_ram; s=$SECONDS
    timeout -k 60 7200 "$PY" -m distill.static_pool --doc-model "$DOC_MODEL" --rows "$ROWS" --out "$POOL" --teacher-queries \
        --max-seq-length 512 --batch-size 16 --device cuda --dtype fp16 > "$LOGS/static_pool_n5kj.log" 2>&1
    [ -f "$POOL/pool.json" ] && { log "pool done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$POOL/pool.json'));print(d['rows'], 'rows', d['docs'], 'docs', d['doc_dims_mean'], 'dims/doc', d['query_dims_mean'], 'dims/query (symmetric)')")"; record "static_pool:n5kj" ok "-" "$((SECONDS - s))s"; } \
        || { log "pool FAILED (logs/static_pool_n5kj.log); stopping"; record "static_pool:n5kj" FAILED "-"; exit 1; }
fi

log "=== s4 step 3: training ==="
T_COMMON=(--epochs 3 --batch-size 64 --warmup-steps 100 --schedule linear --val-rows 1000 --eval-every 250 --log-every 50 --device cuda --seed 42)
strain idf-rows idf --target rows --lr 5e-3 "${T_COMMON[@]}"
strain credit-k32-rows credit-k32 --target rows --lr 5e-3 "${T_COMMON[@]}"
strain credit-k32-mix credit-k32 --target mix --mix-weight 0.5 --in-batch --lr 5e-3 "${T_COMMON[@]}"

log "=== s4 step 4: tiny10 in asymmetric mode ==="
for name in idf idf-rows credit-k32 credit-k32-rows credit-k32-mix; do asym "$name" tiny10; done
best=$(for name in idf idf-rows credit-k32 credit-k32-rows credit-k32-mix; do v=$(lbv "asym-n5kj-q-static-$name" tiny10); [ -n "$v" ] && echo "$v $name"; done | sort -rn | head -1 | awk '{print $2}')
if [ -n "$best" ]; then
    log "best static encoder on tiny10: $best"
    asym "$best" heldout8
fi

log "=== s4 step 5: summary ==="
SUM="$RESULTS_ROOT/summary_s4.md"
{
    echo "# s4 ($(date '+%d.%m.%Y %H:%M'), quantized regime, nDCG@10): static query encoders over the index of $(basename "$DOC_MODEL")"
    echo
    echo "Document side frozen ($DOC_MODEL, symmetric tiny10 64.14). Query side = a lookup table (distill.static_query): idf = BM25 idf weights of 300 k kd_v1 passages (k = 1, only the query's own tokens); credit-k32 = expansion table credited from the document model on the kd_v1 queries (32 slots per token). -rows = trained on the ensemble scores of $(basename "$ROWS") (KL over each row's candidates); -mix = 0.5 symmetric SPLADE with in-batch passages + 0.5 row scores. Validation on 1 000 held-out pool rows with quantized query weights; postings fraction = sum of df / N over the query's dims on the pool's $( "$PY" -c "import json;print(json.load(open('$POOL/pool.json'))['docs'])" 2>/dev/null ) passages (x corpus size = postings per query)."
    echo
    echo '```'
    "$PY" - "$RESULTS_ROOT/static" "$REGIMES_ROOT/quantized" <<'EOF' 2>> "$LOGS/queue_summary.log"
import json, sys
from pathlib import Path
static, regimes = Path(sys.argv[1]), Path(sys.argv[2])
def nd(label, ts):
    f = regimes / label / f"{ts}.json"
    return json.load(open(f))["mean_ndcg@10"] if f.exists() else None
fmt = lambda v, p=2: f"{v:.{p}f}" if isinstance(v, (int, float)) else "-"
print(f"{'query side':22s} {'tiny10':>7s} {'heldout8':>9s} {'val mrr':>8s} {'tau':>6s} {'dims/q':>7s} {'post.frac':>10s}")
ref = None
for name in ["idf", "idf-rows", "credit-k32", "credit-k32-rows", "credit-k32-mix"]:
    label = f"asym-n5kj-q-static-{name}"
    s = json.load(open(static / f"{name}.json")) if (static / f"{name}.json").exists() else {}
    v = s.get("val_final", {})
    ref = ref or s.get("val_splade_reference")
    print(f"{name:22s} {fmt(nd(label, 'tiny10')):>7s} {fmt(nd(label, 'heldout8')):>9s} {fmt(v.get('mrr'), 4):>8s} {fmt(v.get('kendall_tau'), 3):>6s} {fmt(v.get('dims_per_query'), 1):>7s} {fmt(v.get('postings_fraction'), 3):>10s}")
if ref:
    print(f"{'symmetric n5kj':22s} {'64.14':>7s} {'-':>9s} {fmt(ref.get('mrr'), 4):>8s} {fmt(ref.get('kendall_tau'), 3):>6s} {fmt(ref.get('dims_per_query'), 1):>7s} {fmt(ref.get('postings_fraction'), 3):>10s}")
EOF
    echo '```'
    echo; echo "Init tables (idf, credit-k32) carry no validation of their own; their val numbers are the val_init of the -rows runs."
    echo; echo "## Queue state (s4)"; echo; echo '```'
    grep -E "^(static_|regime:quantized:asym-n5kj-q-static)" "$STATE"
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_s4.out"
log "summary written to $SUM"
# no paths inside the third_party/pirb submodule here: one such path aborts the whole git add
git add "$SUM" "$RESULTS_ROOT"/static "$REGIMES_ROOT"/quantized "$EXP_ROOT/queues/queue_s4.sh" 2>/dev/null
git commit -q -m "Add the s4 results: static query encoders over the n5kj index" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_s4.log" 2>&1 && { echo pushed >> "$LOGS/push_s4.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== s4 done ($(date '+%d.%m %H:%M')) ==="
