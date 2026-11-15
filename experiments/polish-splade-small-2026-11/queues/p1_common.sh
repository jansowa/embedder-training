# Shared by the parts of the pause queue p1 (15.11, in the pause of queue b1 after its KD): the static query table of the
# release in its released format (static-sparse-query/1) over the index of the frozen full model, and short GPU speed
# measurements. Sourced by queue_p1a.sh (W1-W3), queue_p1b.sh (A1-A4) and queue_p1c.sh (GPU benchmarks), which the driver
# logs/p1_chain.sh runs one after another while it holds logs/b1.pause. State shared by the parts: $ST (key<TAB>value).
# Do not edit while a part is running.
set -uo pipefail
export QUEUE_TAG=${QUEUE_TAG:-p1} MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
MAIN=experiments/polish-splade-2026-10
source "$MAIN/queues/arm_common.sh"
guard_alive

FULL=soup-v20-dqf-bias015; DOC="models/$FULL"; T=phr2-k8-kd-r2
SM="models/static-$FULL"; POOL2="cache/static_pools/kd-$FULL-r2"
RS="$RESULTS_ROOT/static-$FULL"; P1="$RS/p1"; ST="$P1/state.tsv"
MAX_POSTINGS=16807
REL_T10=60.74; REL_H8=39.24      # the released table (st1); its cost on the cost sample: 89.7 dims/query, 11 764 postings/query
TRAIN=(--target rows --learn-scale --batch-size 64 --warmup-steps 200 --schedule linear --val-rows 2000 --eval-every 500 --log-every 100 --device cuda --seed 42 --keep-best --best-metric top1_agreement)
mkdir -p "$P1"; touch "$ST"

sget() { awk -F'\t' -v k="$1" '$1==k {v=$2} END {print v}' "$ST" 2>/dev/null; }
sset() { printf '%s\t%s\n' "$1" "$2" >> "$ST"; }
lab()  { echo "asym-$FULL-q-static-$1"; }                       # label of a table folder $SM/NAME
olab() { echo "asym-$FULL-q-oracle-$1"; }                       # label of an oracle folder $SM/oracle-MODE
t10()  { lbv "$(lab "$1")" tiny10; }
ev() {  # ev NAME [TASKSET]: asymmetric evaluation, queries from $SM/NAME, the index of the frozen model
    [ -e "$SM/$1" ] || { log "  ev $1: $SM/$1 missing"; return 1; }
    PIRB_QUERY_MODEL="$SM/$1" regime quantized "$(lab "$1")" "$DOC" "${2:-tiny10}" 16
}
cost() {  # cost TAG NAME...: dims and postings per query of the tables on the cost sample (per task, arguana apart) -> $P1/cost_TAG.json, state pq:/dq:
    local tag="$1" out="$P1/cost_$1.json" q=() n s; shift
    for n in "$@"; do [ -e "$SM/$n" ] && [ -z "$(sget "pq:$n")" ] && q+=(--query "$n=$SM/$n"); done
    [ "${#q[@]}" -gt 0 ] || return 0
    wait_gpu_free; wait_ram; s=$SECONDS
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 2400 "$PY" scripts/splade_index_cost_asym.py --doc-model "full=$DOC" "${q[@]}" --output-json "$out" > "$LOGS/p1_cost_$tag.log" 2>&1 \
        || { log "  cost $tag FAILED (logs/p1_cost_$tag.log)"; record "p1:cost:$tag" FAILED "-" "$((SECONDS - s))s"; return 1; }
    "$PY" - "$out" <<'PYEOF' | while IFS=$'\t' read -r k v l; do sset "$k" "$v"; [ -n "$l" ] && log "  $l"; done
import json, sys
d = json.load(open(sys.argv[1]))
for name, r in d["pairs"].items():
    a = r.get("per_task", {}).get("arguana-pl", {}); w = r.get("without_arguana", {})
    print("pq:%s\t%.0f\t" % (name, r["postings_per_query"]))
    print("pqa:%s\t%.0f\t" % (name, a.get("postings_per_query", 0)))
    print("pqr:%s\t%.0f\t" % (name, w.get("postings_per_query", 0)))
    print("dqa:%s\t%.1f\t" % (name, a.get("query_active_mean", 0)))
    print("dq:%s\t%.1f\tcost %s: %.1f dims/query, %.0f postings/query (arguana-pl %.1f / %.0f, the other tasks %.1f / %.0f); full model symmetric %.1f / %.0f" % (
        name, r["query_active_mean"], name, r["query_active_mean"], r["postings_per_query"], a.get("query_active_mean", 0), a.get("postings_per_query", 0),
        w.get("query_active_mean", 0), w.get("postings_per_query", 0), d["symmetric"]["query_active_mean"], d["symmetric"]["postings_per_query"]))
PYEOF
    record "p1:cost:$tag" ok "-" "$((SECONDS - s))s"
}
within() { [ -n "$(sget "pq:$1")" ] && [ "$(gte "$MAX_POSTINGS" "$(sget "pq:$1")")" = 1 ]; }      # within NAME: at most 16 807 postings/query
prune() {  # prune SRC OUT [prune args]: a CPU step of its own (distill.static_query prune)
    [ -f "$SM/$2/static_query.json" ] && return 0
    local src="$1" out="$2"; shift 2
    [ -f "$SM/$src/static_query.json" ] || { log "  prune $out: $SM/$src missing"; return 1; }
    wait_ram
    timeout -k 60 900 "$PY" -m distill.static_query prune --model "$SM/$src" --out "$SM/$out" "$@" > "$LOGS/p1_prune_$out.log" 2>&1
    [ -f "$SM/$out/static_query.json" ] && log "  pruned $out from $src ($*): $(tail -n 1 "$LOGS/p1_prune_$out.log" | cut -c1-200)" || { log "  prune $out FAILED (logs/p1_prune_$out.log)"; return 1; }
}
strain() {  # strain NAME INIT_NAME [train_static args]: the baseline recipe of the released table (lr 5e-3, 3 epochs on pool r2) + the given options
    local name="$1" init="$2"; shift 2
    [ -f "$SM/$name/static_query.json" ] && return 0
    [ -f "$SM/$init/static_query.json" ] || { log "  train $name: init $init missing"; return 1; }
    wait_gpu_free; wait_ram; local s=$SECONDS; log "train $name from $init on $(basename "$POOL2") ($*)"
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 2400 "$PY" -m distill.train_static --pool "$POOL2" --init "$SM/$init" --out "runs/static-$FULL-$name" --lr 5e-3 --epochs 3 \
        "${TRAIN[@]}" "$@" > "$LOGS/p1_train_$name.log" 2>&1
    [ -f "runs/static-$FULL-$name/final/static_query.json" ] || { log "  train $name FAILED after $((SECONDS - s))s (logs/p1_train_$name.log): $(tail -n 1 "$LOGS/p1_train_$name.log" | cut -c1-200)"; record "p1:train:$name" FAILED "-" "$((SECONDS - s))s"; return 1; }
    cp -r "runs/static-$FULL-$name/final" "$SM/$name"; cp "runs/static-$FULL-$name/static_summary.json" "$RS/$name.json"
    log "  train $name done in $((SECONDS - s))s: $("$PY" -c "
import json,sys
d=json.load(open(sys.argv[1]));i,f,t=d['val_init'],d['val_final'],d.get('train_final') or {}
print('val top1 %.4f -> %.4f (train rows %s), dims/query %.1f -> %.1f, saved step %s of %s' % (i['top1_agreement'], f['top1_agreement'], ('%.4f' % t['top1_agreement']) if t else '-', i['dims_per_query'], f['dims_per_query'], d.get('saved_step'), d.get('steps')))" "$RS/$name.json" 2>&1 | tail -n 1)"
    record "p1:train:$name" ok "$(sval "$name" top1_agreement)" "$((SECONDS - s))s"
}
sval() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]))[sys.argv[3] if len(sys.argv)>3 else 'val_final'];v=d.get(sys.argv[2]);print('%.4f' % v if v is not None else '')" "$RS/$1.json" "$2" ${3:-} 2>/dev/null; }
cand() {  # cand NAME NOTE: remember a measured candidate; the best one (highest tiny10 within the postings limit) goes to state best
    local n="$1" v b; v=$(t10 "$n")
    [ -n "$v" ] || return 0
    grep -q -P "^cand\t$n\$" "$ST" || sset cand "$n"
    sset "note:$n" "${2:-}"
    within "$n" || { log "  $n: tiny10 $v but ${3:-$(sget "pq:$n")} postings/query: over the limit of $MAX_POSTINGS, not a candidate"; return 0; }
    b=$(sget best)
    if [ -z "$b" ] || [ "$(gte "$v" "$(t10 "$b")" 0.005)" = 1 ]; then sset best "$n"; log "  best table so far: $n (tiny10 $v, $(sget "pq:$n") postings/query; released $REL_T10 at $(sget "pq:$T"))"; fi
}
table() {  # table: the candidates so far as a markdown table -> $P1/candidates.md
    {
        echo "| table | tiny10 | vs released | heldout8 | dims/query | postings/query | arguana-pl: dims / postings | other tasks: postings | within 16 807 | note |"
        echo "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"
        local n v
        for n in $(awk -F'\t' '$1=="cand" {print $2}' "$ST" | awk '!seen[$0]++'); do
            v=$(t10 "$n")
            echo "| \`$n\` | $v | $([ -n "$v" ] && fsub "$v" "$REL_T10") | $(lbv "$(lab "$n")" heldout8) | $(sget "dq:$n") | $(sget "pq:$n") | $(sget "dqa:$n") / $(sget "pqa:$n") | $(sget "pqr:$n") | $(within "$n" && echo yes || echo NO) | $(sget "note:$n") |"
        done
        echo; echo "Best within the limit: \`$(sget best)\`. Released table: \`$T\` ($REL_T10 / $REL_H8). Full model symmetric: 65.06 / 43.64, 123.0 dims/query, 13 945 postings/query."
    } > "$P1/candidates.md"
}
elapsed_min() { local t0; t0=$(sget t0); echo $(( ($(date +%s) - ${t0:-$(date +%s)}) / 60 )); }
