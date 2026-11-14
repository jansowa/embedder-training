#!/usr/bin/env bash
# Queue st1 (3060 Ti, after r1; ~1.5-2 h): the static query table trained again over the index of the FROZEN full model.
# The table has to work over the index of the model that is released, so the best recipe so far, phr2-k8-kd-r2 (s7 / s8:
# 60.20 / 38.60 over r2v12-bias005), is repeated step by step with the frozen model as the document side and teacher:
#   1  tables credited from the frozen model on the kd_v1 queries: credit-k32 -> credit-k8 (token rows) and
#      phr2-k32 -> phr2-k8 (+ rows for whole words and adjacent word pairs, min 5 queries, at most 200 k); the idf table
#      is the tokenizer's (copied)
#   2  pool 1 mined with the new index (250 k passages, 120 k queries of kd_v1; students idf + credit-k32), credit-k8-kd
#      and phr2-k8-kd trained on it (KL against the teacher's scores, learned scale, lr 5e-3, 2 epochs)
#   3  pool 2 (students = the two trained tables, seed 7), phr2-k8-kd-r2 = phr2-k8-kd trained on it for 3 epochs
#   4  tiny10 and heldout8 asymmetrically (queries from the table, the index of the frozen model), dims and postings per
#      query; the Sentence Transformers package (release/st_export.py: router with the table as the query route and the
#      frozen model as the document route, and the query side alone); the package for the 5070
#   5  summary_st1.md, commit, push loop
# The joint training of the document encoder with the table is NOT repeated (v54: the document model lost 0.4).
#   setsid nohup bash experiments/polish-splade-small-2026-11/queues/queue_st1.sh > logs/queue_st1.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=st1 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
MAIN=experiments/polish-splade-2026-10
source "$MAIN/queues/arm_common.sh"
source "$MAIN/queues/release_common.sh"
echo $$ > "$LOGS/queue_st1.pid"
guard_alive
wait_for_queues ${WAIT_FOR:-v61 v61b v62 r1}

mlbv() { regime_mean "$MAIN/results-regimes/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
D3B=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015
FULL=$(final_full); [ -n "$FULL" ] && [ -d "models/$FULL" ] || { log "  no frozen full model in $FINAL_FULL_FILE (${FULL:-empty}): dose3-bias015 is used"; FULL="$D3B"; }
FTAG=$(echo "$FULL" | sed -e 's/^arm-dense-kl-n5kj-mined-//' -e 's/-ens400-r2v12//')
DOC_MODEL="models/$FULL"
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
POOL=cache/static_pools/kd-$FTAG; POOL2=cache/static_pools/kd-$FTAG-r2
SM="models/static-$FTAG"; RS="$RESULTS_ROOT/static-$FTAG"
T=phr2-k8-kd-r2
ASYM="asym-$FTAG-q-static-$T"
ST="models/st/$FTAG-static-$T"
mkdir -p "$SM" "$RS" models/st
log "=== st1 start: static query table $T over the frozen full model $FULL ($(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8)); earlier over r2v12-bias005 60.20 / 38.60; $(free_gb) GB free on / ==="
[ -s "$CORPUS" ] || { log "  $CORPUS missing: st1 stops"; exit 1; }
need_space || { log "  not enough space on /: st1 stops"; exit 1; }

build() {  # build NAME PHRASES: credited table NAME-k32 (PHRASES 0 = token rows only) and its cut NAME-k8
    local name="$1" ph="$2" extra=()
    [ "$ph" != 0 ] && extra=(--phrases "$ph" --phrase-min-count 5 --max-phrases 200000)
    [ -f "$SM/$name-k32/static_query.json" ] || {
        wait_gpu_free; wait_ram; local s=$SECONDS; log "build $name-k32 (phrases $ph)"
        ORPHAN_GUARD_EXEMPT=1 timeout -k 60 5400 "$PY" -m distill.static_query build --out "$SM/$name-k32" --init credit --teacher "$DOC_MODEL" --texts "$CORPUS" \
            --kinds query --top-k 32 --min-count 3 --batch-size 64 --device cuda --dtype fp16 "${extra[@]}" > "$LOGS/static_build_${FTAG}_$name.log" 2>&1
        [ -f "$SM/$name-k32/static_query.json" ] || { log "build $name FAILED (logs/static_build_${FTAG}_$name.log)"; record "static_build:$FTAG:$name" FAILED "-" "$((SECONDS - s))s"; return 1; }
        log "build $name-k32 done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$SM/$name-k32/static_query.json'));print('rows', d.get('rows'), 'phrases', d.get('phrases'))")"
        record "static_build:$FTAG:$name" ok "-" "$((SECONDS - s))s"
    }
    [ -f "$SM/$name-k8/static_query.json" ] || "$PY" -m distill.static_query prune --model "$SM/$name-k32" --out "$SM/$name-k8" --top-k 8 > "$LOGS/static_prune_${FTAG}_$name.log" 2>&1
    [ -f "$SM/$name-k8/static_query.json" ]
}
mine() {  # mine POOL_DIR [static_mine args]: a candidate pool of 250 k passages and 120 k queries with the frozen model as the teacher
    local pool="$1"; shift
    [ -f "$pool/pool.json" ] && return 0
    need_space || return 1; wait_gpu_free; wait_ram; local s=$SECONDS; log "mine $pool"
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 7200 "$PY" -m distill.static_mine --doc-model "$DOC_MODEL" --corpus "$CORPUS" --out "$pool" "$@" --device cuda --dtype fp16 \
        --n-docs 250000 --n-queries 120000 --teacher-top 32 --student-top 16 --max-candidates 64 --batch-size 16 --search-batch 128 > "$LOGS/static_mine_$(basename "$pool").log" 2>&1
    [ -f "$pool/pool.json" ] && { log "mine $(basename "$pool") done in $((SECONDS - s))s"; record "static_mine:$(basename "$pool")" ok "-" "$((SECONDS - s))s"; } \
        || { log "mine $(basename "$pool") FAILED (logs/static_mine_$(basename "$pool").log)"; record "static_mine:$(basename "$pool")" FAILED "-" "$((SECONDS - s))s"; return 1; }
}
strain() {  # strain NAME INIT POOL EPOCHS
    local name="$1" init="$2" pool="$3" ep="$4"
    [ -f "$SM/$name/static_query.json" ] && return 0
    [ -f "$SM/$init/static_query.json" ] || { log "train $name: init $init missing"; return 1; }
    wait_gpu_free; wait_ram; local s=$SECONDS; log "train $name from $init on $(basename "$pool"), $ep epochs"
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" -m distill.train_static --pool "$pool" --init "$SM/$init" --out "runs/static-$FTAG-$name" --lr 5e-3 \
        --target rows --learn-scale --epochs "$ep" --batch-size 64 --warmup-steps 200 --schedule linear --val-rows 2000 --eval-every 500 \
        --log-every 100 --device cuda --seed 42 > "$LOGS/static_train_${FTAG}_$name.log" 2>&1
    [ -f "runs/static-$FTAG-$name/final/static_query.json" ] || { log "train $name FAILED (logs/static_train_${FTAG}_$name.log)"; record "static_train:$FTAG:$name" FAILED "-" "$((SECONDS - s))s"; return 1; }
    cp -r "runs/static-$FTAG-$name/final" "$SM/$name"; cp "runs/static-$FTAG-$name/static_summary.json" "$RS/$name.json"
    log "train $name done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$RS/$name.json'));i,f=d['val_init'],d['val_final'];print('val top1 %.3f -> %.3f, dims/q %.1f -> %.1f' % (i['top1_agreement'], f['top1_agreement'], i['dims_per_query'], f['dims_per_query']))")"
    record "static_train:$FTAG:$name" ok "-" "$((SECONDS - s))s"
}
sval() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]))['val_final'];v=d.get(sys.argv[2]);print('%.3f' % v if v is not None else '')" "$RS/$1.json" "$2" 2>/dev/null; }

# 1. tables
log "=== st1 step 1: tables credited from $FULL ==="
if [ ! -f "$SM/idf/static_query.json" ]; then
    if [ -f models/static-r2v12b/idf/static_query.json ]; then cp -r models/static-r2v12b/idf "$SM/idf" && log "  idf table copied from models/static-r2v12b (the tokenizer's, shared)"
    else
        wait_gpu_free; wait_ram
        ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" -m distill.static_query build --out "$SM/idf" --init idf --teacher "$DOC_MODEL" --texts "$CORPUS" --kinds passage --n 300000 --device cpu --dtype fp32 > "$LOGS/static_build_${FTAG}_idf.log" 2>&1
    fi
fi
[ -f "$SM/idf/static_query.json" ] || { log "  no idf table: st1 stops"; exit 1; }
build credit 0 || { log "  st1 stops"; exit 1; }
build phr2 2 || { log "  st1 stops"; exit 1; }

# 2. pool 1 and the first training
log "=== st1 step 2: pool 1 with the new index, first training ==="
mine "$POOL" --student "$SM/idf" --student "$SM/credit-k32" || { log "  st1 stops"; exit 1; }
strain credit-k8-kd credit-k8 "$POOL" 2
strain phr2-k8-kd phr2-k8 "$POOL" 2 || { log "  st1 stops"; exit 1; }

# 3. pool 2 and the final table
log "=== st1 step 3: pool 2 (students = the trained tables), $T ==="
STUD=(--student "$SM/phr2-k8-kd"); [ -f "$SM/credit-k8-kd/static_query.json" ] && STUD=(--student "$SM/credit-k8-kd" --student "$SM/phr2-k8-kd")
mine "$POOL2" "${STUD[@]}" --seed 7 || { log "  st1 stops"; exit 1; }
strain "$T" phr2-k8-kd "$POOL2" 3 || { log "  st1 stops"; exit 1; }

# 4. measure, package
log "=== st1 step 4: asymmetric tiny10 and heldout8 over the index of $FULL ==="
PIRB_QUERY_MODEL="$SM/$T" regime quantized "$ASYM" "$DOC_MODEL" tiny10 16
PIRB_QUERY_MODEL="$SM/$T" regime quantized "$ASYM" "$DOC_MODEL" heldout8 16
log "  static $T over $FULL: $(lbs "$ASYM") (over r2v12-bias005: 60.20 / 38.60; full model symmetric $(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8)); val top1 $(sval $T top1_agreement), dims/query $(sval $T dims_per_query), postings fraction $(sval $T postings_fraction)"
if [ ! -f "$ST/modules.json" ]; then
    wait_gpu_free; wait_ram; rm -rf "$ST" "$ST-query"
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 1800 "$PY" -m release.st_export --doc-model "$DOC_MODEL" --table "$SM/$T" --out "$ST" --query-only-out "$ST-query" > "$LOGS/st1_st_export.log" 2>&1 \
        && log "  Sentence Transformers package written: $ST (and $ST-query): $(tail -n 1 "$LOGS/st1_st_export.log" | cut -c1-300)" || { log "  st_export FAILED (logs/st1_st_export.log)"; rm -rf "$ST" "$ST-query"; }
fi
if [ -f "$ST/modules.json" ]; then
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 900 "$PY" - "$ST" "$SM/$T" > "$LOGS/st1_st_check.log" 2>&1 <<'PYEOF' && log "  package check: $(tail -n 1 "$LOGS/st1_st_check.log" | cut -c1-300)" || log "  package check FAILED (logs/st1_st_check.log)"
import sys
import torch
from sentence_transformers import SparseEncoder
m = SparseEncoder(sys.argv[1], trust_remote_code=True, device="cpu")
qs = ["jak działa splade?", "ile kosztuje wymiana dowodu osobistego", "objawy grypy u dzieci"]
q = m.encode_query(qs, convert_to_tensor=True); d = m.encode_document(["SPLADE to rzadki model wyszukiwania, który rozszerza tekst o powiązane terminy."], convert_to_tensor=True)
q = q.to_dense() if q.layout != torch.strided else q; d = d.to_dense() if d.layout != torch.strided else d
nq = (q > 0).sum(dim=1).tolist(); nd = (d > 0).sum(dim=1).tolist()
assert min(nq) > 0 and min(nd) > 0, (nq, nd)
print("loaded with trust_remote_code; active dims per query %s, per document %s, score of the first query %.2f" % (nq, nd, float((q[0] * d[0]).sum())))
PYEOF
fi
{ echo "$SM/$T"; echo "# first line = the static query table to release (over the index of models/$FULL); written by queue st1 on $(date '+%d.%m %H:%M')"; echo "$ST"; } > "$FINAL_STATIC_FILE"
pack_add "static/$FTAG-static-$T" "$SM/$T" "static query table $T over the index of $FULL (static-sparse-query/1); asymmetric $(lbs "$ASYM"), dims/query $(sval $T dims_per_query)"
[ -f "$ST/modules.json" ] && pack_add "static/st-$FTAG-static-$T" "$ST" "Sentence Transformers SparseEncoder: query route = the table, document route = $FULL (load with trust_remote_code=True)"
[ -d "$ST-query" ] && pack_add "static/st-$FTAG-static-$T-query" "$ST-query" "the query side alone as a Sentence Transformers model (no document encoder)"

# 5. summary
log "=== st1 step 5: summary ==="
SUM="$RESULTS_ROOT/summary_st1.md"
{
    echo "# st1 (quantized regime, nDCG@10): the static query table over the index of the frozen full model"; echo
    echo "The recipe of phr2-k8-kd-r2 (s7 / s8) repeated with \`$FULL\` as the document side and the teacher: tables credited on the kd_v1 queries (token rows; rows for whole words and adjacent word pairs), a pool of 250 000 passages and 120 000 queries mined with the new index, 2 epochs, a second pool whose student candidates come from the trained tables, 3 epochs. The document encoder is frozen; the joint training of v54 is not repeated."; echo
    echo "| query side | index | tiny10 | heldout8 | val top1 | dims/query | postings fraction |"; echo "|---|---|---:|---:|---:|---:|---:|"
    echo "| **static $T (st1)** | $FULL | $(lbv "$ASYM" tiny10) | $(lbv "$ASYM" heldout8) | $(sval $T top1_agreement) | $(sval $T dims_per_query) | $(sval $T postings_fraction) |"
    echo "| static phr2-k8-kd (first pool only) | $FULL | - | - | $(sval phr2-k8-kd top1_agreement) | $(sval phr2-k8-kd dims_per_query) | $(sval phr2-k8-kd postings_fraction) |"
    echo "| static phr2-k8-kd-r2 (s7 / s8) | r2v12-bias005 | 60.20 | 38.60 | 0.671 | 69.8 | 8.440 |"
    echo "| the full model itself (symmetric) | $FULL | $(mlbv "$FULL" tiny10) | $(mlbv "$FULL" heldout8) | 1.000 | | |"
    echo "| r2v12-bias005 (symmetric) | r2v12-bias005 | 64.21 | 43.08 | 1.000 | | |"
    echo; echo "val columns: 2 000 held-out kd_v1 queries of the pool each table trained on (top-1 agreement with the full model; postings fraction as reported by distill.train_static, relative to its own reference). The val numbers of different pools are not comparable 1:1."
    echo; echo "Released table: \`$SM/$T\` (first line of \`logs/final_static_table.txt\`). Sentence Transformers package: \`$ST\` (router: query = the table, document = the frozen model; load with \`trust_remote_code=True\`) and \`$ST-query\` (the query side alone). Package check: $(tail -n 1 "$LOGS/st1_st_check.log" 2>/dev/null | cut -c1-300)"
    echo; echo "## Queue state (st1)"; echo; echo '```'; grep -E "static_(build|mine|train):.*$FTAG|$ASYM" "$STATE" | tail -n 40; echo '```'
} > "$SUM"
log "summary written to $SUM"
git add "$SUM" "$EXP_ROOT/queues/queue_st1.sh" "$RS" 2>/dev/null
[ -e "$REGIMES_ROOT/quantized/$ASYM" ] && git add "$REGIMES_ROOT/quantized/$ASYM" 2>/dev/null
git commit -q -m "Add the st1 results: the static query table over the frozen full model" 2>/dev/null && log "committed $(git log --oneline -1)"
push_loop st1
log "=== st1 done ($(date '+%d.%m %H:%M')); table $SM/$T ==="
