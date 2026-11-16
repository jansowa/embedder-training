#!/usr/bin/env bash
# Queue x2 (16.11, at a pause point of queue b1c after a point of the dose curve; ~45 min): the int8 exports of the
# released models with the MLM head in int8 as well. The released mixed variant keeps the head (the projection onto the
# 50 001 vocabulary entries, about half of the multiplications of the 6-layer model) in fp32; on the earlier model of
# the same architecture the full dynamic int8 encoded a query in 4.9 ms against 9.0 ms (6 threads) at the same PIRB
# numbers (results/onnx/summary_r2v12b.md). The user's decision of 16.11: build and measure it for the release models.
#   1  full model: dynamic int8 of every MatMul / Gemm (u8 weights), built from the fp32 graph of the package
#   2  ALBERT: the repaired variant (the extra head layer and the FFN output of group 2 fp32) with the MLM head in int8
#   for both: agreement with the fp32 export on the 128 check texts, tiny10 with the int8 queries over the torch index
#   3  a variant whose tiny10 is at most 0.15 below the head-fp32 int8 variant goes NEXT TO the package as an additional
#      path of $UPD (nothing is replaced, the package is not touched); README section and SHA256SUMS of $UPD
# No CPU timing. Run by logs/p3_chain.sh; every step is one job.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=x2 MIN_RAM_GB=6
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
echo $$ > "$LOGS/queue_x2.pid"
guard_alive
START=$SECONDS
THREADS=${THREADS:-6}
UPD=datasets/transfer/release-20261113-update1
WORK=runs/onnx-int8-full
RES="$RESULTS_ROOT/onnx"; mkdir -p "$RES" "$WORK"
FULL=$(final_full); ALB=$(final_albert)
MARGIN=0.15
log "=== x2 start: int8 exports with the MLM head in int8 for $FULL and $ALB; $(free_gb) GB free on / ==="

num() {  # num JSON PATH...: a number of a JSON file
    "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]))
for k in sys.argv[2:]: d=d[k]
print('%.4f' % d if isinstance(d, float) else d)" "$@" 2>/dev/null
}
tv() { local v; v=$(lbv "$1" tiny10); [ -n "$v" ] || v=$(regime_mean "experiments/polish-splade-small-2026-11/results-regimes/quantized/$1/tiny10.json" | awk '{print $1}'); echo "$v"; }   # torch tiny10 of a model, either series
ok_margin() { "$PY" -c "import sys;print(1 if float(sys.argv[1]) >= float(sys.argv[2]) - float(sys.argv[3]) else 0)" "$1" "$2" "$MARGIN" 2>/dev/null; }
build() {  # build TAG VARIANT [int8_sensitivity args]: one int8 variant from the fp32 graph of the package -> $WORK/TAG/VARIANT
    local tag="$1" var="$2" s; shift 2
    [ -f "$PACK/onnx/$tag/fp32/model.onnx" ] || { log "  $tag: no fp32 export in the package"; return 1; }
    if [ ! -f "$WORK/$tag/$var.json" ]; then
        wait_ram; s=$SECONDS
        CUDA_VISIBLE_DEVICES= ORPHAN_GUARD_EXEMPT=1 timeout -k 60 1800 "$PY" -m release.int8_sensitivity --fp32-dir "$PACK/onnx/$tag/fp32" --texts datasets/kd/kd_v1.jsonl \
            --out "$WORK/$tag" --fp32-groups "" --weight-type u8 --threads "$THREADS" --build "$var" "$@" > "$LOGS/x2_build_$tag.log" 2>&1
        record "x2:build:$tag" "$([ -f "$WORK/$tag/$var.json" ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    fi
    [ -f "$WORK/$tag/$var.json" ] || { log "  $tag: building $var FAILED (logs/x2_build_$tag.log): $(tail -n 1 "$LOGS/x2_build_$tag.log" | cut -c1-300)"; return 1; }
    cp "$WORK/$tag/$var.json" "$RES/int8_full_$tag.json"
    log "  $tag $var: int8 nodes $(num "$WORK/$tag/$var.json" int8_nodes), fp32 groups '$("$PY" -c "import json,sys;print(','.join(json.load(open(sys.argv[1]))['quantization']['fp32_groups']))" "$WORK/$tag/$var.json")'; against fp32: cosine $(num "$WORK/$tag/$var.json" fp32_check cosine mean), active Jaccard $(num "$WORK/$tag/$var.json" fp32_check active_jaccard mean), dims ratio $(num "$WORK/$tag/$var.json" fp32_check dims_ratio); $(du -sh "$WORK/$tag/$var" | cut -f1)"
}
measure() {  # measure TAG VARIANT MODEL_LABEL REF NOTE: tiny10 with the int8 queries over the torch index; REF = tiny10 of the head-fp32 int8 variant
    local tag="$1" var="$2" label="$3" ref="$4" lab="onnx-$1-q-$2" v
    [ -f "$WORK/$tag/$var/model.onnx" ] || return 1
    PIRB_QUERY_MODEL="$WORK/$tag/$var" regime quantized "$lab" "models/$label" tiny10 16
    v=$(lbv "$lab" tiny10)
    [ -n "$v" ] || { log "  $tag $var: tiny10 was not measured"; return 1; }
    log "  $tag $var: tiny10 with int8 queries $v (head-fp32 int8 $ref, torch $5)"
    if [ "$(ok_margin "$v" "$ref")" = 1 ]; then
        if [ ! -e "$UPD/onnx/$tag/$var" ]; then
            mkdir -p "$UPD/onnx/$tag" && cp -r "$WORK/$tag/$var" "$UPD/onnx/$tag/$var" && cp "$WORK/$tag/$var.json" "$UPD/onnx/$tag/$var.json" \
                && log "  $UPD/onnx/$tag/$var <- $WORK/$tag/$var ($(du -sh "$UPD/onnx/$tag/$var" | cut -f1)), an additional path" || log "  copying $var of $tag FAILED"
        fi
        ADDED="$ADDED $tag/$var"
    else
        log "  $tag $var is more than $MARGIN below the head-fp32 variant: it stays in $WORK, nothing is added next to the package"
    fi
    ROWS="$ROWS| \`onnx/$tag/$var\` | $(num "$WORK/$tag/$var.json" int8_nodes) | $(num "$WORK/$tag/$var.json" fp32_check cosine mean) | $(num "$WORK/$tag/$var.json" fp32_check active_jaccard mean) | $(num "$WORK/$tag/$var.json" fp32_check dims_ratio) | $(du -sh "$WORK/$tag/$var" | cut -f1) | $v | $ref | $5 |"$'\n'
}
ADDED=""; ROWS=""

# ---------------------------------------------------------------- 1. the full model
log "=== x2 step 1: $FULL, dynamic int8 of every node ==="
FT="full-$FULL"
build "$FT" dynamic-int8 && measure "$FT" dynamic-int8 "$FULL" "$(lbv "onnx-$FT-q-mixed-int8" tiny10)" "$(tv "$FULL")"

# ---------------------------------------------------------------- 2. the ALBERT
log "=== x2 step 2: $ALB, the repaired variant with the MLM head in int8 ==="
AT="albert-$ALB"; REP="runs/onnx-int8-diag/$AT/int8_sensitivity.json"
if [ -f "$REP" ]; then
    build "$AT" mixed-int8-int8head --fp32-nodes-from "$REP" --int8-groups head \
        && measure "$AT" mixed-int8-int8head "$ALB" "$(lbv "onnx-$AT-q-mixed-int8-repaired" tiny10)" "$(tv "$ALB")"
else log "  $REP is missing: no variant of the ALBERT"; fi

# ---------------------------------------------------------------- 3. the update directory
if [ -n "$ADDED" ] && [ -f "$UPD/README.md" ] && ! grep -q "^## Added 16.11: int8 variants with the MLM head in int8" "$UPD/README.md"; then
    {
        echo; echo "## Added 16.11: int8 variants with the MLM head in int8"; echo
        echo "Additional paths, nothing in the package is replaced by them. The mixed int8 variants keep the MLM head (the projection onto the vocabulary, about half of the multiplications of the 6-layer model) in fp32; in these variants it is dynamic int8 as well (u8 weights), which makes the file smaller and the encoding faster. Built by \`release.int8_sensitivity --build\` from the fp32 graph of the package; the report of each variant is the \`.json\` next to it. tiny10 with the int8 queries over the torch index of the same model; no CPU timing was run."; echo
        echo "| path | int8 nodes | cosine to fp32 | active Jaccard | dims ratio | size | tiny10 with int8 queries | head-fp32 int8 variant | torch |"; echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        printf '%s' "$ROWS"
        echo; echo "Added next to the package (tiny10 at most $MARGIN below the head-fp32 variant):$(for a in $ADDED; do printf ' `onnx/%s`' "$a"; done). heldout8 with int8 queries was not measured."
    } >> "$UPD/README.md"
    ( cd "$UPD" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS )
    log "  $UPD: $(du -sh "$UPD" | cut -f1), $(grep -c . "$UPD/SHA256SUMS") files in SHA256SUMS, README section added"
elif [ -z "$ADDED" ]; then log "  nothing was added next to the package"; fi
printf '%s' "$ROWS" > "$RES/int8_full_rows.md"
log "=== x2 done ($(date '+%d.%m %H:%M'), $(( (SECONDS - START) / 60 )) min) ==="
