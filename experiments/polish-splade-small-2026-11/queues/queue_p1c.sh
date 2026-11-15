#!/usr/bin/env bash
# Queue p1, part c (15.11, in the pause of b1 after its KD; ~40-50 min, at most 100 min): GPU inference speed of the
# frozen full model for the model card, the short version of PROMPT-3060-2026-11-15-gpu-bench.md.
#   0     smoke of release.gpu_bench (fp16 eager, fast head, torchao int8 on a small sample); when it fails the PyTorch
#         part is skipped
#   1-7   release.gpu_bench --variants fp16:eager fp16 fp16:fasthead,eager fp16:fasthead int8-dyn
#   11    ONNX fp32 with the CUDA provider; when it fails the TensorRT steps are skipped
#   12-13 ONNX through TensorRT: the released graph and the max-first graph
# The CPU step (10, one thread) is not run: no CPU timing is queued. Skipped as a whole when part b set skip57 (time).
# Run by logs/p1_chain.sh; every step is one job on the card.
cd "$(dirname "$0")/../../.." || exit 1
source experiments/polish-splade-small-2026-11/queues/p1_common.sh
echo $$ > "$LOGS/queue_p1.pid"
V=/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/FlagEmbedding-venvs
GQ="$MAIN/results/gpu-quant"; OX="$MAIN/results/onnx"; P=datasets/transfer/release-20261113/onnx/full-$FULL/fp32; MF=runs/onnx-gpu/dqf-maxfirst/fp32
LIMIT=6000; T0=$SECONDS
left() { echo $(( LIMIT - (SECONDS - T0) )); }
log "=== p1 part c start: GPU benchmarks of $FULL (short version) ==="
[ "$(sget skip57)" = 1 ] && { log "  skipped: more than 4 h had passed after A3; the benchmarks stay for later"; exit 0; }
mkdir -p "$GQ" "$OX" runs/gpu-quant runs/trt-cache
export ORPHAN_GUARD_EXEMPT=1

# PyTorch
if [ -x "$V/gpuq/bin/python" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS
    timeout -k 60 900 "$V/gpuq/bin/python" -m release.gpu_bench --model "$DOC" --n query=64,passage=256 --check query=32,passage=64 --repeats 1 --latency-n 20 \
        --variants fp16:eager fp16:fasthead int8-dyn --out runs/gpu-quant/smoke_3060ti.json > "$LOGS/p1_gpu_smoke.log" 2>&1; rc=$?
    log "  gpu_bench smoke: exit $rc in $((SECONDS - s))s; $(grep -c "failed:" "$LOGS/p1_gpu_smoke.log") variant(s) failed"
    grep -E "^\| " "$LOGS/p1_gpu_smoke.log" | cut -c1-300 | while read -r l; do log "    $l"; done
    record "p1:gpu_smoke" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    if [ $rc = 0 ] && ! grep -q -E "^\| *fp16:eager *\| *failed" "$LOGS/p1_gpu_smoke.log"; then
        wait_gpu_free; wait_ram; s=$SECONDS; to=$(left); [ "$to" -gt 3600 ] && to=3600
        timeout -k 60 "$to" "$V/gpuq/bin/python" -m release.gpu_bench --model "$DOC" --variants fp16:eager fp16 fp16:fasthead,eager fp16:fasthead int8-dyn \
            --out "$GQ/${FULL}_3060ti.json" > "$LOGS/p1_gpu_bench.log" 2>&1; rc=$?
        log "  gpu_bench: exit $rc in $((SECONDS - s))s ($GQ/${FULL}_3060ti.json)"
        grep -E "^\| " "$LOGS/p1_gpu_bench.log" | cut -c1-400 | while read -r l; do log "    $l"; done
        record "p1:gpu_bench" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    else log "  the smoke failed (logs/p1_gpu_smoke.log): the PyTorch variants are skipped"; fi
else log "  $V/gpuq/bin/python missing: the PyTorch part is skipped"; fi

# ONNX
ox() {  # ox TAG TIMEOUT VARIANT...: one release.onnx_bench call (the first variant is the reference)
    local tag="$1" to="$2" s rc; shift 2
    [ "$(left)" -gt 120 ] || { log "  onnx $tag: the 100 minutes are used up, skipped"; return 1; }
    [ "$to" -gt "$(left)" ] && to=$(left)
    wait_gpu_free; wait_ram; s=$SECONDS
    timeout -k 60 "$to" "$V/ort-gpu/bin/python-gpu" -m release.onnx_bench --variants "$@" --threads 1 --query-batch 1 --passage-batch 32 --max-rows 64 \
        --trt-cache runs/trt-cache --out "$OX/dqf_gpu_3060ti_$tag.json" > "$LOGS/p1_onnx_$tag.log" 2>&1; rc=$?
    log "  onnx_bench $tag: exit $rc in $((SECONDS - s))s ($OX/dqf_gpu_3060ti_$tag.json)"
    grep -E "^\| " "$LOGS/p1_onnx_$tag.log" | cut -c1-400 | while read -r l; do log "    $l"; done
    [ $rc = 0 ] || log "    $(tail -n 2 "$LOGS/p1_onnx_$tag.log" | tr '\n' ' ' | cut -c1-400)"
    record "p1:onnx_$tag" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    return $rc
}
if [ -x "$V/ort-gpu/bin/python-gpu" ] && [ -f "$P/model.onnx" ]; then
    if ox cuda 900 "$P@cuda"; then
        ox trt 1500 "$P@cuda" "$P@tensorrt"
        if [ -f "$MF/model.onnx" ]; then ox trt_maxfirst 1500 "$P@cuda" "$MF@tensorrt"; else log "  $MF/model.onnx missing: the max-first graph is skipped"; fi
    else log "  ONNX with the CUDA provider failed (logs/p1_onnx_cuda.log): the TensorRT steps are skipped"; fi
else log "  $V/ort-gpu/bin/python-gpu or $P/model.onnx missing: the ONNX part is skipped"; fi
log "=== p1 part c done ($(date '+%d.%m %H:%M'), $(( (SECONDS - T0) / 60 )) min; $(elapsed_min) min since the start of p1) ==="
