#!/usr/bin/env bash
# Dynamic mixed export of r2v12b (8.11): dynamic int8 (u8 weights, the best repair in fix_r2v12b.md) with the MLM head
# kept fp32 (release.onnx_splade --mixed-method dynamic), no calibration. Then PIRB tiny10, asymmetric (torch index,
# ONNX queries on the CPU), label onnx-$NAME-dm-q-mixed-int8; the result is appended to $RES/fix_$NAME.md.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=onnxdm MIN_RAM_GB="${MIN_RAM_GB:-6}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_onnxdm.pid"
guard_alive

MODEL="${MODEL:-models/arm-dense-kl-n5kj-ens400-r2v12-bias005}"
NAME="${NAME:-r2v12b}"
THREADS="${THREADS:-6}"
OUT="models/onnx/$NAME-dm"
RES="$RESULTS_ROOT/onnx"

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 30; done
        sleep 20
    fi
done

if [ -f "$OUT/export_report.json" ]; then
    log "export: present"
else
    mkdir -p "$OUT"; [ -e "$OUT/fp32" ] || ln -s "$(realpath "models/onnx/$NAME/fp32")" "$OUT/fp32"
    wait_gpu_free; wait_ram; s=$SECONDS
    if CUDA_VISIBLE_DEVICES= timeout -k 60 3600 "$PY" -m release.onnx_splade --model "$MODEL" --out "$OUT" \
            --variants fp32,mixed --mixed-method dynamic --weight-type u8 --calib-texts datasets/kd/kd_v1.jsonl \
            --threads "$THREADS" > "$LOGS/onnx_export_${NAME}-dm.log" 2>&1; then
        log "export done in $((SECONDS - s)) s: $(grep -E 'mixed-int8 vs fp32' "$LOGS/onnx_export_${NAME}-dm.log" | sed 's/.*INFO //')"
        record "onnx-export:$NAME-dm" ok "-" "$((SECONDS - s))s"
    else
        log "export FAILED (see $LOGS/onnx_export_${NAME}-dm.log)"; record "onnx-export:$NAME-dm" FAILED "-" "$((SECONDS - s))s"; exit 1
    fi
fi

PIRB_QUERY_MODEL="$OUT/mixed-int8" regime quantized "onnx-$NAME-dm-q-mixed-int8" "$MODEL" tiny10 16

"$PY" - "$OUT/export_report.json" "$REGIMES_ROOT/quantized/onnx-$NAME-dm-q-mixed-int8/tiny10.json" "$RES/fix_${NAME}.md" <<'EOF'
import json, sys
from pathlib import Path
rep, res, md = (Path(a) for a in sys.argv[1:4])
v = json.loads(rep.read_text())["variants"]["mixed"]
ch = v["fp32_check"]
try:
    t = f"{json.loads(res.read_text())['mean_ndcg@10']:.2f}"
except (OSError, ValueError, KeyError, TypeError):
    t = "-"
md.write_text(md.read_text() + "\n## Dynamic mixed (u8 dynamic int8, MLM head fp32, no calibration)\n\n"
              "| cosine | active Jaccard | top-10 terms | dims ratio | MB | tiny10 (q-only) |\n|---:|---:|---:|---:|---:|---:|\n"
              f"| {ch['cosine']['mean']:.4f} | {ch['active_jaccard']['mean']:.3f} | {ch['top10_term_overlap']['mean']:.3f} | "
              f"{ch['dims_ratio']:.3f} | {v['bytes'] / 2**20:.0f} | {t} |\n")
EOF
log "=== onnxdm done ($(date '+%d.%m %H:%M')) ==="
