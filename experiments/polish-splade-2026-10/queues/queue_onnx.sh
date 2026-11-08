#!/usr/bin/env bash
# ONNX release check for one model (prepared 7.11, NOT scheduled). Start it only when the 3060 has nothing else to do:
# the CPU timings are worthless next to a training run, and the export plus a 50 k-vocabulary calibration want a few GB.
#   1  export fp32 / dynamic-int8 / static-int8 / mixed-int8 of $MODEL into models/onnx/$NAME (CPU only, $THREADS threads;
#      static and mixed share one MinMax calibration on 256 queries + 256 passages of kd_v1; mixed keeps the MLM head fp32,
#      MIXED_AUTO_K more blocks chosen by the per-block sensitivity probe, default 1).
#   2  (runs LAST, only after the user's go: touch logs/onnx_cpu_bench.ok when nothing else uses the CPU)
#      CPU benchmark of the four exports and the PyTorch fp32 checkpoint at 1 and 6 threads: query latency (batch 1),
#      passage throughput (batch 8), RSS, file size, agreement with fp32 (cosine, indexed-dim Jaccard, dims per query and
#      per passage, in-sample top-10 agreement) -> $RESULTS_ROOT/onnx/${NAME}_cpu.json and .md.
#   3  PIRB, leaderboard regime (quantized), asymmetric: the index is built by the torch model on the GPU, the queries are
#      encoded by each ONNX export on the CPU (the deployment where only the query side runs on user hardware).
#      tiny10 + heldout8 for every variant; labels onnx-$NAME-q-<variant>; reference = the existing $LABEL results.
#   4  PIRB, leaderboard regime, symmetric: both sides ONNX on the CPU, $SYM_TASKSETS (default tiny10) for every variant;
#      labels onnx-$NAME-<variant>. The fp32 export is the reference for the int8 ones (the torch numbers are fp16 on the GPU).
#   5  summary table -> $RESULTS_ROOT/onnx/summary_${NAME}.md
#   MODEL=models/arm-dense-kl-n5kj-ens400 NAME=n5kj LABEL=arm-dense-kl-n5kj-ens400 \
#     setsid nohup bash experiments/polish-splade-2026-10/queues/queue_onnx.sh > logs/queue_onnx.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=onnx MIN_RAM_GB="${MIN_RAM_GB:-8}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_onnx.pid"
guard_alive

MODEL="${MODEL:-models/arm-dense-kl-n5kj-ens400}"
NAME="${NAME:-n5kj}"
LABEL="${LABEL:-$(basename "$MODEL")}"
THREADS="${THREADS:-6}"
MIXED_AUTO_K="${MIXED_AUTO_K:-1}"
ASYM_TASKSETS="${ASYM_TASKSETS:-tiny10 heldout8}"
SYM_TASKSETS="${SYM_TASKSETS:-tiny10}"
read -ra VARIANTS <<< "${VARIANTS:-fp32 dynamic-int8 static-int8 mixed-int8}"  # 8.11: r2v12b runs "fp32 mixed-int8" (the int8 ones broke)
OUT="models/onnx/$NAME"
RES="$RESULTS_ROOT/onnx"
mkdir -p "$RES"

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
[ -d "$MODEL" ] || { log "model $MODEL missing"; exit 1; }
log "=== onnx start: $MODEL -> $OUT ($THREADS threads); $(free_gb) GB free, $(free -g | awk '/^Mem:/ {print $7}') GB RAM available ==="

# 1 export
if [ -f "$OUT/export_report.json" ]; then
    log "export: $OUT/export_report.json present, skipping"
else
    wait_gpu_free; wait_ram; s=$SECONDS
    if CUDA_VISIBLE_DEVICES= timeout -k 60 7200 "$PY" -m release.onnx_splade --model "$MODEL" --out "$OUT" \
            --variants fp32,dynamic,static,mixed --calib-texts datasets/kd/kd_v1.jsonl --threads "$THREADS" \
            --mixed-auto-k "$MIXED_AUTO_K" > "$LOGS/onnx_export_${NAME}.log" 2>&1; then
        log "export done in $((SECONDS - s)) s: $(grep -E 'MB' "$LOGS/onnx_export_${NAME}.log" | tail -4 | sed 's/.*INFO //' | tr '\n' ';')"
        record "onnx-export:$NAME" ok "-" "$((SECONDS - s))s"
    else
        log "export FAILED after $((SECONDS - s)) s (see $LOGS/onnx_export_${NAME}.log)"; record "onnx-export:$NAME" FAILED "-" "$((SECONDS - s))s"; exit 1
    fi
fi

# 3 asymmetric PIRB: torch index, ONNX queries. Order (8.11): the first asymmetric taskset (tiny10) for every variant,
#    then the symmetric tasksets, then the remaining asymmetric tasksets (heldout8): the cheap readings land first.
asym() { for v in "${VARIANTS[@]}"; do PIRB_QUERY_MODEL="$OUT/$v" regime quantized "onnx-$NAME-q-$v" "$MODEL" "$1" 16; done; }
set -- $ASYM_TASKSETS
[ $# -gt 0 ] && { asym "$1"; shift; }

# 4 symmetric PIRB: both sides ONNX on the CPU (regime() still waits for an idle GPU and free RAM: the CPU encode is the bottleneck)
for ts in $SYM_TASKSETS; do
    for v in "${VARIANTS[@]}"; do
        regime quantized "onnx-$NAME-$v" "$OUT/$v" "$ts" 16
    done
done

# 3b the remaining asymmetric tasksets
for ts in "$@"; do asym "$ts"; done

# 2 CPU benchmark: NOT queued (8.11, the user's rule). It runs only when the user explicitly says so, by hand:
#   python -m release.onnx_bench --variants <export dirs> --torch-model $MODEL --threads 1,$THREADS \
#     --n query=64,passage=32 --latency-n 50 --warmup 5 --out $RES/${NAME}_cpu.json
log "cpu bench: not queued; it runs only on the user's explicit go"

# 5 summary
"$PY" - "$REGIMES_ROOT/quantized" "$NAME" "$LABEL" "$RES" "$ASYM_TASKSETS" "$SYM_TASKSETS" "${VARIANTS[*]}" <<'EOF' > "$RES/summary_${NAME}.md"
import json, sys
from pathlib import Path
root, name, label, res, asym, sym = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5].split(), sys.argv[6].split()
variants = sys.argv[7].split() if len(sys.argv) > 7 else ["fp32", "dynamic-int8", "static-int8", "mixed-int8"]
def mean(lab, ts):
    p = root / lab / f"{ts}.json"
    try:
        v = json.loads(p.read_text())["mean_ndcg@10"]
        return f"{v:.2f}" if v is not None else "-"
    except (OSError, ValueError, KeyError):
        return "-"
print(f"# ONNX exports of {name} ({label})\n")
print("## PIRB, leaderboard regime (mean NDCG@10)\n")
cols = [f"q-only {t}" for t in asym] + [f"both {t}" for t in sym]
print("| variant | " + " | ".join(cols) + " |")
print("|---|" + "---|" * len(cols))
print(f"| torch fp16 (GPU) | " + " | ".join([mean(label, t) for t in asym] + [mean(label, t) for t in sym]) + " |")
for v in variants:
    print(f"| {v} | " + " | ".join([mean(f"onnx-{name}-q-{v}", t) for t in asym] + [mean(f"onnx-{name}-{v}", t) for t in sym]) + " |")
md = res / f"{name}_cpu.md"
if md.is_file():
    print("\n## CPU benchmark\n")
    print(md.read_text())
EOF
log "summary: $RES/summary_${NAME}.md"
# commit: the release package (written 7.11, first run here), its docs and deps, the queue, the results; the PIRB fork's
# OnnxSpladeEncoder lives in the submodule and is committed there (no submodule file paths in this add)
if [ -n "$(git -C third_party/pirb status --short search/splade.py)" ]; then
    git -C third_party/pirb add search/splade.py && git -C third_party/pirb commit -q -m "Add an ONNX runtime encoder for SPLADE export directories" \
        && log "submodule committed $(git -C third_party/pirb log --oneline -1)"
fi
git add third_party/pirb release docs/onnx-release.md requirements.in requirements.lock "$EXP_ROOT/queues/queue_onnx.sh" \
    "$RES/summary_${NAME}.md" "$RES/fix_${NAME}.md" "$EXP_ROOT/queues/queue_onnxfix.sh" "$EXP_ROOT/queues/queue_onnxdm.sh" tests/test_onnx_release.py "$RES/${NAME}_cpu.json" "$RES/${NAME}_cpu.md" "$REGIMES_ROOT"/quantized/onnx-"$NAME"-* 2>/dev/null
git commit -q -m "Add the ONNX release exports of $NAME: int8 variants, CPU benchmark and PIRB checks" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_onnx_${NAME}.log" 2>&1 && { echo pushed >> "$LOGS/push_onnx_${NAME}.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== onnx done ($(date '+%d.%m %H:%M')) ==="
