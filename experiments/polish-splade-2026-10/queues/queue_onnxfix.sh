#!/usr/bin/env bash
# Repair attempt for the broken int8 exports of r2v12b (8.11): dynamic-int8 tiny10 0.25, static-int8 13.66
# (fp32 64.22, mixed-int8 63.69). fp32 checks: dynamic cosine 0.03 with 2.7x the dims, static 0.14 with 26x the dims;
# a single int8 block costs 1-cos 0.11 in layer.4 alone. Hypothesis: the 3060 box has AVX2 without VNNI, where
# onnxruntime's u8s8 / s8s8 int8 GEMM (vpmaddubsw) saturates its int16 pair sums; reduce_range (7-bit weights) or
# u8 weights avoid that. A clipped calibration (percentile) is the second, outlier-range hypothesis.
#   1  exports (CPU, a few minutes each; the fp32 graph is shared by symlink, nothing is re-exported):
#        rr      --reduce-range
#        u8      --weight-type u8 --activation-type u8
#        rr-pct  --reduce-range --calib-method percentile (99.999)
#      every config writes dynamic, static and mixed (head fp32 + 1 auto block) with its own fp32 check.
#   2  fix table -> $RES/fix_$NAME.md (fp32 checks of every config next to the original exports)
#   3  PIRB asymmetric tiny10 (torch index, ONNX queries) for the best config of each of dynamic / static / mixed
#      (best = highest fp32 cosine); dynamic and static only when that cosine reaches $MIN_COS, mixed only when it
#      beats the original mixed export's 0.917 by 0.02. Labels onnx-$NAME-<config>-q-<variant>.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=onnxfix MIN_RAM_GB="${MIN_RAM_GB:-8}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_onnxfix.pid"
guard_alive

MODEL="${MODEL:-models/arm-dense-kl-n5kj-ens400-r2v12-bias005}"
NAME="${NAME:-r2v12b}"
THREADS="${THREADS:-6}"
MIN_COS="${MIN_COS:-0.9}"
BASE="models/onnx/$NAME"
RES="$RESULTS_ROOT/onnx"
mkdir -p "$RES"

for p in ${WAIT_PIDS:-}; do
    if kill -0 "$p" 2>/dev/null; then log "waiting for pid $p to exit"; while kill -0 "$p" 2>/dev/null; do sleep 30; done; sleep 20; fi
done
[ -d "$BASE/fp32" ] || { log "$BASE/fp32 missing"; exit 1; }
log "=== onnxfix start: $MODEL ($THREADS threads); $(free -g | awk '/^Mem:/ {print $7}') GB RAM available ==="

declare -A FLAGS=(
    [rr]="--reduce-range"
    [u8]="--weight-type u8 --activation-type u8"
    [rr-pct]="--reduce-range --calib-method percentile"
)
CONFIGS=(rr u8 rr-pct)
for c in "${CONFIGS[@]}"; do
    out="models/onnx/$NAME-$c"
    if [ -f "$out/export_report.json" ]; then log "export $c: present"; continue; fi
    mkdir -p "$out"; [ -e "$out/fp32" ] || ln -s "$(realpath "$BASE/fp32")" "$out/fp32"
    wait_gpu_free; wait_ram; s=$SECONDS
    if CUDA_VISIBLE_DEVICES= timeout -k 60 3600 "$PY" -m release.onnx_splade --model "$MODEL" --out "$out" \
            --variants fp32,dynamic,static,mixed --calib-texts datasets/kd/kd_v1.jsonl --threads "$THREADS" \
            --mixed-auto-k 1 ${FLAGS[$c]} > "$LOGS/onnx_export_${NAME}-$c.log" 2>&1; then
        log "export $c done in $((SECONDS - s)) s: $(grep -E 'int8 .*cosine' "$LOGS/onnx_export_${NAME}-$c.log" | tail -3 | sed 's/.*INFO //' | tr -s ' ' | tr '\n' ';')"
        record "onnx-export:$NAME-$c" ok "-" "$((SECONDS - s))s"
    else
        log "export $c FAILED after $((SECONDS - s)) s (see $LOGS/onnx_export_${NAME}-$c.log)"; record "onnx-export:$NAME-$c" FAILED "-" "$((SECONDS - s))s"
    fi
done

# 2 fix table + the PIRB picks (one line per pick: "<config> <variant dir name>")
"$PY" - "$NAME" "$MIN_COS" "$RES/fix_${NAME}.md" "${CONFIGS[@]}" > "$LOGS/onnxfix_picks.txt" <<'EOF'
import json, sys
from pathlib import Path
name, min_cos, md, configs = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
kinds = {"dynamic": "dynamic-int8", "static": "static-int8", "mixed": "mixed-int8"}
rows, best = [], {}
for cfg in ["orig"] + configs:
    p = Path(f"models/onnx/{name}" + ("" if cfg == "orig" else f"-{cfg}")) / "export_report.json"
    if not p.is_file():
        continue
    rep = json.loads(p.read_text())
    for kind, vname in kinds.items():
        v = rep["variants"].get(kind)
        if not v or not v.get("fp32_check"):
            continue
        ch = v["fp32_check"]
        extra = ",".join(v["quantization"].get("fp32_blocks", [])) if kind == "mixed" else ""
        rows.append((cfg, vname, ch["cosine"]["mean"], ch["active_jaccard"]["mean"], ch["top10_term_overlap"]["mean"], ch["dims_ratio"], v["bytes"] / 2**20, extra))
        if cfg != "orig" and ch["cosine"]["mean"] > best.get(kind, ("", -1))[1]:
            best[kind] = (cfg, ch["cosine"]["mean"])
lines = [f"# Int8 repair attempt for {name}\n",
         "fp32 checks on 128 held-out kd_v1 texts (64 queries, 64 passages). orig = the first export (no reduce_range, s8 weights, minmax).\n",
         "| config | variant | cosine | active Jaccard | top-10 terms | dims ratio | MB | fp32 blocks |", "|---|---|---:|---:|---:|---:|---:|---|"]
lines += [f"| {r[0]} | {r[1]} | {r[2]:.4f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.0f} | {r[7]} |" for r in rows]
md.write_text("\n".join(lines) + "\n")
orig_mixed = next((r[2] for r in rows if r[0] == "orig" and r[1] == "mixed-int8"), 0.917)
for kind, (cfg, cos) in best.items():
    if (kind == "mixed" and cos >= orig_mixed + 0.02) or (kind != "mixed" and cos >= min_cos):
        print(cfg, kinds[kind])
EOF
log "fix table: $RES/fix_${NAME}.md; picks: $(tr '\n' ';' < "$LOGS/onnxfix_picks.txt")"

# 2b the CPU benchmark of the picks runs in queue_onnx.sh (last, on the user's go), not here

# 3 PIRB asymmetric tiny10 for the picks
while read -r cfg v; do
    [ -n "$cfg" ] || continue
    PIRB_QUERY_MODEL="models/onnx/$NAME-$cfg/$v" regime quantized "onnx-$NAME-$cfg-q-$v" "$MODEL" tiny10 16
done < "$LOGS/onnxfix_picks.txt"

"$PY" - "$REGIMES_ROOT/quantized" "$NAME" "$RES/fix_${NAME}.md" < "$LOGS/onnxfix_picks.txt" <<'EOF'
import json, sys
from pathlib import Path
root, name, md = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
def mean(label):
    try:
        return f"{json.loads((root / label / 'tiny10.json').read_text())['mean_ndcg@10']:.2f}"
    except (OSError, ValueError, KeyError, TypeError):
        return "-"
out = ["\n## PIRB tiny10, queries ONNX on the CPU, index torch (leaderboard regime)\n", "| config | variant | tiny10 |", "|---|---|---:|",
       f"| torch fp16 | - | {mean('arm-dense-kl-n5kj-ens400-r2v12-bias005')} |"]
out += [f"| orig | {v} | {mean(f'onnx-{name}-q-{v}')} |" for v in ("fp32", "dynamic-int8", "static-int8", "mixed-int8")]
for line in sys.stdin:
    if line.split():
        cfg, v = line.split()
        out.append(f"| {cfg} | {v} | {mean(f'onnx-{name}-{cfg}-q-{v}')} |")
md.write_text(md.read_text() + "\n".join(out) + "\n")
EOF
log "=== onnxfix done ($(date '+%d.%m %H:%M')) ==="
