#!/usr/bin/env bash
# Queue x1 (3060 Ti, after st1; ~1 h): ONNX exports of the models that are released, and the package for the 5070.
# For the frozen full model (logs/final_full_model.txt) and the ALBERT to release (logs/final_albert_model.txt):
#   1  release.onnx_splade: fp32 and the dynamic mixed int8 (u8 weights, MLM head kept fp32: the repair that worked in
#      results/onnx/fix_r2v12b.md on a CPU without VNNI), written straight into the package
#   2  fidelity from the export report (fp32 ONNX against torch; mixed int8 against fp32: cosine, active Jaccard, top-10
#      terms, dims ratio) and tiny10 asymmetrically (torch index, ONNX mixed-int8 queries on the CPU)
#   3  the package datasets/transfer/release-20261113/: the models in release form, the static table and its Sentence
#      Transformers package (added by v62 / r1 / st1), the ONNX exports, benchmark texts and PIRB task sets, README.md
#      with the commands, SHA256SUMS
# NO CPU timing runs here: the README carries the release.onnx_bench command for the user to start by hand (or on the
# 5070). The full publication benchmark runs on the 5070.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_x1.sh > logs/queue_x1.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=x1 MIN_RAM_GB=6
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
echo $$ > "$LOGS/queue_x1.pid"
guard_alive
wait_for_queues ${WAIT_FOR:-v61 v61b v62 r1 st1}

SEXP=experiments/polish-splade-small-2026-11
THREADS=${THREADS:-6}
RES="$RESULTS_ROOT/onnx"; mkdir -p "$RES" "$PACK/onnx" "$PACK/bench"
D3B=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015
FULL=$(final_full); [ -n "$FULL" ] && [ -d "models/$FULL" ] || { log "  no frozen full model in $FINAL_FULL_FILE (${FULL:-empty}): dose3-bias015 is used"; FULL="$D3B"; }
ALB=$(final_albert); [ -n "$ALB" ] && [ -d "models/$ALB" ] || { log "  no ALBERT label in $FINAL_ALBERT_FILE (${ALB:-empty}): no ALBERT export"; ALB=""; }
STATIC=$(first_line "$FINAL_STATIC_FILE")
FTAG=$(echo "$FULL" | sed -e 's/^arm-dense-kl-n5kj-mined-//' -e 's/-ens400-r2v12//')
log "=== x1 start: ONNX exports of the full model $FULL and the ALBERT ${ALB:-none}; package $PACK; $(free_gb) GB free on / ==="
DATA_GB=$(df -BG --output=avail "$PACK/" | tail -1 | tr -dc '0-9')
[ "${DATA_GB:-0}" -ge 5 ] || { log "  only ${DATA_GB:-0} GB free under $PACK: x1 stops"; exit 1; }

note() {  # note DEST SRC NOTE: a line in CONTENTS.tsv for something written straight into the package
    touch "$PACK/CONTENTS.tsv"; grep -v -P "^\Q$1\E\t" "$PACK/CONTENTS.tsv" > "$PACK/CONTENTS.tsv.tmp"; mv "$PACK/CONTENTS.tsv.tmp" "$PACK/CONTENTS.tsv"
    printf '%s\t%s\t%s\t%s\n' "$1" "$(du -sh "$PACK/$1" | cut -f1)" "$2" "$3" >> "$PACK/CONTENTS.tsv"
}
rep() {  # rep REPORT PATH...: a number of an export report, e.g. rep R variants mixed fp32_check cosine mean
    "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]))
for k in sys.argv[2:]: d=d[k]
print('%.4f' % d if isinstance(d, float) else d)" "$@" 2>/dev/null
}
export_one() {  # export_one TAG LABEL: fp32 + dynamic mixed int8 into $PACK/onnx/TAG, tiny10 with ONNX mixed-int8 queries over the torch index
    local tag="$1" label="$2" out="$PACK/onnx/$1" s
    [ -d "models/$label" ] || return 1
    if [ -f "$out/export_report.json" ]; then log "export $tag: present"
    else
        mkdir -p "$out"; wait_gpu_free; wait_ram; s=$SECONDS
        if CUDA_VISIBLE_DEVICES= ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" -m release.onnx_splade --model "models/$label" --out "$out" \
                --variants fp32,mixed --mixed-method dynamic --weight-type u8 --calib-texts datasets/kd/kd_v1.jsonl --threads "$THREADS" > "$LOGS/onnx_export_$tag.log" 2>&1; then
            log "export $tag done in $((SECONDS - s)) s: fp32 against torch max |diff| $(rep "$out/export_report.json" variants fp32 torch_check max_abs_diff); mixed int8 against fp32: cosine $(rep "$out/export_report.json" variants mixed fp32_check cosine mean), active Jaccard $(rep "$out/export_report.json" variants mixed fp32_check active_jaccard mean), dims ratio $(rep "$out/export_report.json" variants mixed fp32_check dims_ratio)"
            record "onnx-export:$tag" ok "-" "$((SECONDS - s))s"
        else log "export $tag FAILED (logs/onnx_export_$tag.log)"; record "onnx-export:$tag" FAILED "-" "$((SECONDS - s))s"; return 1; fi
    fi
    cp "$out/export_report.json" "$RES/export_$tag.json"
    PIRB_QUERY_MODEL="$out/mixed-int8" regime quantized "onnx-$tag-q-mixed-int8" "models/$label" tiny10 16
    log "  $tag: tiny10 with ONNX mixed-int8 queries over the torch index $(lbv "onnx-$tag-q-mixed-int8" tiny10)"
    note "onnx/$tag" "models/$label" "ONNX exports of $label: fp32 and dynamic mixed int8 (u8 weights, MLM head fp32); mixed against fp32 cosine $(rep "$out/export_report.json" variants mixed fp32_check cosine mean); tiny10 with mixed-int8 queries $(lbv "onnx-$tag-q-mixed-int8" tiny10)"
}

# 1-2. exports and checks
log "=== x1 step 1: exports ==="
export_one "full-$FTAG" "$FULL"
[ -n "$ALB" ] && export_one "albert-$ALB" "$ALB"

# 3. the package
log "=== x1 step 2: the package ==="
[ -d "$PACK/full/$FULL" ] || pack_add "full/$FULL" "models/$FULL" "full model (82 M), release form with the bias baked in"
[ -n "$ALB" ] && { [ -d "$PACK/albert/$ALB" ] || pack_add "albert/$ALB" "models/$ALB" "ALBERT student (50 M), release form with the bias baked in"; }
if [ ! -s "$PACK/bench/texts_kd_v1_sample.jsonl" ]; then
    "$PY" - datasets/kd/kd_v1.jsonl "$PACK/bench/texts_kd_v1_sample.jsonl" <<'PYEOF'
import json, sys
need = {"query": 3000, "passage": 1500}; out = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        k = json.loads(line).get("kind")
        if need.get(k, 0) > 0:
            need[k] -= 1; out.append(line)
        if not any(need.values()):
            break
open(sys.argv[2], "w", encoding="utf-8").writelines(out)
PYEOF
fi
rm -rf "$PACK/bench/pirb_tasks"; cp -r configs/pirb_tasks "$PACK/bench/pirb_tasks"
mkdir -p "$PACK/results"
for f in "$RESULTS_ROOT/summary_v62.md" "$RESULTS_ROOT/v62_freeze.json" "$SEXP/results/summary_r1.md" "$SEXP/results/r1_freeze.json" "$SEXP/results/summary_st1.md" "$RES"/export_*.json; do [ -e "$f" ] && cp "$f" "$PACK/results/"; done
note "bench" "datasets/kd/kd_v1.jsonl, configs/pirb_tasks" "benchmark inputs: 3 000 queries + 1 500 passages of kd_v1 for release.onnx_bench; the PIRB task sets"
note "results" "$RESULTS_ROOT, $SEXP/results" "summaries and choices of v62 / r1 / st1, the ONNX export reports"
ATXT="(no ALBERT was chosen)"; [ -n "$ALB" ] && ATXT="\`albert/$ALB\`"
{
    echo "# Release package 2026-11-13 (from the 3060 box)"; echo
    echo "Models in the form they are released in (the bias is baked into the checkpoints), their ONNX exports and the inputs of the benchmarks. Check the copy first:"; echo
    echo '```bash'; echo "cd datasets/transfer/release-20261113 && sha256sum -c SHA256SUMS"; echo '```'; echo
    echo "## Contents"; echo; echo "| path | size | source | what |"; echo "|---|---:|---|---|"
    sort "$PACK/CONTENTS.tsv" | awk -F'\t' '{printf "| `%s` | %s | `%s` | %s |\n", $1, $2, $3, $4}'
    echo; echo "## Models"; echo
    echo "- Full model (82 M): \`full/$FULL\` — quantized regime tiny10 / heldout8 $(lbs "$FULL"), trec-covid $(tc "$FULL"), $(dims "$FULL") dims/doc. Choice and table: \`results/summary_v62.md\`."
    echo "- ALBERT student (50 M): $ATXT. Choice and table: \`results/summary_r1.md\`."
    echo "- Static query table: \`${STATIC:-not built}\` on the 3060; here under \`static/\` (the table, the Sentence Transformers model with the table as the query route, and the query side alone). Load the Sentence Transformers model with \`SparseEncoder(path, trust_remote_code=True)\`. Numbers: \`results/summary_st1.md\`."
    echo; echo "## Publication benchmark (PIRB) on the 5070"; echo
    echo "From the repository root, with P=datasets/transfer/release-20261113. One task set per call; the task sets are \`configs/pirb_tasks/*.json\` (copied to \`bench/pirb_tasks/\`): tiny10, heldout8, heldout2_czywiesz, heldout2_gpt35wiki, heldout2_quora, heldout2_poquad, heldout2_treccovid are the 23 tasks measured on the 3060; the remaining files are the large PIRB tasks. Batch 8 for poquad and trec-covid."; echo
    echo '```bash'
    echo "P=datasets/transfer/release-20261113"
    echo "# symmetric, quantized regime (the regime of every number in the summaries); old-pyserini = the 2023 leaderboard regime"
    echo "scripts/pirb_regime_eval.sh quantized $FULL \$P/full/$FULL tiny10 16"
    [ -n "$ALB" ] && echo "scripts/pirb_regime_eval.sh quantized $ALB \$P/albert/$ALB tiny10 16"
    echo "# asymmetric: the index of the full model, another query encoder"
    [ -n "$ALB" ] && echo "PIRB_QUERY_MODEL=\$P/albert/$ALB scripts/pirb_regime_eval.sh quantized asym-$FTAG-q-$ALB \$P/full/$FULL tiny10 16"
    for d in "$PACK"/static/*; do b=$(basename "$d"); case "$b" in st-*|'*') ;; *) echo "PIRB_QUERY_MODEL=\$P/static/$b scripts/pirb_regime_eval.sh quantized asym-$FTAG-q-static \$P/full/$FULL tiny10 16" ;; esac; done
    echo "PIRB_QUERY_MODEL=\$P/onnx/full-$FTAG/mixed-int8 scripts/pirb_regime_eval.sh quantized onnx-full-$FTAG-q-mixed-int8 \$P/full/$FULL tiny10 16"
    echo '```'
    echo; echo "## CPU benchmark (not run on the 3060: no CPU timing in the queues)"; echo
    echo "Start by hand on an idle machine; it times queries (batch 1) and passages (batch 8) at 1 and 6 threads and compares each variant with the fp32 export:"; echo
    echo '```bash'
    echo ".venv/bin/python -m release.onnx_bench --variants \$P/onnx/full-$FTAG/fp32 \$P/onnx/full-$FTAG/mixed-int8 --torch-model \$P/full/$FULL \\"
    echo "    --texts \$P/bench/texts_kd_v1_sample.jsonl --threads 1,6 --out experiments/polish-splade-2026-10/results/onnx/bench_full-$FTAG.json"
    [ -n "$ALB" ] && { echo ".venv/bin/python -m release.onnx_bench --variants \$P/onnx/albert-$ALB/fp32 \$P/onnx/albert-$ALB/mixed-int8 --torch-model \$P/albert/$ALB \\"
                       echo "    --texts \$P/bench/texts_kd_v1_sample.jsonl --threads 1,6 --out experiments/polish-splade-2026-10/results/onnx/bench_albert-$ALB.json"; }
    echo '```'
    echo; echo "ONNX fidelity (export reports in \`results/export_*.json\`): the fp32 export is checked against torch on 16 texts, the mixed int8 export against fp32 on 128 texts (cosine, active Jaccard, top-10 terms, dims ratio)."
} > "$PACK/README.md"
pack_sums
log "  package: $(du -shL "$PACK" | cut -f1) in $PACK ($(wc -l < "$PACK/SHA256SUMS") files in SHA256SUMS)"

# 4. summary
SUM="$RESULTS_ROOT/summary_x1.md"
{
    echo "# x1: ONNX exports of the released models and the package for the 5070"; echo
    echo "| model | export | fp32 vs torch: max abs diff | mixed int8 vs fp32: cosine | active Jaccard | top-10 terms | dims ratio | fp32 MB | mixed MB | tiny10, ONNX mixed-int8 queries over the torch index | tiny10 torch |"; echo "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for pair in "full-$FTAG=$FULL" ${ALB:+"albert-$ALB=$ALB"}; do
        tag=${pair%%=*}; label=${pair#*=}; r="$PACK/onnx/$tag/export_report.json"; [ -f "$r" ] || continue
        tt=$(lbv "$label" tiny10); [ -n "$tt" ] || tt=$(regime_mean "$SEXP/results-regimes/quantized/$label/tiny10.json" 2>/dev/null | awk '{print $1}')
        echo "| \`$label\` | \`onnx/$tag\` | $(rep "$r" variants fp32 torch_check max_abs_diff) | $(rep "$r" variants mixed fp32_check cosine mean) | $(rep "$r" variants mixed fp32_check active_jaccard mean) | $(rep "$r" variants mixed fp32_check top10_term_overlap mean) | $(rep "$r" variants mixed fp32_check dims_ratio) | $(( $(rep "$r" variants fp32 bytes) / 1048576 )) | $(( $(rep "$r" variants mixed bytes) / 1048576 )) | $(lbv "onnx-$tag-q-mixed-int8" tiny10) | $tt |"
    done
    echo; echo "Mixed int8 = dynamic int8 with u8 weights and the MLM head kept fp32 (\`release.onnx_splade --variants fp32,mixed --mixed-method dynamic --weight-type u8\`), no calibration. Earlier, for r2v12-bias005: cosine 0.9700, tiny10 64.14 against 64.21 in torch. No CPU timing was run; the commands are in the README of the package."
    echo; echo "## Package"; echo; echo "\`$PACK\` (a link to the data drive), $(du -shL "$PACK" | cut -f1):"; echo; echo '```'; sort "$PACK/CONTENTS.tsv" | cut -f1,2,4 | cut -c1-260; echo '```'
    echo; echo "## Queue state (x1)"; echo; echo '```'; grep -E "onnx-export:(full|albert)-|regime:quantized:onnx-(full|albert)-" "$STATE" | tail -n 20; echo '```'
} > "$SUM"
log "summary written to $SUM"
git add "$SUM" "$EXP_ROOT/queues/queue_x1.sh" "$RES"/export_*.json 2>/dev/null
for d in "$REGIMES_ROOT"/quantized/onnx-full-* "$REGIMES_ROOT"/quantized/onnx-albert-*; do [ -e "$d" ] && git add "$d" 2>/dev/null; done
git commit -q -m "Add the x1 results: ONNX exports of the released models" 2>/dev/null && log "committed $(git log --oneline -1)"
push_loop x1
log "=== x1 done ($(date '+%d.%m %H:%M')); package $PACK ==="
