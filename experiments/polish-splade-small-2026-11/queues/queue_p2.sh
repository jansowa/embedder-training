#!/usr/bin/env bash
# Queue p2 (night 15/16.11, at the pause point of queue b1c after the R-a training; ~2-2.5 h): the user's three decisions
# of the evening of 15.11 after the results of p1.
#   1  ONNX int8 of the released ALBERT: the repaired mixed variant (the extra head layer and the FFN output of group 2
#      back in fp32; cosine 0.998 to fp32, tiny10 with int8 queries 64.30 against 62.10) as a NEW VERSION OF THE PACKAGE
#      FILES NEXT TO THE PACKAGE ($UPD); the package itself is not touched.
#   2  THE CHEAPER TABLE: p1 found a table with the quality of the released one at 16 % fewer postings per query
#      (a1-sparseadam-mw020). The user: worth swapping when the quality is really the same. Measured here: the released
#      table itself with the same threshold (no training; tiny10, cost), heldout8 of both, the cost on the heldout8
#      tasks. "The same quality" = tiny10 and heldout8 each at most 0.10 below the released table (60.74 / 39.24) and
#      at least 500 postings per query cheaper; the cheapest such table is exported as a Sentence Transformers package
#      and put next to the package with a README. The swap itself stays the user's decision.
#   3  AN IMPROVEMENT OF THE TABLE: the oracles of p1 put the ceiling of context-dependent weights at +2.2 on tiny10,
#      so one probe of the context gate (path B3, format 2, d = 32): fit to the teacher's query vectors, ranking
#      training on pool r2, tiny10, cost. Format 2 has no Sentence Transformers module yet, so this cannot enter the
#      release of 26.11; the probe says whether the module is worth building.
# Run by logs/p2_chain.sh; every step is one job on the card.
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=p2
source experiments/polish-splade-small-2026-11/queues/p1_common.sh
echo $$ > "$LOGS/queue_p2.pid"
START=$SECONDS
ALB=$(sed -n '1p' "$LOGS/final_albert_model.txt" | awk '{print $1}')
PACK=datasets/transfer/release-20261113; UPD=datasets/transfer/release-20261113-update1
OD="runs/onnx-int8-diag/albert-$ALB"
SM2="models/static2-$FULL"
H8TASKS="maupqa-gpt3.5-cc,maupqa-multilingual-NLI,mfaq,poleval-2022/legal-questions,maupqa-wiki-def,scidocs-pl,maupqa-1z10,fiqa-pl"
h8v() { lbv "$(lab "$1")" heldout8; }
h8n() { regime_mean "$REGIMES_ROOT/quantized/$(lab "$1")/heldout8.json" 2>/dev/null | awk '{print $2}'; }
h8a() {  # h8a NAME: asymmetric heldout8 with all 8 tasks (fiqa-pl alone and merged when it failed in the batch run)
    ev "$1" heldout8
    if [ "$(h8n "$1")" = 7 ]; then
        log "  $1: heldout8 has 7 tasks; fiqa-pl alone"
        PIRB_QUERY_MODEL="$SM/$1" regime quantized "$(lab "$1")" "$DOC" heldout8_fiqa 8 && "$PY" scripts/merge_regime_task.py --into "$REGIMES_ROOT/quantized/$(lab "$1")/heldout8.json" \
            --from "$REGIMES_ROOT/quantized/$(lab "$1")/heldout8_fiqa.json" --n-tasks 8 | while read -r l; do log "  $l"; done
    fi
    [ "$(h8n "$1")" = 8 ] || { log "  $1: heldout8 is not complete ($(h8n "$1") tasks): not usable for the comparison"; return 1; }
}
log "=== p2 start: int8 files of the ALBERT, the cheaper static table, one context-gate probe; $(free_gb) GB free on / ==="
mkdir -p "$UPD"

# ---------------------------------------------------------------- 1. ONNX int8 of the ALBERT
log "=== p2 step 1: the repaired int8 export of $ALB next to the package ==="
if [ -f "$OD/mixed-int8-repaired/model.onnx" ]; then
    if [ ! -f "$UPD/onnx/albert-$ALB/mixed-int8/model.onnx" ]; then
        mkdir -p "$UPD/onnx/albert-$ALB" && cp -r "$OD/mixed-int8-repaired" "$UPD/onnx/albert-$ALB/mixed-int8" && cp "$OD/int8_sensitivity.json" "$UPD/onnx/albert-$ALB/int8_sensitivity.json" \
            && log "  $UPD/onnx/albert-$ALB/mixed-int8 <- $OD/mixed-int8-repaired ($(du -sh "$UPD/onnx/albert-$ALB/mixed-int8" | cut -f1)); the package has $(du -sh "$PACK/onnx/albert-$ALB/mixed-int8" | cut -f1)" \
            || log "  copying the repaired variant FAILED"
    fi
    sset onnx_update "onnx/albert-$ALB/mixed-int8"
else log "  $OD/mixed-int8-repaired is missing: nothing to put next to the package"; fi

# ---------------------------------------------------------------- 2. the cheaper table
C1=a1-sparseadam-mw020; C2=$T-mw020
log "=== p2 step 2: the cheaper table ($C1 from p1; $C2 = the released table with the same threshold) ==="
prune "$T" "$C2" --min-weight 0.20
cost p2 "$C2"
if within "$C2"; then ev "$C2"; fi
cand "$C2" "the released table with weights below 0.20 dropped (no training)"
log "  $C2: tiny10 $(t10 "$C2"), $(sget "pq:$C2") postings/query; $C1: tiny10 $(t10 "$C1"), $(sget "pq:$C1"); released $(t10 "$T") at $(sget "pq:$T")"
h8a "$C1"
[ "$(gte "$(t10 "$C2")" "$REL_T10" -0.10)" = 1 ] && h8a "$C2"
if [ ! -s "$P1/cost_h8.json" ]; then
    wait_gpu_free; wait_ram
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 2400 "$PY" scripts/splade_index_cost_asym.py --doc-model "full=$DOC" --query "$T=$SM/$T" --query "$C1=$SM/$C1" --query "$C2=$SM/$C2" \
        --tasks "$H8TASKS" --output-json "$P1/cost_h8.json" > "$LOGS/p2_cost_h8.log" 2>&1 || log "  cost on the heldout8 tasks FAILED (logs/p2_cost_h8.log)"
fi
grep -E "^full|over full|^sample" "$LOGS/p2_cost_h8.log" 2>/dev/null | cut -c1-300 | while read -r l; do log "  heldout8 tasks: $l"; done
CH=""
for n in "$C1" "$C2"; do
    t=$(t10 "$n"); h=$(h8v "$n")
    log "  $n: tiny10 ${t:-not measured} (released $REL_T10), heldout8 ${h:-not measured} with $(h8n "$n") tasks (released $REL_H8), $(sget "pq:$n") postings/query (released $(sget "pq:$T"))"
    [ -n "$t" ] && [ -n "$h" ] && [ "$(h8n "$n")" = 8 ] && within "$n" || continue
    [ "$(gte "$t" "$REL_T10" -0.10)" = 1 ] && [ "$(gte "$h" "$REL_H8" -0.10)" = 1 ] && [ "$(gte "$(sget "pq:$T")" "$(sget "pq:$n")" 500)" = 1 ] || continue
    if [ -z "$CH" ] || [ "$(gte "$(sget "pq:$CH")" "$(sget "pq:$n")" 1)" = 1 ]; then CH=$n; fi
done
if [ -n "$CH" ]; then
    sset cheaper "$CH"
    log "  THE SAME QUALITY, CHEAPER: $CH (tiny10 $(t10 "$CH") / heldout8 $(h8v "$CH") against $REL_T10 / $REL_H8; $(sget "pq:$CH") postings/query against $(sget "pq:$T")). Files for the swap follow; the swap is the user's decision"
    STD="models/st/$FULL-static-$CH"
    if [ ! -f "$STD/modules.json" ]; then
        wait_gpu_free; wait_ram
        ORPHAN_GUARD_EXEMPT=1 timeout -k 60 1800 "$PY" -m release.st_export --doc-model "$DOC" --table "$SM/$CH" --out "$STD" --query-only-out "$STD-query" > "$LOGS/p2_st_export.log" 2>&1 \
            && log "  Sentence Transformers package written: $STD (and $STD-query): $(tail -n 1 "$LOGS/p2_st_export.log" | cut -c1-300)" || log "  st_export FAILED (logs/p2_st_export.log)"
    fi
    if [ -f "$STD/modules.json" ]; then
        ORPHAN_GUARD_EXEMPT=1 timeout -k 60 900 "$PY" - "$STD" > "$LOGS/p2_st_check.log" 2>&1 <<'PYEOF' && log "  package check: $(tail -n 1 "$LOGS/p2_st_check.log" | cut -c1-300)" || log "  package check FAILED (logs/p2_st_check.log)"
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
        mkdir -p "$UPD/static"
        for pair in "$FULL-static-$CH=$SM/$CH" "st-$FULL-static-$CH=$STD" "st-$FULL-static-$CH-query=$STD-query"; do
            dest="$UPD/static/${pair%%=*}"; src="${pair#*=}"
            [ -e "$dest" ] || { [ -e "$src" ] && cp -r "$src" "$dest" && log "  $dest <- $src ($(du -sh "$dest" | cut -f1))" || log "  copying $src FAILED"; }
        done
    fi
else
    log "  no table has the quality of the released one at a clearly lower cost by the rule above: nothing to swap"
fi
table

# ---------------------------------------------------------------- the update directory: README, CONTENTS, sums
{
    echo "# Update 1 of the release package 2026-11-13 (prepared $(date '+%d.%m.%Y %H:%M') on the 3060 box)"; echo
    echo "New versions of single paths of \`$PACK\`, in the same layout. Nothing in the package was changed; applying a path is the user's decision (copy the directory over the package path, then rebuild \`SHA256SUMS\` and the line of \`CONTENTS.tsv\`). Check the copy first: \`sha256sum -c SHA256SUMS\`."; echo
    if [ -f "$UPD/onnx/albert-$ALB/mixed-int8/model.onnx" ]; then
        echo "## \`onnx/albert-$ALB/mixed-int8\`: the int8 export of the ALBERT, repaired"; echo
        echo "The released mixed int8 export (dynamic int8, u8 weights, MLM head fp32) is broken: cosine 0.7499 to the fp32 export, 4.7 times the active dimensions, tiny10 with int8 queries 62.10 against 64.34 in torch. Cause (\`int8_sensitivity.json\` next to the export): the layer added on top of the four shared ALBERT groups; alone in int8 it gives cosine 0.73, every other group alone at least 0.994. In this version that layer (4 nodes) and the FFN output of group 2 (1 node; its input range leaves 97 % of the values below one quantization step) stay fp32 next to the MLM head."; echo
        echo "| | int8 nodes | cosine to fp32 | active Jaccard | dims ratio | size | tiny10 with int8 queries |"; echo "|---|---:|---:|---:|---:|---:|---:|"
        echo "| released | 29 | 0.7499 | 0.2103 | 4.732 | $(du -sh "$PACK/onnx/albert-$ALB/mixed-int8" | cut -f1) | 62.10 |"
        echo "| this version | 24 | 0.9980 | 0.9197 | 1.000 | $(du -sh "$UPD/onnx/albert-$ALB/mixed-int8" | cut -f1) | $(regime_mean "$MAIN/results-regimes/quantized/onnx-albert-$ALB-q-mixed-int8-repaired/tiny10.json" | awk '{print $1}') |"
        echo; echo "torch: 64.34. The fp32 export of the package is exact and unchanged. heldout8 with int8 queries was not measured for either version."; echo
    fi
    if [ -n "$CH" ] && [ -e "$UPD/static/$FULL-static-$CH" ]; then
        echo "## \`static/\`: a cheaper static query table of the same quality"; echo
        echo "\`$CH\` in place of \`$T\` (the same format static-sparse-query/1, the same rows and k = 8): $(sget "note:$CH"). Three paths as in the package: the table, the Sentence Transformers model with the table as the query route (load with \`trust_remote_code=True\`), and the query side alone."; echo
        echo "| table | tiny10 | heldout8 | dims/query | postings/query (cost sample) | postings/query on the heldout8 tasks |"; echo "|---|---:|---:|---:|---:|---:|"
        for n in "$T" "$CH"; do
            echo "| \`$n\`$([ "$n" = "$T" ] && echo " (released)") | $(t10 "$n") | $(h8v "$n") | $(sget "dq:$n") | $(sget "pq:$n") | $("$PY" -c "import json,sys;print('%.0f' % json.load(open(sys.argv[1]))['pairs'][sys.argv[2]]['postings_per_query'])" "$P1/cost_h8.json" "$n" 2>/dev/null) |"
        done
        echo; echo "Seed noise on tiny10 is 0.13-0.2, so the two tables are equal in quality; the gain is the cost of a query. Query encoding time is the same (a table lookup). When this table is taken, the static-table benchmark on the 5070 has to be run again for it (by 20.11)."; echo
    fi
    echo "## Contents"; echo; echo "| path | size |"; echo "|---|---:|"
    ( cd "$UPD" && for d in onnx/*/* static/*; do [ -e "$d" ] && echo "| \`$d\` | $(du -sh "$d" | cut -f1) |"; done )
} > "$UPD/README.md"
( cd "$UPD" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS )
log "  $UPD: $(du -sh "$UPD" | cut -f1), $(grep -c . "$UPD/SHA256SUMS") files in SHA256SUMS, README.md written"

# ---------------------------------------------------------------- 3. the context gate (B3), one probe
G=gate32; GL="asym-$FULL-q-static2-$G-kd"
log "=== p2 step 3: context gate d = 32 on the released table (format 2), one probe ==="
if [ $((SECONDS - START)) -gt 10800 ]; then log "  more than 3 h in p2: the probe is skipped"
else
    mkdir -p "$SM2"
    [ -d "$SM2/$G" ] || { wait_ram; timeout -k 60 900 "$PY" -m distill.static_v2 convert --model "$SM/$T" --ctx-dim 32 --out "$SM2/$G" > "$LOGS/p2_b3_convert.log" 2>&1 || log "  convert FAILED (logs/p2_b3_convert.log)"; }
    if [ -d "$SM2/$G" ] && [ ! -d "runs/static2-$FULL-$G-vec/final" ]; then
        wait_gpu_free; wait_ram; s=$SECONDS; log "  fit of the gate to the teacher's query vectors (400 000 kd_v1 queries, 2 epochs)"
        ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" -m distill.train_static_vec --teacher "$DOC" --init "$SM2/$G" --texts datasets/kd/kd_v1.jsonl --kinds query --n 400000 \
            --train extra --extra-lr 1e-2 --epochs 2 --out "runs/static2-$FULL-$G-vec" > "$LOGS/p2_b3_vec.log" 2>&1
        log "  vector stage: exit $? in $((SECONDS - s))s: $(tail -n 1 "$LOGS/p2_b3_vec.log" | cut -c1-300)"; record "p2:b3_vec" "$([ -d "runs/static2-$FULL-$G-vec/final" ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    fi
    if [ -d "runs/static2-$FULL-$G-vec/final" ] && [ ! -d "$SM2/$G-kd" ]; then
        wait_gpu_free; wait_ram; s=$SECONDS; log "  ranking training of the gated encoder on pool r2 (3 epochs)"
        ORPHAN_GUARD_EXEMPT=1 timeout -k 60 2400 "$PY" -m distill.train_static --pool "$POOL2" --init "runs/static2-$FULL-$G-vec/final" --out "runs/static2-$FULL-$G-kd" --lr 5e-3 --extra-lr 5e-3 --epochs 3 \
            "${TRAIN[@]}" > "$LOGS/p2_b3_kd.log" 2>&1
        if [ -d "runs/static2-$FULL-$G-kd/final" ]; then
            cp -r "runs/static2-$FULL-$G-kd/final" "$SM2/$G-kd"; cp "runs/static2-$FULL-$G-kd/static_summary.json" "$RS/static2-$G-kd.json" 2>/dev/null
            log "  ranking stage done in $((SECONDS - s))s: val top1 $(sval "static2-$G-kd" top1_agreement val_init) -> $(sval "static2-$G-kd" top1_agreement) (the released table 0.6850, oracle weights 0.7905), dims/query $(sval "static2-$G-kd" dims_per_query)"
        else log "  ranking stage FAILED after $((SECONDS - s))s (logs/p2_b3_kd.log): $(tail -n 1 "$LOGS/p2_b3_kd.log" | cut -c1-300)"; fi
        record "p2:b3_kd" "$([ -d "$SM2/$G-kd" ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    fi
    if [ -d "$SM2/$G-kd" ]; then
        PIRB_QUERY_MODEL="$SM2/$G-kd" regime quantized "$GL" "$DOC" tiny10 16
        if [ ! -s "$P1/cost_b3.json" ]; then
            wait_gpu_free; wait_ram
            ORPHAN_GUARD_EXEMPT=1 timeout -k 60 2400 "$PY" scripts/splade_index_cost_asym.py --doc-model "full=$DOC" --query "$G-kd=$SM2/$G-kd" --output-json "$P1/cost_b3.json" > "$LOGS/p2_cost_b3.log" 2>&1 \
                || log "  cost of the gated encoder FAILED (logs/p2_cost_b3.log)"
        fi
        log "  CONTEXT GATE d = 32: tiny10 $(lbv "$GL" tiny10) (released table $REL_T10, oracle weights $(lbv "$(olab weights)" tiny10), full model 65.06); $(grep "over full" "$LOGS/p2_cost_b3.log" 2>/dev/null | tail -n 1 | cut -c1-200); folder $(du -sh "$SM2/$G-kd" | cut -f1) against $(du -sh "$SM/$T" | cut -f1)"
    else log "  no gated encoder: the probe gave no result"; fi
fi
log "=== p2 done ($(date '+%d.%m %H:%M'), $(( (SECONDS - START) / 60 )) min) ==="
