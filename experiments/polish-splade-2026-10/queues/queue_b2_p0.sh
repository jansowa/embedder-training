#!/usr/bin/env bash
# Queue b2, stage p0 (CPU only, about 10 min): the release package with the swaps the user approved on 16.11, as a NEW
# version datasets/transfer/release-20261117 next to release-20261113 (which is not changed; nothing is deleted there):
#   1  the int8 export of the ALBERT: the repaired variant (MLM head fp32) replaces the broken mixed-int8; the repaired
#      variant with the MLM head in int8 (mixed-int8-int8head) is added
#   2  the static query table phr2-k8-kd-r2-mw020 replaces phr2-k8-kd-r2 (the table, the Sentence Transformers model, the
#      query side alone); logs/final_static_table.txt follows
#   3  the full model with every node in int8 (dynamic-int8) as a third ONNX variant
#   4  SHA256SUMS without entries of SHA256SUMS* (the package of 13.11 lists ./SHA256SUMS.tmp)
# README with the list of changes (old / new numbers), CONTENTS.tsv, and a tar with only the difference for the 5070
# (its size is logged before it is written). Sources: datasets/transfer/release-20261113-update1 (queues p2 and x2).
# Run by queue_b2.sh. B2_TEST=1 with OLD / UPD / TDIR / NEW pointing at a small copy: no label file, no summary, no commit.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=${B2_TAG:-b2p0} MIN_RAM_GB=4
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
source "$EXP_ROOT/queues/b2_common.sh"
guard_alive
TEST=${B2_TEST:-0}
OLD=${OLD:-datasets/transfer/release-20261113}; UPD=${UPD:-datasets/transfer/release-20261113-update1}; NAME=release-20261117
OLDREAL=$(readlink -f "$OLD"); TDIR=${TDIR:-$(dirname "$OLDREAL")}
NEWREAL="$TDIR/$NAME"; NEW=${NEW:-datasets/transfer/$NAME}; BUILD="$NEWREAL.building"
TARNAME="$NAME-diff-from-20261113.tar"; TAR="$TDIR/$TARNAME"
AT="albert-$ALB"; FT="full-$FULL"; T=phr2-k8-kd-r2; TM=$T-mw020
OLD_ST=("$FULL-static-$T" "st-$FULL-static-$T" "st-$FULL-static-$T-query"); NEW_ST=("$FULL-static-$TM" "st-$FULL-static-$TM" "st-$FULL-static-$TM-query")
log "=== b2 p0 start: $NEW next to $OLD (sources: $UPD); $(df -BG --output=avail "$TDIR" | tail -1 | tr -dc '0-9') GB free under $TDIR ==="

num() {  # num JSON PATH...: a number of a JSON file
    "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]))
for k in sys.argv[2:]: d=d[k]
print('%.4f' % d if isinstance(d, float) else d)" "$@" 2>/dev/null
}
slbv() { regime_mean "$SEXP/results-regimes/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
sums() { ( cd "$1" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS ); }
note() {  # note DEST SRC NOTE: the line of DEST in CONTENTS.tsv of the build
    grep -v -P "^\Q$1\E\t" "$BUILD/CONTENTS.tsv" > "$BUILD/CONTENTS.tsv.tmp"; mv "$BUILD/CONTENTS.tsv.tmp" "$BUILD/CONTENTS.tsv"
    [ -n "${2:-}" ] && printf '%s\t%s\t%s\t%s\n' "$1" "$(du -sh "$BUILD/$1" | cut -f1)" "$2" "$3" >> "$BUILD/CONTENTS.tsv"
    return 0
}
stop() { log "  $*: p0 stops, nothing of the old package was changed"; exit 1; }

# ---------------------------------------------------------------- 0. the sources
[ -d "$OLDREAL" ] && [ -s "$OLDREAL/SHA256SUMS" ] && [ -s "$OLDREAL/CONTENTS.tsv" ] || stop "$OLD is not a package"
[ -n "$ALB" ] && [ -n "$FULL" ] || stop "no frozen labels"
for p in "onnx/$AT/mixed-int8/model.onnx" "onnx/$AT/mixed-int8-int8head/model.onnx" "onnx/$AT/mixed-int8-int8head.json" "onnx/$AT/int8_sensitivity.json" \
         "onnx/$FT/dynamic-int8/model.onnx" "onnx/$FT/dynamic-int8.json" "static/${NEW_ST[0]}/static_query.safetensors" "static/${NEW_ST[1]}/modules.json" "static/${NEW_ST[2]}/modules.json"; do
    [ -s "$UPD/$p" ] || stop "$UPD/$p is missing"
done
for p in "onnx/$AT/mixed-int8" "onnx/$AT/fp32" "onnx/$FT/mixed-int8" "onnx/$FT/fp32" "static/${OLD_ST[0]}" "static/${OLD_ST[1]}" "static/${OLD_ST[2]}" "full/$FULL" "albert/$ALB"; do
    [ -d "$OLDREAL/$p" ] || stop "$OLD/$p is missing"
done
[ "$(df -BG --output=avail "$TDIR" | tail -1 | tr -dc '0-9')" -ge 8 ] || stop "less than 8 GB free under $TDIR"
if [ ! -s "$NEWREAL/SHA256SUMS" ]; then
    wait_ram
    ( cd "$OLDREAL" && grep -v ' \./SHA256SUMS' SHA256SUMS | sha256sum -c --quiet ) > "$LOGS/b2_p0_check_old.log" 2>&1 || stop "the old package fails its checksums (logs/b2_p0_check_old.log)"
    ( cd "$UPD" && grep -v ' \./SHA256SUMS' SHA256SUMS | sha256sum -c --quiet ) > "$LOGS/b2_p0_check_upd.log" 2>&1 || stop "$UPD fails its checksums (logs/b2_p0_check_upd.log)"
    log "  checksums of $OLD ($(grep -c -v ' \./SHA256SUMS' "$OLDREAL/SHA256SUMS") files, without its stray line ./SHA256SUMS.tmp) and of $UPD pass"
fi

# ---------------------------------------------------------------- 1. the new version
if [ -s "$NEWREAL/SHA256SUMS" ]; then log "  $NEWREAL is present: not built again"
else
    [ -e "$NEWREAL" ] && stop "$NEWREAL exists without SHA256SUMS (an unfinished or foreign directory)"
    [ -e "$BUILD" ] && { mv "$BUILD" "$BUILD.aside-$(date +%H%M%S)"; log "  an unfinished build was moved aside ($BUILD.aside-*), not deleted"; }
    s=$SECONDS
    mkdir -p "$BUILD" && cp -a "$OLDREAL/." "$BUILD/" || stop "copying $OLD failed"
    case "$BUILD" in *"/$NAME.building") ;; *) stop "unexpected build path $BUILD";; esac
    # the replaced paths are removed from the fresh copy only
    rm -f "$BUILD/SHA256SUMS"
    rm -rf "$BUILD/onnx/$AT/mixed-int8"; for d in "${OLD_ST[@]}"; do rm -rf "$BUILD/static/$d"; done
    cp -a "$UPD/onnx/$AT/mixed-int8" "$UPD/onnx/$AT/mixed-int8-int8head" "$UPD/onnx/$AT/mixed-int8-int8head.json" "$UPD/onnx/$AT/int8_sensitivity.json" "$BUILD/onnx/$AT/" \
        && cp -a "$UPD/onnx/$FT/dynamic-int8" "$UPD/onnx/$FT/dynamic-int8.json" "$BUILD/onnx/$FT/" || stop "copying the ONNX variants failed"
    for d in "${NEW_ST[@]}"; do cp -a "$UPD/static/$d" "$BUILD/static/$d" || stop "copying static/$d failed"; done
    for f in "$RESULTS_ROOT/summary_x2.md" "$SEXP/results/summary_p1.md" "$RESULTS_ROOT/onnx/int8_full_$AT.json" "$RESULTS_ROOT/onnx/int8_full_$FT.json"; do [ -e "$f" ] && cp "$f" "$BUILD/results/"; done
    # the export reports follow the variants that are in the package now
    "$PY" - "$BUILD" "$AT" "$FT" <<'PYEOF' || stop "updating the export reports failed"
import json, sys
from pathlib import Path
build, at, ft = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def variant(tag, name, note):
    d = build / "onnx" / tag / name
    meta = json.loads((d / "onnx_splade.json").read_text())
    return {"dir": f"onnx/{tag}/{name}", "bytes": (d / "model.onnx").stat().st_size, "quantization": meta.get("quantization"), "fp32_check": meta.get("fp32_check"), "note": note}
p = build / "onnx" / at / "export_report.json"; r = json.loads(p.read_text())
old = r["variants"]["mixed"]
r["variants"]["mixed_of_release_20261113"] = {"bytes": old.get("bytes"), "fp32_check": old.get("fp32_check"), "note": "the broken export of release-20261113, replaced; not in this package"}
r["variants"]["mixed"] = variant(at, "mixed-int8", "repaired on 16.11 by release.int8_sensitivity: the extra head layer (4 nodes) and the FFN output of group 2 stay fp32 next to the MLM head; report int8_sensitivity.json")
r["variants"]["mixed-int8-int8head"] = variant(at, "mixed-int8-int8head", "added on 16.11: the repaired variant with the MLM head in int8 as well; report mixed-int8-int8head.json")
p.write_text(json.dumps(r, indent=2) + "\n"); (build / "results" / f"export_{at}.json").write_text(json.dumps(r, indent=2) + "\n")
p = build / "onnx" / ft / "export_report.json"; r = json.loads(p.read_text())
r["variants"]["dynamic-int8"] = variant(ft, "dynamic-int8", "added on 16.11: dynamic int8 of every MatMul / Gemm, the MLM head included; report dynamic-int8.json")
p.write_text(json.dumps(r, indent=2) + "\n"); (build / "results" / f"export_{ft}.json").write_text(json.dumps(r, indent=2) + "\n")
PYEOF
    A_OLD=$(lbv "onnx-$AT-q-mixed-int8" tiny10); A_REP=$(lbv "onnx-$AT-q-mixed-int8-repaired" tiny10); A_HEAD=$(lbv "onnx-$AT-q-mixed-int8-int8head" tiny10); A_T=$(slbv "$ALB" tiny10)
    F_MIX=$(lbv "onnx-$FT-q-mixed-int8" tiny10); F_DYN=$(lbv "onnx-$FT-q-dynamic-int8" tiny10); F_T=$(lbv "$FULL" tiny10)
    A_OLD_COS=$(num "$OLDREAL/onnx/$AT/export_report.json" variants mixed fp32_check cosine mean); A_REP_COS=$(num "$BUILD/onnx/$AT/export_report.json" variants mixed fp32_check cosine mean)
    A_HEAD_COS=$(num "$BUILD/onnx/$AT/mixed-int8-int8head.json" fp32_check cosine mean); F_MIX_COS=$(num "$BUILD/onnx/$FT/export_report.json" variants mixed fp32_check cosine mean); F_DYN_COS=$(num "$BUILD/onnx/$FT/dynamic-int8.json" fp32_check cosine mean)
    mb() { echo "$(( $(stat -c %s "$1") / 1048576 )) MB"; }
    for d in "${OLD_ST[@]}"; do note "static/$d"; done
    note "static/${NEW_ST[0]}" "models/static-$FULL/$TM" "static query table $TM over the index of $FULL (static-sparse-query/1): the table $T of release-20261113 with weights below 0.20 dropped; asymmetric 60.67 / 39.20, 75.9 dims/query, 9651 postings/query on the cost sample"
    note "static/${NEW_ST[1]}" "models/st/$FULL-static-$TM" "Sentence Transformers SparseEncoder: query route = the table, document route = $FULL (load with trust_remote_code=True)"
    note "static/${NEW_ST[2]}" "models/st/$FULL-static-$TM-query" "the query side alone as a Sentence Transformers model (no document encoder)"
    note "onnx/$AT" "models/$ALB" "ONNX exports of $ALB: fp32; mixed-int8 = dynamic int8 (u8 weights) with the MLM head, the extra head layer and the FFN output of group 2 in fp32 (cosine to fp32 $A_REP_COS, tiny10 with int8 queries $A_REP); mixed-int8-int8head = the same with the MLM head in int8 (cosine $A_HEAD_COS, tiny10 $A_HEAD)"
    note "onnx/$FT" "models/$FULL" "ONNX exports of $FULL: fp32; mixed-int8 = dynamic int8 (u8 weights), MLM head fp32 (cosine to fp32 $F_MIX_COS, tiny10 with int8 queries $F_MIX); dynamic-int8 = every node int8, the MLM head included (cosine $F_DYN_COS, tiny10 $F_DYN)"
    note "results" "$RESULTS_ROOT, $SEXP/results" "summaries and choices of v62 / r1 / a1 / st1, the ONNX export reports, the int8 repair and the cheaper table (summary_p1.md, summary_x2.md)"
    # files of the old package that are not in the new one (for the difference tar)
    ( cd "$BUILD" && find . -type f ! -name 'SHA256SUMS*' ! -name REMOVED-from-20261113.txt | sort ) > "$LOGS/b2_p0_new_files.txt"
    grep -v ' \./SHA256SUMS' "$OLDREAL/SHA256SUMS" | sed 's/^[0-9a-f]*  //' | sort > "$LOGS/b2_p0_old_files.txt"
    comm -23 "$LOGS/b2_p0_old_files.txt" "$LOGS/b2_p0_new_files.txt" > "$BUILD/REMOVED-from-20261113.txt"
    {
        echo "# Release package 2026-11-17 (from the 3060 box)"; echo
        echo "Models in the form they are released in (the bias is baked into the checkpoints), their ONNX exports and the inputs of the benchmarks. It replaces \`release-20261113\`, which is kept unchanged next to it. Check the copy first:"; echo
        echo '```bash'; echo "cd datasets/transfer/$NAME && sha256sum -c SHA256SUMS"; echo '```'; echo
        echo "## Changes against release-20261113"; echo
        echo "| path | release-20261113 | this version |"; echo "|---|---|---|"
        echo "| \`onnx/$AT/mixed-int8\` | broken int8 export of the ALBERT: cosine to fp32 $A_OLD_COS, tiny10 with int8 queries $A_OLD (torch $A_T), $(mb "$OLDREAL/onnx/$AT/mixed-int8/model.onnx") | **replaced** by the repaired export (the extra head layer and the FFN output of group 2 stay fp32 next to the MLM head): cosine $A_REP_COS, tiny10 $A_REP, $(mb "$BUILD/onnx/$AT/mixed-int8/model.onnx"); report \`onnx/$AT/int8_sensitivity.json\` |"
        echo "| \`onnx/$AT/mixed-int8-int8head\` | - | **added**: the repaired export with the MLM head in int8 as well: cosine $A_HEAD_COS, tiny10 $A_HEAD, $(mb "$BUILD/onnx/$AT/mixed-int8-int8head/model.onnx") |"
        echo "| \`onnx/$FT/dynamic-int8\` | - | **added**: the full model with every node in int8, the MLM head included: cosine $F_DYN_COS, tiny10 $F_DYN, $(mb "$BUILD/onnx/$FT/dynamic-int8/model.onnx") (mixed-int8 with the head in fp32, unchanged: cosine $F_MIX_COS, tiny10 $F_MIX, $(mb "$BUILD/onnx/$FT/mixed-int8/model.onnx"); torch $F_T) |"
        echo "| \`static/\` (the table, the Sentence Transformers model, the query side alone) | table \`$T\`: asymmetric tiny10 / heldout8 60.74 / 39.24, 89.7 dims/query, 11764 postings/query | **replaced** by table \`$TM\` (the same table with weights below 0.20 dropped, no training): 60.67 / 39.20, 75.9 dims/query, 9651 postings/query (-18 %) |"
        echo "| \`SHA256SUMS\` | lists a stray \`./SHA256SUMS.tmp\`, so \`sha256sum -c\` reports one unreadable file | no entries of \`SHA256SUMS*\`; \`sha256sum -c\` passes |"
        echo
        echo "The full model, the ALBERT, both fp32 exports and the mixed-int8 export of the full model are the files of release-20261113, byte for byte. Seed noise on tiny10 is 0.13-0.2, so the two static tables are equal in quality; the gain is the cost of a query. heldout8 with int8 queries was not measured for any variant; CPU timing of the new int8 variants was not run. **The static-table benchmark on the 5070 has to be run again for \`$TM\`.** Details: \`results/summary_p1.md\` (the int8 fault, the cheaper table), \`results/summary_x2.md\` (the MLM head in int8). Files of release-20261113 that are not in this version: \`REMOVED-from-20261113.txt\`."; echo
        echo "### Applying the difference on the 5070"; echo
        echo "\`$TARNAME\` holds only the new and the changed files (and this README, CONTENTS.tsv, SHA256SUMS). Next to an intact \`release-20261113\`:"; echo
        echo '```bash'
        echo "cp -a release-20261113 $NAME && tar -xf $TARNAME"
        echo "cd $NAME && xargs -d '\\n' rm -f < REMOVED-from-20261113.txt && rm -f SHA256SUMS.tmp && find . -type d -empty -delete && sha256sum -c --quiet SHA256SUMS"
        echo '```'; echo
        echo "## Contents"; echo; echo "| path | size | source | what |"; echo "|---|---:|---|---|"
        sort "$BUILD/CONTENTS.tsv" | awk -F'\t' '{printf "| `%s` | %s | `%s` | %s |\n", $1, $2, $3, $4}'
        echo; echo "## Models"; echo
        echo "- Full model (82 M): \`full/$FULL\` — quantized regime tiny10 / heldout8 $(lbs "$FULL"), trec-covid $(tc "$FULL"), $(dims "$FULL") dims/doc. Choice and table: \`results/summary_v62.md\`."
        echo "- ALBERT student (50 M): \`albert/$ALB\`. Choice and table: \`results/summary_a1.md\` (the earlier lineage from soup-v9b: \`results/summary_r1.md\`)."
        echo "- Static query table: \`models/static-$FULL/$TM\` on the 3060; here under \`static/\` (the table, the Sentence Transformers model with the table as the query route, and the query side alone). Load the Sentence Transformers model with \`SparseEncoder(path, trust_remote_code=True)\`. Numbers: \`results/summary_st1.md\` (the table before the cut), \`results/summary_p1.md\` (the cut)."
        echo; echo "## Publication benchmark (PIRB) on the 5070"; echo
        echo "From the repository root, with P=datasets/transfer/$NAME. One task set per call; the task sets are \`configs/pirb_tasks/*.json\` (copied to \`bench/pirb_tasks/\`): tiny10, heldout8, heldout2_czywiesz, heldout2_gpt35wiki, heldout2_quora, heldout2_poquad, heldout2_treccovid are the 23 tasks measured on the 3060; the remaining files are the large PIRB tasks. Batch 8 for poquad and trec-covid."; echo
        echo '```bash'
        echo "P=datasets/transfer/$NAME"
        echo "# symmetric, quantized regime (the regime of every number in the summaries); old-pyserini = the 2023 leaderboard regime"
        echo "scripts/pirb_regime_eval.sh quantized $FULL \$P/full/$FULL tiny10 16"
        echo "scripts/pirb_regime_eval.sh quantized $ALB \$P/albert/$ALB tiny10 16"
        echo "# asymmetric: the index of the full model, another query encoder"
        echo "PIRB_QUERY_MODEL=\$P/albert/$ALB scripts/pirb_regime_eval.sh quantized asym-$FULL-q-$ALB \$P/full/$FULL tiny10 16"
        echo "PIRB_QUERY_MODEL=\$P/static/${NEW_ST[0]} scripts/pirb_regime_eval.sh quantized asym-$FULL-q-static-mw020 \$P/full/$FULL tiny10 16"
        echo "PIRB_QUERY_MODEL=\$P/onnx/$FT/mixed-int8 scripts/pirb_regime_eval.sh quantized onnx-$FT-q-mixed-int8 \$P/full/$FULL tiny10 16"
        echo "PIRB_QUERY_MODEL=\$P/onnx/$FT/dynamic-int8 scripts/pirb_regime_eval.sh quantized onnx-$FT-q-dynamic-int8 \$P/full/$FULL tiny10 16"
        echo "PIRB_QUERY_MODEL=\$P/onnx/$AT/mixed-int8 scripts/pirb_regime_eval.sh quantized onnx-$AT-q-mixed-int8 \$P/albert/$ALB tiny10 16"
        echo "PIRB_QUERY_MODEL=\$P/onnx/$AT/mixed-int8-int8head scripts/pirb_regime_eval.sh quantized onnx-$AT-q-mixed-int8-int8head \$P/albert/$ALB tiny10 16"
        echo '```'
        echo; echo "## CPU benchmark (not run on the 3060: no CPU timing in the queues)"; echo
        echo "Start by hand on an idle machine; it times queries (batch 1) and passages (batch 8) at 1 and 6 threads and compares each variant with the fp32 export:"; echo
        echo '```bash'
        echo ".venv/bin/python -m release.onnx_bench --variants \$P/onnx/$FT/fp32 \$P/onnx/$FT/mixed-int8 \$P/onnx/$FT/dynamic-int8 --torch-model \$P/full/$FULL \\"
        echo "    --texts \$P/bench/texts_kd_v1_sample.jsonl --threads 1,6 --out experiments/polish-splade-2026-10/results/onnx/bench_$FT.json"
        echo ".venv/bin/python -m release.onnx_bench --variants \$P/onnx/$AT/fp32 \$P/onnx/$AT/mixed-int8 \$P/onnx/$AT/mixed-int8-int8head --torch-model \$P/albert/$ALB \\"
        echo "    --texts \$P/bench/texts_kd_v1_sample.jsonl --threads 1,6 --out experiments/polish-splade-2026-10/results/onnx/bench_$AT.json"
        echo '```'
        echo; echo "ONNX fidelity (export reports in \`results/export_*.json\` and the \`.json\` next to each added variant): the fp32 export is checked against torch on 16 texts, every int8 export against fp32 on 128 texts (cosine, active Jaccard, top-10 terms, dims ratio)."
    } > "$BUILD/README.md"
    sums "$BUILD"
    ( cd "$BUILD" && sha256sum -c --quiet SHA256SUMS ) > "$LOGS/b2_p0_check_new.log" 2>&1 || stop "the new package fails its own checksums (logs/b2_p0_check_new.log); the build stays in $BUILD"
    grep -q 'SHA256SUMS' "$BUILD/SHA256SUMS" && stop "SHA256SUMS of the build lists a SHA256SUMS file"
    mv "$BUILD" "$NEWREAL" || stop "renaming the build failed"
    log "  $NEWREAL built in $((SECONDS - s))s: $(du -sh "$NEWREAL" | cut -f1), $(grep -c . "$NEWREAL/SHA256SUMS") files, sha256sum -c passes; $(grep -c . "$NEWREAL/REMOVED-from-20261113.txt") files of the old package are not in it"
fi
[ -e "$NEW" ] || ln -s "$NEWREAL" "$NEW"

# ---------------------------------------------------------------- 2. the label file of the static table
if [ "$TEST" != 1 ]; then
    if [ "$(first_line "$FINAL_STATIC_FILE")" != "models/static-$FULL/$TM" ] && [ -d "models/static-$FULL/$TM" ] && [ -d "models/st/$FULL-static-$TM" ]; then
        cp "$FINAL_STATIC_FILE" "$FINAL_STATIC_FILE.before-20261117"
        { echo "models/static-$FULL/$TM"; echo "# first line = the static query table to release (over the index of models/$FULL); set by queue b2 on $(date '+%d.%m %H:%M') on the user's decision of 16.11: the table $T with weights below 0.20 dropped (the earlier file: $(basename "$FINAL_STATIC_FILE").before-20261117)"; echo "models/st/$FULL-static-$TM"; } > "$FINAL_STATIC_FILE"
        log "  $FINAL_STATIC_FILE -> models/static-$FULL/$TM"
    fi
fi

# ---------------------------------------------------------------- 3. the difference for the 5070
LIST="$LOGS/b2_p0_diff_files.txt"
"$PY" - "$OLDREAL/SHA256SUMS" "$NEWREAL/SHA256SUMS" "$NAME" > "$LIST" <<'PYEOF'
import sys
def read(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        h, _, name = line.rstrip("\n").partition("  ")
        if name and "SHA256SUMS" not in name:
            out[name] = h
    return out
old, new = read(sys.argv[1]), read(sys.argv[2])
for name in sorted(new):
    if old.get(name) != new[name]:
        print(sys.argv[3] + "/" + name[2:])
print(sys.argv[3] + "/SHA256SUMS")
PYEOF
DBYTES=$(cd "$TDIR" && tr '\n' '\0' < "$LIST" | xargs -0 stat -c %s | awk '{s += $1} END {printf "%d", s}')
FREE=$(df -BM --output=avail "$TDIR" | tail -1 | tr -dc '0-9')
log "  difference for the 5070: $(grep -c . "$LIST") files, $(( DBYTES / 1048576 )) MB; the tar will be $TAR; $FREE MB free under $TDIR"
if [ -s "$TAR" ]; then log "  $TAR is present: not written again"
elif [ "$FREE" -lt $(( 3 * DBYTES / 1048576 + 1024 )) ]; then log "  not enough space for the tar: NOT written"
else
    tar -cf "$TAR.tmp" -C "$TDIR" -T "$LIST" && mv "$TAR.tmp" "$TAR" && ( cd "$TDIR" && sha256sum "$TARNAME" > "$TARNAME.sha256" ) \
        && log "  $TAR written: $(du -sh "$TAR" | cut -f1), $(tar -tf "$TAR" | grep -c .) entries; checksum in $TARNAME.sha256" || log "  writing the tar FAILED"
fi
[ "$TEST" != 1 ] && [ -s "$TAR" ] && { [ -e "datasets/transfer/$TARNAME" ] || ln -s "$TAR" "datasets/transfer/$TARNAME"; [ -e "datasets/transfer/$TARNAME.sha256" ] || ln -s "$TAR.sha256" "datasets/transfer/$TARNAME.sha256"; }

# ---------------------------------------------------------------- 4. notes, summary, commit
[ "$TEST" = 1 ] && { log "=== b2 p0 done (test mode: no label file, no summary, no commit) ==="; exit 0; }
{
    echo "## p0: the release package of 17.11"; echo
    echo "\`datasets/transfer/$NAME\` ($(du -sh "$NEWREAL" | cut -f1), $(grep -c . "$NEWREAL/SHA256SUMS") files; \`sha256sum -c\` passes) next to \`release-20261113\`, which was not changed. The swaps the user approved on 16.11:"; echo
    sed -n '/^## Changes against release-20261113/,/^### Applying/p' "$NEWREAL/README.md" | grep '^|'
    echo; echo "\`logs/final_static_table.txt\` names \`$(first_line "$FINAL_STATIC_FILE")\`. Difference for the 5070: \`datasets/transfer/$TARNAME\` ($([ -s "$TAR" ] && du -sh "$TAR" | cut -f1 || echo "not written"), $(grep -c . "$LIST") files; how to apply it is in the README of the package). The static-table benchmark on the 5070 has to be run again for the new table."
} > "$B2/notes_p0.md"
base_cands
write_summary "Add the b2 results: the release package of 17.11 with the approved swaps"
log "=== b2 p0 done ($(date '+%d.%m %H:%M')); package $NEW ==="
