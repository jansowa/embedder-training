#!/usr/bin/env bash
# Queue b2, stage p1b (about 1.1 h): the true points of the dose curve of R-c' stage 1.
# Queue b1c wrote the "bias forms" of every stage-1 model into the raw model itself (b2_common.sh): each
# arm-b1-rc-s1pK carries the three shifts (e^0.05 - 1) + (e^0.10 - 1) + (e^0.15 - 1) = 0.3183 on its MLM head bias,
# which is the threshold 0.276, not 0.15. So the dose curve of summary_b1.md (64.60 / 64.50 / 64.65 / 64.62 and the
# end of stage 1 64.62 / 43.00 / 67.33 at 315.6 dims/doc) was read in a far sparser form than "the first form within
# the limits", segments 2-4 each started from a model shifted that way, and so did the dose3 round arm-b1-rc-d3.
# Here the shift is added back on real copies (arm-b1-rc-s1p1-unshifted, arm-b1-rc-s1p4-unshifted; the copy must
# reproduce the density b1c measured before its first shift: 793.5 and 704.9 dims/doc), then:
#   segment 1 (50 000 rows; its training started from the clean KD model, so this point is a clean one): first form
#       within the limits, tiny10
#   segment 4 (200 000 rows; segments 2-4 started from shifted models): first form within the limits, tiny10 and trec-covid
# No candidate of the replacement rule comes from here; the numbers correct the reading "the dose curve is flat".
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=b2p1b MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
source "$EXP_ROOT/queues/b2_common.sh"
guard_alive
CURVE="$RESULTS_ROOT/b1c_curve.tsv"
log "=== b2 p1b start: stage-1 models of R-c' with the bias restored ==="
unshift() {  # unshift MODEL: models/MODEL-unshifted = a real copy with the three shifts of b1c added back
    local m="$1" out="$1-unshifted"
    [ -d "models/$out" ] && return 0
    [ -d "models/$m" ] || { log "  models/$m missing"; return 1; }
    need_space || return 1
    if cp -rL "models/$m" "models/$out" && [ ! -L "models/$out" ] && "$PY" - "models/$out" <<'EOF'
import math, sys
from safetensors.torch import load_file, save_file
d = sys.argv[1] + "/model.safetensors"
t = load_file(d); n = 0
for base in ("lm_head.bias", "predictions.bias"):
    if base in t:
        b = t[base]
        for thr in (0.15, 0.10, 0.05):      # the reverse of the order they were subtracted in
            b = b + (math.exp(thr) - 1.0)
        t[base] = b; t[base.replace(".bias", ".decoder.bias")] = b.clone(); n += 1
        print(base, "and its decoder copy shifted back by", sum(math.exp(x) - 1.0 for x in (0.05, 0.10, 0.15)))
assert n, "no MLM head bias tensor found"
save_file(t, d, metadata={"format": "pt"})
EOF
    then log "  $out written (the bias of $m + 0.3183)"; to_dd "$out"
    else
        log "  $out FAILED"
        if [ -L "models/$out" ]; then rm -f "models/$out"; elif [ -d "models/$out" ]; then rm -rf "models/$out"; fi    # only the partial copy just made
        return 1
    fi
}
ROWS=""
point() {  # point K ROWS FULL_MEASURE: unshift segment K, check its density against b1c's, first form, tiny10 (and trec-covid)
    local k="$1" n="$2" m="arm-b1-rc-s1p$1" u="arm-b1-rc-s1p$1-unshifted" was ok=""
    unshift "$m" || return 1
    sparsity "$u" "models/$u" || return 1
    was=$(awk -F'\t' -v m="$m" '$2==m {print $8}' "$CURVE" 2>/dev/null | tail -n 1)
    ok=$("$PY" -c "import sys;print(int(abs(float(sys.argv[1]) - float(sys.argv[2])) <= 1.0))" "$(dims "$u")" "${was:-0}" 2>/dev/null)
    log "  $u: $(dims "$u") dims/doc without a shift; b1c measured $was before its first shift -> $([ "$ok" = 1 ] && echo "the restored model reproduces it" || echo "MISMATCH: the restored model is not the trained one")"
    [ "$ok" = 1 ] || return 1
    form_fit2 "$u" || return 1
    regime quantized "$PICK" "models/$PICK" tiny10 16
    [ "$3" = full ] && regime quantized "$PICK" "models/$PICK" heldout2_treccovid 8
    log "  TRUE POINT, $n rows: $PICK tiny10 $(lbv "$PICK" tiny10)$([ "$3" = full ] && echo ", trec-covid $(tc "$PICK")") at $(dims "$PICK") dims/doc, $(pq "$FORMS/$PICK.json" "$PICK") postings/query; b1c reported $(lbv "$m-bias015" tiny10) at $(dims "$m-bias015") dims/doc for its mislabelled form"
    ROWS="$ROWS| $n | \`$PICK\` | $(lbv "$PICK" tiny10) | $([ "$3" = full ] && tc "$PICK" || echo -) | $(dims "$PICK") | $(qd "$FORMS/$PICK.json" "$PICK") | $(pq "$FORMS/$PICK.json" "$PICK") | $(lbv "$m-bias015" tiny10) | $(dims "$m-bias015") |"$'\n'
    EXTRA="$EXTRA $u $PICK"
}
EXTRA=""
point 1 50000 tiny; pause_point
point 4 200000 full
{
    echo "## p1b: erratum to the dose curve of R-c' stage 1 (summary_b1.md)"; echo
    echo "Queue b1c moved every stage-1 model to the data drive before it wrote the bias forms, and the copy of the forms copied the link instead of the files. The three shifts (thresholds 0.05, 0.10, 0.15) therefore went into the raw model itself and added up: every \`arm-b1-rc-s1pK\` on the disk has its MLM head bias lower by 0.3183 (threshold 0.276), the form labelled \`bias015\` in summary_b1.md is that model, and the labels \`bias005\` / \`bias010\` are the sums of one and two shifts. Consequences: (1) the dose curve (64.60 / 64.50 / 64.65 / 64.62) and the end of stage 1 (64.62 / 43.00 / trec-covid 67.33 at 315.6 dims/doc, \"postings -20 %\") were read in a far sparser form than the first form within the limits; (2) segments 2-4 each started from a model shifted that way, and so did the dose3 round \`$RCD\`; (3) the reading \"the dose curve is flat\" does not follow from those numbers. R-a, the KD model and every 82 M number are not affected (their forms were copies of real directories)."; echo
    echo "The shift added back on real copies (the copy reproduces the density b1c measured before its first shift), first form within both limits:"; echo
    echo "| MS MARCO-pl rows | restored model, measured form | tiny10 | trec-covid | dims/doc | dims/query | postings/query | tiny10 reported by b1c | at dims/doc |"; echo "|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    printf '%s' "$ROWS"
    echo "| 0 | \`$KD15\` (KD alone) | $(lbv "$KD15" tiny10) | $(tc "$KD15") | $(dims "$KD15") | $(qd "$FORMS/$KD15.json" "$KD15") | $(pq "$FORMS/$KD15.json" "$KD15") | | |"
    echo
    echo "The point at 50 000 rows is clean (its training started from the KD model). The point at 200 000 rows is the end of a stage whose segments 2-4 each started from a shifted model, so it is not a clean measurement of what 200 000 rows give. heldout8 was not measured here."
} > "$B2/notes_p1b.md"
EXTRA_LABELS="$EXTRA" write_summary "Add the b2 results: the stage-1 models of R-c' with the bias restored (erratum to the dose curve)"
log "=== b2 p1b done ($(date '+%d.%m %H:%M')) ==="
