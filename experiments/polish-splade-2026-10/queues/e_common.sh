# Shared by the stages of queue e (source after queue_lib.sh, arm_common.sh, release_common.sh and b2_common.sh).
MAIN=experiments/polish-splade-2026-10
EDIR="$MAIN/results/e"; ESUM="$MAIN/results/summary_e.md"; mkdir -p "$EDIR"
A=datasets/arms; DROWS=$A/n5-plus-kj-mined-dose3.jsonl; TR=$A/n5-plus-kj-mined-dose3-trcd.jsonl; TV=$A/n5-plus-kj-mined-dose3-tv9b.jsonl
NEWROWS=$A/n5-plus-kj-mined-dose3-tensrcd.jsonl
DEADLINE=${E_DEADLINE:-2026-11-18 19:00}
left_min() { echo $(( ( $(date -d "$DEADLINE" +%s) - $(date +%s) ) / 60 )); }
form_fit5() {  # form_fit5 MODEL: PICK = the first of no shift, 0.05, 0.10 ... 0.30 with <= MAX_DIMS dims/doc and <= MAX_POSTINGS postings/query:
               # the grid on which the frozen full model and the released ALBERT got their forms, so a candidate is read like the model it would replace
    local m="$1" t f p; PICK=""
    [ -d "models/$m" ] || { log "  $m missing: no form"; return 1; }
    for t in "" 005 010 015 020 025 030; do
        f=$(form_of "$m" "$t")
        if [ -n "$t" ]; then biased2 "$m" "$f" "0.${t#0}" || return 1; to_dd "$f"; fi
        sparsity "$f" "models/$f" || continue
        [ "$(gte "$MAX_DIMS" "$(dims "$f")")" = 1 ] || { log "  $f: $(dims "$f") dims/doc, above $MAX_DIMS"; continue; }
        index_cost "$FORMS/$f.json" "$f"; p=$(pq "$FORMS/$f.json" "$f")
        log "  $f: $(dims "$f") dims/doc, $(qd "$FORMS/$f.json" "$f") dims/query, ${p:-?} postings/query (limits $MAX_DIMS and $MAX_POSTINGS)"
        if [ -n "$p" ] && [ "$(gte "$MAX_POSTINGS" "$p")" = 1 ]; then PICK="$f"; return 0; fi
    done
    log "  $m: no form within both limits up to bias030"; return 1
}
m11of() { "$PY" scripts/b2_candidates.py mean --regimes "$QREG" "$1" m11 2>/dev/null; }
cand5() {  # cand5 STAGE NAME MODEL DESCRIPTION: first form on the 0.05 grid, tiny10 and trec-covid, candidate line; C5_PICK
    C5_PICK=""
    form_fit5 "$3" || return 1
    C5_PICK="$PICK"
    regime quantized "$C5_PICK" "models/$C5_PICK" tiny10 16
    regime quantized "$C5_PICK" "models/$C5_PICK" heldout2_treccovid 8
    log "  $C5_PICK: tiny10 $(lbv "$C5_PICK" tiny10), trec-covid $(tc "$C5_PICK"), mean of the 11 tasks $(m11of "$C5_PICK") at $(dims "$C5_PICK") dims/doc, $(pq "$FORMS/$C5_PICK.json" "$C5_PICK") postings/query"
    cand "$1" "$2" "$C5_PICK" "$3" "$4"
}
esummary() {  # esummary COMMIT_MESSAGE: summary_e.md from the notes of the stages; commit
    {
        echo "# e (quantized regime, nDCG@10): a teacher with the 124 M model for the released models"; echo
        echo "Hypothesis: in the ensemble teacher of the 82 M family (soup-v9b + PolDense-400M) the SPLADE member soup-v9b is replaced by R-c' (\`arm-b1-rc-d3\`, the 124 M model, about 1.5 points of tiny10 above soup-v9b); the way of combining, the dense scale and the weights stay. E1: the dose3 arm of the frozen full model retrained with that teacher and souped with the other arms of the frozen model (model A). E2: one more round of the released ALBERT with that teacher (model B). Gates: at least +0.2 on the mean of the 19 tasks (tiny10 + heldout8 + trec-covid) over the released model within 410 dims/doc and 16 807 postings/query. Candidates are read in their first form within both limits on the grid of 0.05, the grid on which the released models got their forms. One seed each; the seed noise of the ranking recipe is 0.13-0.2 on tiny10."; echo
        echo "A risk that belongs to every number here: R-c' was itself trained on these rows with the ens400 teacher, so its scores are partly circular; a gain, if any, comes from the larger capacity of the 124 M model, not from new data."; echo
        local s; for s in s1 s2 s3; do [ -s "$EDIR/notes_$s.md" ] && { cat "$EDIR/notes_$s.md"; echo; }; done
        echo "## Queue state (e)"; echo; echo '```'; grep -a -E "e_score_|ensrcd|soup-v21-|r3rcd|soup-hl-dqf-r3|soup2-b1-" "$STATE" | tail -n 120; echo '```'
    } > "$ESUM"
    log "summary written to $ESUM"
    git add "$ESUM" "$EDIR" "$MAIN"/queues/queue_e.sh "$MAIN"/queues/queue_e_s*.sh "$MAIN/queues/e_common.sh" scripts/swap_splade_teacher.py 2>/dev/null
    git add "$MAIN"/configs/next/3060/distil_dense_kl_n5kj_mined_dose3_ensrcd_r2v12.yaml experiments/polish-splade-small-2026-11/configs/3060/stage3_hl_dqf_full_r3_dose3_ensrcd.yaml 2>/dev/null
    local l root
    for l in $(cut -f3 "$EDIR"/candidates_*.tsv 2>/dev/null) ${EXTRA_LABELS:-}; do
        for root in "$MAIN" experiments/polish-splade-small-2026-11; do
            [ -e "$root/results-regimes/quantized/$l" ] && git add "$root/results-regimes/quantized/$l" 2>/dev/null
            [ -e "$root/results/sparsity/$l.json" ] && git add "$root/results/sparsity/$l.json" 2>/dev/null
            [ -e "$root/results/index_cost_b1_forms/$l.json" ] && git add "$root/results/index_cost_b1_forms/$l.json" 2>/dev/null
        done
    done
    git commit -q -m "$1" 2>/dev/null && log "committed $(git log --oneline -1)"
    push_loop "$QUEUE_TAG"
}
