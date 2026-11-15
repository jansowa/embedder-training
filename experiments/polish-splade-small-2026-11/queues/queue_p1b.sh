#!/usr/bin/env bash
# Queue p1, part b (15.11, in the pause of b1 after its KD; ~1.5-2.5 h): the released architecture of the static table.
#   A1  training hygiene: six short trainings on pool r2 (3 epochs, from the table before round 2, phr2-k8-kd, in the
#       reading length W1 chose): the baseline, sparse_adam, adagrad, --value-floor, --dim-gain, temperature 2;
#       tiny10 for at most three arms that beat the baseline on the validation rows (top-1 agreement with the teacher)
#   A3  the query weight threshold on the best table so far: prune --min-weight 0.05 / 0.1 / 0.2, cost and tiny10
#   A2  only when the oracle `weights` is >= 1.5 above the best table: slot learning, round 1 (proposals, training, tiny10)
#   A4  only when the best candidate is >= +0.3 on tiny10 over the released table: the options that helped put together
#       (when at least two did), tiny10, then heldout8 and the cost of the final candidate
# After A3: when more than 4 h have passed since the start of p1, A2 and the GPU benchmarks (part c) are skipped.
# Nothing in the release package is touched. Run by logs/p1_chain.sh; every step is one job on the card.
cd "$(dirname "$0")/../../.." || exit 1
source experiments/polish-splade-small-2026-11/queues/p1_common.sh
echo $$ > "$LOGS/queue_p1.pid"
[ -n "$(sget t0)" ] || sset t0 "$(date +%s)"
L=$T-len512; LEN=$(sget len512); SUF=""; INIT=phr2-k8-kd
if [ "$LEN" = 1 ] && prune phr2-k8-kd phr2-k8-kd-len512 --max-length 512; then SUF=-len512; INIT=phr2-k8-kd-len512; fi
REFN=$T; [ -n "$SUF" ] && REFN=$L          # the table the A1 arms are read against on tiny10
log "=== p1 part b start: A1 from $INIT ($([ -n "$SUF" ] && echo "512 query tokens: W1 gained within the limit" || echo "128 query tokens, as released")); reference $REFN $(t10 "$REFN"); best so far $(sget best) ==="

# A1
log "=== p1 A1: training hygiene, six arms from $INIT on pool r2 ==="
declare -A ARM=([base]="" [sparseadam]="--optimizer sparse_adam" [adagrad]="--optimizer adagrad --lr 5e-2" [floor]="--value-floor 1e-4" [gain]="--dim-gain" [temp2]="--temperature 2")
for a in base sparseadam adagrad floor gain temp2; do strain "a1-$a$SUF" "$INIT" ${ARM[$a]}; done
SEL=$("$PY" - "$RS" "$SUF" "$T" <<'PYEOF' 2>> "$LOGS/p1_select.log"
import json, sys
rs, suf, rel = sys.argv[1:4]
def v(name):
    try:
        return json.load(open(f"{rs}/{name}.json"))["val_final"]["top1_agreement"]
    except Exception:
        return None
b, r = v(f"a1-base{suf}"), v(rel)
arms = sorted(((v(f"a1-{a}{suf}"), a) for a in ("sparseadam", "adagrad", "floor", "gain", "temp2") if v(f"a1-{a}{suf}") is not None), reverse=True)
picked = [a for x, a in arms if b is None or x > b + 0.0005][:3]
if not picked and b is not None and r is not None and b >= r + 0.003:
    picked = ["base"]          # the baseline itself beats the released table on the validation rows (keep-best)
sys.stderr.write("baseline %s, released %s, arms %s -> %s\n" % (b, r, arms, picked))
print(" ".join(picked))
PYEOF
)
log "  A1 validation top-1: baseline $(sval "a1-base$SUF" top1_agreement), released $T $(sval "$T" top1_agreement), sparse_adam $(sval "a1-sparseadam$SUF" top1_agreement), adagrad $(sval "a1-adagrad$SUF" top1_agreement), value floor $(sval "a1-floor$SUF" top1_agreement), dim gain $(sval "a1-gain$SUF" top1_agreement), temperature 2 $(sval "a1-temp2$SUF" top1_agreement); tiny10 for: ${SEL:-none}"
A1N=(); for a in $SEL; do ev "a1-$a$SUF" && A1N+=("a1-$a$SUF"); done
[ "${#A1N[@]}" -gt 0 ] && cost a1 "${A1N[@]}"
for a in $SEL; do cand "a1-$a$SUF" "A1 arm $a from $INIT (${ARM[$a]:-baseline options})"; done
table

# A3
B=$(sget best); B=${B:-$T}
log "=== p1 A3: weight threshold on the best table so far, $B ($(t10 "$B"), $(sget "pq:$B") postings/query) ==="
case "$B" in
    *-mw0??) log "  $B is already a thresholded table (A3 ran on its source in W1): not repeated" ;;
    *)  for mw in 005 010 020; do prune "$B" "$B-mw$mw" --min-weight "0.${mw#0}"; done
        cost "a3-$B" "$B-mw005" "$B-mw010" "$B-mw020"
        for mw in 005 010 020; do
            if within "$B-mw$mw"; then ev "$B-mw$mw"; cand "$B-mw$mw" "$B with table weights below 0.${mw#0} dropped"
            else log "  $B-mw$mw: no cost within the limit ($(sget "pq:$B-mw$mw") postings/query): no tiny10"; fi
        done ;;
esac
table
EL=$(elapsed_min)
if [ "$EL" -gt 240 ]; then sset skip57 1; log "  $EL min since the start of p1 (over 4 h): A2 and the GPU benchmarks are skipped and stay for later"; fi

# A2 (conditional)
B=$(sget best); B=${B:-$T}; OW=$(lbv "$(olab weights)" tiny10)
log "=== p1 A2: slot learning only when the oracle weights (${OW:-not measured}) is >= 1.5 above the best table ($B $(t10 "$B")) ==="
if [ "$(sget skip57)" = 1 ]; then log "  skipped (time)"
elif [ -n "$OW" ] && [ "$(gte "$OW" "$(t10 "$B")" 1.5)" = 1 ]; then
    G="$B-g1"
    if [ ! -f "$SM/$G/static_query.json" ]; then
        wait_gpu_free; wait_ram; s=$SECONDS
        ORPHAN_GUARD_EXEMPT=1 timeout -k 60 1800 "$PY" -m distill.static_slots --pool "$POOL2" --model "$SM/$B" --out "$SM/$G" --new-slots 2 --device cuda > "$LOGS/p1_slots_$G.log" 2>&1
        [ -f "$SM/$G/static_query.json" ] && { log "  slots $G in $((SECONDS - s))s: $(tail -n 1 "$LOGS/p1_slots_$G.log" | cut -c1-300)"; record "p1:slots:$G" ok "-" "$((SECONDS - s))s"; } \
            || { log "  slot proposals FAILED (logs/p1_slots_$G.log)"; record "p1:slots:$G" FAILED "-" "$((SECONDS - s))s"; }
    fi
    if strain "$G-kd" "$G"; then ev "$G-kd"; cost a2 "$G-kd"; cand "$G-kd" "A2 round 1: up to 2 new slots per row of $B, 3 epochs on pool r2"; fi
    log "  A2 round 1: $G-kd tiny10 $(t10 "$G-kd") against $B $(t10 "$B"); round 2 (a fresh pool) was not run in this pause"
else log "  condition not met: A2 skipped"; fi
table

# A4 (conditional)
B=$(sget best); B=${B:-$T}
log "=== p1 A4: only when the best candidate ($B $(t10 "$B")) is >= +0.3 over the released table ($REL_T10) ==="
if [ "$B" != "$T" ] && [ "$(gte "$(t10 "$B")" "$REL_T10" 0.3)" = 1 ]; then
    # the A1 options that helped on tiny10 (>= +0.1 over the reference of their reading length), one optimiser at most
    HELP=(); OPT=""; OPTV=""
    for a in $SEL; do
        [ "$a" = base ] && continue
        v=$(t10 "a1-$a$SUF"); [ -n "$v" ] && [ "$(gte "$v" "$(t10 "$REFN")" 0.1)" = 1 ] || continue
        case "$a" in
            sparseadam|adagrad) if [ -z "$OPT" ] || [ "$(gte "$v" "$OPTV" 0.005)" = 1 ]; then OPT=$a; OPTV=$v; fi ;;
            *) HELP+=("$a") ;;
        esac
    done
    [ -n "$OPT" ] && HELP+=("$OPT")
    if [ "${#HELP[@]}" -ge 2 ]; then
        ARGS=(); for a in "${HELP[@]}"; do ARGS+=(${ARM[$a]}); done
        C="a4-combo$SUF"
        log "  options that helped: ${HELP[*]}; one training with all of them: $C (${ARGS[*]})"
        if strain "$C" "$INIT" "${ARGS[@]}"; then
            ev "$C"; cost a4 "$C"; cand "$C" "A4: the options ${HELP[*]} together, from $INIT"
            case "$B" in *-mw0??) mw=${B##*-mw}; if prune "$C" "$C-mw$mw" --min-weight "0.${mw#0}"; then cost "a4-mw" "$C-mw$mw"; within "$C-mw$mw" && ev "$C-mw$mw"; cand "$C-mw$mw" "A4: $C with table weights below 0.${mw#0} dropped"; fi ;; esac
        fi
    else log "  fewer than two A1 options helped on tiny10 (${HELP[*]:-none}): nothing to put together, the final candidate is the best table"; fi
    F=$(sget best)
    log "  final candidate $F: tiny10 $(t10 "$F"); heldout8 and the cost on the heldout8 tasks follow"
    ev "$F" heldout8
    if [ ! -s "$P1/cost_h8.json" ]; then
        wait_gpu_free; wait_ram
        ORPHAN_GUARD_EXEMPT=1 timeout -k 60 2400 "$PY" scripts/splade_index_cost_asym.py --doc-model "full=$DOC" --query "$T=$SM/$T" --query "$F=$SM/$F" \
            --tasks "maupqa-gpt3.5-cc,maupqa-multilingual-NLI,mfaq,poleval-2022/legal-questions,maupqa-wiki-def,scidocs-pl,maupqa-1z10,fiqa-pl" \
            --output-json "$P1/cost_h8.json" > "$LOGS/p1_cost_h8.log" 2>&1 || log "  cost on the heldout8 tasks FAILED (logs/p1_cost_h8.log)"
    fi
    grep -E "^full|over full|sample" "$LOGS/p1_cost_h8.log" 2>/dev/null | cut -c1-300 | while read -r l; do log "  heldout8 tasks: $l"; done
    H=$(lbv "$(lab "$F")" heldout8)
    if [ -n "$H" ] && [ "$(gte "$H" "$REL_H8")" = 1 ] && within "$F"; then
        sset swap "$F"; log "  $F MEETS THE SWAP RULE: tiny10 $(t10 "$F") (>= +0.3 over $REL_T10), heldout8 $H (>= $REL_H8), $(sget "pq:$F") postings/query (<= $MAX_POSTINGS). The package files are not touched; the decision is the user's"
    else log "  $F does not meet the swap rule: heldout8 ${H:-not measured} against $REL_H8, $(sget "pq:$F") postings/query"; fi
else log "  condition not met: A4 skipped; no candidate for the swap"; fi
table
log "=== p1 part b done ($(date '+%d.%m %H:%M'), $(elapsed_min) min since the start of p1); best table $(sget best) $(t10 "$(sget best)") ==="
