#!/usr/bin/env bash
# Queue p1, part a (15.11, in the pause of b1 after its KD; ~1 h 15 min): the diagnostics of the static query table.
#   W1  the released table reading 512 query tokens instead of 128 ($T-len512, rows unchanged): tiny10 asymmetrically and
#       the query cost (dims and postings per query on the cost sample, per task, arguana apart). Over 16 807
#       postings/query it is not a candidate: the weight threshold (A3: prune --min-weight 0.05 / 0.1 / 0.2) is applied to
#       it at once, the cost measured again and tiny10 run for the variants within the limit.
#   W2  the oracles on pool r2 (agreement with the teacher's ranking on 2 000 held-out queries)
#   W3  the oracles on tiny10: weights, support, own, student-own (W3b, heldout8, is left out)
# Run by logs/p1_chain.sh; every step is one job on the card.
cd "$(dirname "$0")/../../.." || exit 1
source experiments/polish-splade-small-2026-11/queues/p1_common.sh
echo $$ > "$LOGS/queue_p1.pid"
[ -n "$(sget t0)" ] || sset t0 "$(date +%s)"
L=$T-len512
log "=== p1 part a start: static table $T over $FULL (released: $REL_T10 / $REL_H8); $(free_gb) GB free on / ==="
for f in "$SM/$T/static_query.json" "$SM/$L/static_query.json" "$POOL2/pool.json" "$DOC/model.safetensors"; do [ -s "$f" ] || log "  $f missing"; done

# W1
log "=== p1 W1: $L, tiny10 and query cost ==="
ev "$L"
cost w1 "$T" "$L"
cand "$T" "the released table (128 query tokens)"
cand "$L" "reads 512 query tokens, rows unchanged"
log "  W1: $L tiny10 $(t10 "$L") (released $(t10 "$T")); postings/query $(sget "pq:$L") against $(sget "pq:$T"), arguana-pl alone $(sget "pqa:$L") against $(sget "pqa:$T")"
if [ -n "$(sget "pq:$L")" ] && ! within "$L"; then
    log "  $L is over $MAX_POSTINGS postings/query: the weight threshold is applied to it now (A3)"
    for mw in 005 010 020; do prune "$L" "$L-mw$mw" --min-weight "0.${mw#0}"; done
    cost a3-len512 "$L-mw005" "$L-mw010" "$L-mw020"
    for mw in 005 010 020; do
        if within "$L-mw$mw"; then ev "$L-mw$mw"; cand "$L-mw$mw" "512 query tokens, table weights below 0.${mw#0} dropped"
        else log "  $L-mw$mw: $(sget "pq:$L-mw$mw") postings/query, still over the limit: no tiny10"; fi
    done
    sset a3_done "$L"
fi
B=$(sget best); LEN=0
case "$B" in "$L"*) [ "$(gte "$(t10 "$B")" "$(t10 "$T")" 0.05)" = 1 ] && LEN=1;; esac
sset len512 "$LEN"
log "  W1 reading: best within the limit is ${B:-none} ($(t10 "${B:-none}")); the 512-token reading $([ "$LEN" = 1 ] && echo "gains and fits: the next steps start from it" || echo "does not gain within the limit: the next steps start from the released 128-token reading")"

# W2
log "=== p1 W2: oracles on pool r2 ==="
if [ ! -s "$RS/oracles_$T.json" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 1800 "$PY" -m distill.static_diag oracles --pool "$POOL2" --model "$SM/$T" --device cuda --out "$RS/oracles_$T.json" > "$LOGS/p1_oracles.log" 2>&1 \
        && record "p1:oracles" ok "-" "$((SECONDS - s))s" || { log "  oracles FAILED (logs/p1_oracles.log)"; record "p1:oracles" FAILED "-" "$((SECONDS - s))s"; }
fi
grep -v "^$" "$LOGS/p1_oracles.log" 2>/dev/null | tail -n 12 | cut -c1-300 | while read -r l; do log "  $l"; done

# W3
log "=== p1 W3: oracles on tiny10 ==="
for m in weights support own student-own; do
    [ -e "$SM/oracle-$m" ] || { log "  $SM/oracle-$m missing"; continue; }
    PIRB_QUERY_MODEL="$SM/oracle-$m" regime quantized "$(olab "$m")" "$DOC" tiny10 16
done
REF=$(t10 "$L"); REF=${REF:-$(t10 "$T")}
log "  oracles on tiny10 (the oracles stand on the 512-token table: reference $REF; full model 65.06): weights $(lbv "$(olab weights)" tiny10), support $(lbv "$(olab support)" tiny10), own $(lbv "$(olab own)" tiny10), student-own $(lbv "$(olab student-own)" tiny10)"
table
log "=== p1 part a done ($(date '+%d.%m %H:%M'), $(elapsed_min) min since the start of p1) ==="
