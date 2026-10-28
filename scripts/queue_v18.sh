#!/usr/bin/env bash
# 28.10 evening, after queue_v17: the teacher pass for the length-decorrelated negatives
# and the third package for the 16 GB machine. The student-negative arms could not separate
# "long" from "hard" (their negatives are both); scripts/build_fiqa_length_negatives.py
# prepared student-easy candidates chosen by length only (long-easy: longer than the
# query's longest positive; short-easy: shorter than its shortest positive, the control).
# Here roberta-v3 vetoes false negatives among them and the export keeps 10 per query
# (VARIANT=long-easy / short-easy of scripts/fiqa_splade_mine.sh, ~5 min each), then the
# package datasets/transfer/fiqa-splade-5070c-<date>.tar.gz is packed with the band cap3
# (the 10 hardest safe negatives at least 3 below the best positive) and the three configs.
# Start with: setsid nohup bash scripts/queue_v18.sh > logs/queue_v18.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v18 MIN_RAM_GB=6
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v18.pid"

waited=0
until [ "${SKIP_WAIT:-0}" = 1 ] || grep -qE '=== v17 done ===' logs/queue_v17.out 2>/dev/null; do
    [ "$waited" -ge 21600 ] && { log "v18: v17 not finished after 6 h; exiting"; exit 1; }
    sleep 120; waited=$((waited + 120))
done
wait_gpu_free
log "=== v18 start: teacher pass for the length negatives ==="

for v in long-easy short-easy; do
    f="datasets/arms/fiqa-2073-splade-$v.jsonl"; s=$SECONDS
    if [ -f "$f" ]; then log "data $v already present"; else
        VARIANT="$v" bash scripts/fiqa_splade_mine.sh > "$LOGS/fiqa_splade_$v.out" 2>&1 \
            && log "data $v done in $((SECONDS - s))s" || log "data $v FAILED after $((SECONDS - s))s (logs/fiqa_splade_$v.out)"
    fi
    "$PY" - "$v" <<'EOF' 2>/dev/null | while read -r line; do log "  $line"; done
import json, sys
v = sys.argv[1]; f = f"results/data/fiqa-2073-splade-{v}"
try:
    r = json.load(open(f + ".json")); m = json.load(open(f + "-mining.json")); t = json.load(open(f + "-teacher-filter.json"))
    print(f"{v}: rows {r.get('rows_written')} tiers {r.get('negative_tiers')} v3 neg median {r['v3_neg_scores']['median']:.2f} "
          f"margin median {r['v3_min_pos_minus_max_neg']['median']:.2f} neg>pos {r.get('queries_where_a_negative_outscores_a_positive')}")
    print(f"candidates: tokens median {m['candidate_tokens']['median']:.0f} (positive bound median {m['positive_bound_tokens']['median']:.0f}), "
          f"over 512 {m['candidates_over_512_share']:.1%}, filled by closest {m['filled_by_closest_length']}; teacher rejected top20 {t.get('top20_rejected_by_teacher_share')}")
except Exception as e:
    print(f"{v}: no report ({e})")
EOF
done

log "=== v18: ten negatives per row, package ==="
ok=1
for f in datasets/arms/fiqa-2073-splade-long-easy.jsonl datasets/arms/fiqa-2073-splade-short-easy.jsonl datasets/arms/fiqa-2073-splade-band-cap3.jsonl; do
    [ -f "$f" ] || { log "  missing $f"; ok=0; continue; }
    n=$("$PY" -c "
import json, sys
c = {}
for line in open('$f'):
    r = json.loads(line); c[len(r['neg'])] = c.get(len(r['neg']), 0) + 1
print(c)")
    log "  $(basename "$f"): negatives per row $n"; [[ "$n" == "{10: "* ]] || ok=0
done
if [ "$ok" = 1 ]; then
    P="datasets/transfer/fiqa-splade-5070c-$(date +%Y%m%d).tar.gz"
    tar -czf "$P" datasets/arms/fiqa-2073-splade-long-easy.jsonl datasets/arms/fiqa-2073-splade-short-easy.jsonl datasets/arms/fiqa-2073-splade-band-cap3.jsonl \
        configs/next/3060/mnrl_fiqa_splade_long_easy.yaml configs/next/3060/mnrl_fiqa_splade_short_easy.yaml configs/next/3060/mnrl_fiqa_splade_band_cap3.yaml \
        && log "package ready: $P ($(du -h "$P" | cut -f1))" || log "packing FAILED"
else log "package not built: a file is missing or has rows without 10 negatives"; fi
log "=== v18 done ==="
