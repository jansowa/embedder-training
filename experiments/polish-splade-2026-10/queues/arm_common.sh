# Shared by the arm queues from v60 on (source after scripts/queue_lib.sh): result getters, bias forms, the measured form
# at a reference cost, row files (base rows verbatim + new rows), the query-side index cost. Do not edit while a queue
# that sources this file is running.
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
wait_for_queues() {  # wait_for_queues NAME...: wait, in this order, for every queue that is alive
    local prev
    for prev in "$@"; do
        if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
    done
}
lbv()  { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs()  { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
gte()  { "$PY" -c "import sys;a,b,d=sys.argv[1:4];print(int(bool(a) and bool(b) and float(a)>=float(b)+float(d)))" "$1" "$2" "${3:-0}"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
tc()   { lbv "$1" heldout2_treccovid; }
jget() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get(sys.argv[2]) if d.get(sys.argv[2]) is not None else '')" "$1" "$2" 2>/dev/null; }
h2task() { local bs=16; case "$2" in heldout2_poquad|heldout2_treccovid) bs=8;; esac; [ -d "models/$1" ] && regime quantized "$1" "models/$1" "$2" "$bs"; }
measured_form() {  # measured_form ARM: the (last) bias form of the arm that has a heldout8 result
    local p="" t; for t in "" -bias005 -bias010 -bias011 -bias012 -bias013 -bias014 -bias015 -bias016 -bias017 -bias018 -bias019 -bias020 -bias022 -bias025; do [ -n "$(lbv "$1$t" heldout8)" ] && p="$1$t"; done; echo "$p"
}
biased() {  # biased SRC_DIR OUT_NAME THRESHOLD: models/OUT_NAME = SRC with lm_head.bias (and its decoder copy) shifted by -(e^THRESHOLD - 1)
    local src="$1" out="$2" thr="$3"
    [ -d "models/$out" ] && return 0
    [ -d "$src" ] || { log "$out: $src missing"; return 1; }
    cp -r "$src" "models/$out" && "$PY" - "models/$out" "$thr" <<'EOF' && log "$out written (lm_head.bias = -(e^$thr - 1))" || { log "$out FAILED"; rm -rf "models/$out"; return 1; }
import math, sys
from safetensors.torch import load_file, save_file
d = sys.argv[1] + "/model.safetensors"; thr = float(sys.argv[2])
t = load_file(d); b = math.exp(thr) - 1.0; n = 0
for base in ("lm_head.bias", "predictions.bias"):
    if base in t:
        shifted = t[base] - b
        t[base] = shifted; t[base.replace(".bias", ".decoder.bias")] = shifted.clone(); n += 1
        print(base, "and its decoder copy shifted by", -b)
assert n, "no MLM head bias tensor found: " + ", ".join(k for k in t if "bias" in k and ("head" in k or "pred" in k or "decoder" in k))
save_file(t, d, metadata={"format": "pt"})
EOF
}
at_cost() {  # at_cost MODEL REFLINE [REF_LABEL]: the first bias form with <= 410 dims/doc -> tiny10, heldout8, trec-covid; PICK holds its label.
             # With REF_LABEL the cap is min(410, 1.02 x the dims/doc of REF_LABEL) and the bias grows in steps of 0.01 from 0.10 (about
             # 2.5 % of the dims each), so the new arm is measured at the reference's cost or just below it, never above.
    local m="$1" ref="$2" rl="${3:-}" t cap=410 list="005 010 015 020"; PICK="$m"
    [ -d "models/$m" ] || { log "$m missing: not measured"; return 1; }
    sparsity "$m" "models/$m"
    if [ -n "$rl" ] && [ -n "$(dims "$rl")" ]; then
        cap=$("$PY" -c "import sys;print(round(min(410.0, 1.02 * float(sys.argv[1])), 1))" "$(dims "$rl")")
        list="005 010 011 012 013 014 015 016 017 018 019 020 022 025"
    fi
    if [ "$(gte "$cap" "$(dims "$m")")" != 1 ]; then
        for t in $list; do
            biased "models/$m" "$m-bias$t" "0.${t#0}" || return 1
            sparsity "$m-bias$t" "models/$m-bias$t"; PICK="$m-bias$t"
            [ "$(gte "$cap" "$(dims "$PICK")")" = 1 ] && break
        done
        for t in $list; do [ "$m-bias$t" != "$PICK" ] && rm -rf "models/$m-bias$t"; done
    fi
    log "  $m: $(dims "$m") dims/doc unbiased; measured form $PICK at $(dims "$PICK") (cap $cap)"
    regime quantized "$PICK" "models/$PICK" tiny10 16; regime quantized "$PICK" "models/$PICK" heldout8 16
    log "  $PICK: $(lbs "$PICK") ($ref), $(dims "$PICK") dims/doc"
    regime quantized "$PICK" "models/$PICK" heldout2_treccovid 8; log "  $PICK trec-covid $(tc "$PICK"), $(dims "$PICK") dims/doc"
}
index_cost() {  # index_cost OUT_JSON LABEL...: query-side index cost (cost sample) of polish-splade and the given models
    local out="$1" s m; shift
    [ -s "$out" ] && return 0
    wait_gpu_free; wait_ram; s=$SECONDS
    local args=(--model base=models/base-splade-wrapped)
    for m in "$@"; do [ -n "$m" ] && [ -d "models/$m" ] && args+=(--model "$m=models/$m"); done
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${args[@]}" --output-json "$out" > "$LOGS/index_cost_$QUEUE_TAG.log" 2>&1 \
        && log "index cost written ($out) in $((SECONDS - s))s" || { log "index cost FAILED (logs/index_cost_$QUEUE_TAG.log)"; rm -f "$out"; }
}
merge_rows() {  # merge_rows BASE NEW OUT [DATASET] [LIMIT]: OUT = the rows of BASE verbatim + the rows of NEW (only those of group DATASET,
                # at most LIMIT), shuffled with seed 42. New rows whose query (case-folded, whitespace collapsed) already occurs in BASE
                # or earlier in NEW are left out and counted; a query_id repeated inside NEW is an error. Counts go to OUT.rows.json.
    local base="$1" new="$2" out="$3" ds="${4:-}" limit="${5:-0}"
    [ -s "$out" ] && [ -s "$out.rows.json" ] && return 0
    "$PY" - "$base" "$new" "$out" "$ds" "$limit" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, os, random, sys
base, new, out, ds, limit = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
norm = lambda q: " ".join((q or "").casefold().split())
lines, ids, queries, pairs = [], set(), set(), 0
with open(base, "rb") as f:
    for raw in f:
        if not raw.strip():
            continue
        r = json.loads(raw)   # the ids of the base rows repeat across their datasets (numeric ids), so only the query text is a key
        ids.add(str(r["query_id"])); queries.add(norm(r["query"])); pairs += len(r["pos"]); lines.append(raw if raw.endswith(b"\n") else raw + b"\n")
n_base, read, kept, dup_q, by_ds, new_ids, id_clash = len(lines), 0, 0, 0, {}, set(), 0
with open(new, "rb") as f:
    for raw in f:
        if not raw.strip():
            continue
        r = json.loads(raw)
        if ds and r.get("dataset") != ds:
            continue
        read += 1
        if limit and kept >= limit:
            continue
        qid = str(r["query_id"]); q = norm(r["query"])
        assert qid not in new_ids, "query_id repeated in the new rows: " + qid
        new_ids.add(qid); id_clash += qid in ids
        if q in queries:
            dup_q += 1
            continue
        assert len(r["neg"]) == len(r["neg_scores"]) and len(r["pos"]) == len(r["pos_scores"]), "row without teacher scores: " + qid
        queries.add(q); pairs += len(r["pos"]); kept += 1; by_ds[r.get("dataset")] = by_ds.get(r.get("dataset"), 0) + 1
        lines.append(raw if raw.endswith(b"\n") else raw + b"\n")
order = list(range(len(lines))); random.Random(42).shuffle(order)
with open(out + ".tmp", "wb") as f:
    for i in order:
        f.write(lines[i])
os.replace(out + ".tmp", out)
json.dump({"rows": len(lines), "base_rows": n_base, "new_rows": kept, "new_rows_read": read, "new_rows_left_out_duplicate_query": dup_q, "new_rows_with_a_query_id_of_a_base_row": id_clash, "new_rows_by_dataset": by_ds,
           "training_pairs": pairs, "base_rows_file": base, "new_rows_file": new, "new_rows_group": ds or None, "seed": 42}, open(out + ".rows.json", "w"), indent=2)
print("rows: %d = %d base (verbatim) + %d new (%d read, %d left out as duplicate queries); training pairs %d" % (len(lines), n_base, kept, read, dup_q, pairs))
PYEOF
    [ -s "$out" ] && [ -s "$out.rows.json" ]
}
train_budget() {  # train_budget PAIRS: seconds for 2 epochs at 2x16 (0.2405 s per pair in v56), x 1.4 plus half an hour
    "$PY" -c "import sys;print(int(int(sys.argv[1]) * 0.2405 * 1.4 + 1800))" "$1"
}
push_loop() {  # push_loop TAG: six tries, 30 minutes apart, detached
    (for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_$1.log" 2>&1 && { echo pushed >> "$LOGS/push_$1.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
    disown
}
