#!/usr/bin/env bash
# Queue s6 (3060 Ti, 8.11 evening, ~35 min): the s5 recipe for a static query table, re-run over the publication
# candidate r2v12-bias005 instead of n5kj (s5 best: credit-k8-kd, tiny10 59.65 over the n5kj index).
#   1  credit-k32 credited from r2v12-bias005 on kd_v1 queries, cut to credit-k8 (the idf table is the tokenizer's, shared)
#   2  mined pool: 250 k kd_v1 passages, 120 k kd_v1 queries encoded by r2v12-bias005 (teacher top 32 + student top 16)
#   3  credit-k8-kd: KL against the teacher scores, learned scale, lr 5e-3, 2 epochs (the s5 settings)
#   4  tiny10 in asymmetric mode (r2v12-bias005 index, the table as the query encoder); no heldout8
#   5  summary_s6.md, commit, push
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=s6 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_s6.pid"
guard_alive
DOC_MODEL=${DOC_MODEL:-models/arm-dense-kl-n5kj-ens400-r2v12-bias005}
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
POOL=${POOL:-cache/static_pools/kd-r2v12b}
SM="models/static-r2v12b"
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }

for p in ${WAIT_PIDS:-}; do
    if kill -0 "$p" 2>/dev/null; then log "waiting for pid $p to exit"; while kill -0 "$p" 2>/dev/null; do sleep 30; done; sleep 20; fi
done
mkdir -p "$SM" "$RESULTS_ROOT/static-r2v12b"
[ -e "$SM/idf" ] || ln -s "$(realpath models/static/idf)" "$SM/idf"
log "=== s6 start: static query table over $DOC_MODEL; $(free_gb) GB free on / ==="

if [ ! -f "$SM/credit-k32/static_query.json" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS; log "build credit-k32"
    timeout -k 60 5400 "$PY" -m distill.static_query build --out "$SM/credit-k32" --init credit --teacher "$DOC_MODEL" --texts "$CORPUS" \
        --kinds query --top-k 32 --min-count 3 --batch-size 64 --device cuda --dtype fp16 > "$LOGS/static_build_r2v12b_credit-k32.log" 2>&1
    [ -f "$SM/credit-k32/static_query.json" ] || { log "build credit-k32 FAILED"; record "static_build:r2v12b:credit-k32" FAILED "-" "$((SECONDS - s))s"; exit 1; }
    log "build credit-k32 done in $((SECONDS - s))s"; record "static_build:r2v12b:credit-k32" ok "-" "$((SECONDS - s))s"
fi
if [ ! -f "$SM/credit-k8/static_query.json" ]; then
    "$PY" -m distill.static_query prune --model "$SM/credit-k32" --out "$SM/credit-k8" --top-k 8 > "$LOGS/static_prune_r2v12b_credit-k8.log" 2>&1 \
        || { log "prune credit-k8 FAILED"; exit 1; }
    log "credit-k8 written"
fi

if [ ! -f "$POOL/pool.json" ]; then
    need_space || exit 1; wait_gpu_free; wait_ram; s=$SECONDS; log "mine $POOL"
    timeout -k 60 7200 "$PY" -m distill.static_mine --doc-model "$DOC_MODEL" --corpus "$CORPUS" --out "$POOL" \
        --student "$SM/idf" --student "$SM/credit-k32" --device cuda --dtype fp16 --n-docs 250000 --n-queries 120000 \
        --teacher-top 32 --student-top 16 --max-candidates 64 --batch-size 16 --search-batch 128 > "$LOGS/static_mine_kd-r2v12b.log" 2>&1
    [ -f "$POOL/pool.json" ] || { log "mine FAILED"; record "static_mine:kd-r2v12b" FAILED "-" "$((SECONDS - s))s"; exit 1; }
    log "mine done in $((SECONDS - s))s"; record "static_mine:kd-r2v12b" ok "-" "$((SECONDS - s))s"
fi

name=credit-k8-kd
if [ ! -f "$SM/$name/static_query.json" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS; log "train $name"
    timeout -k 60 3600 "$PY" -m distill.train_static --pool "$POOL" --init "$SM/credit-k8" --out "runs/static-r2v12b-$name" --lr 5e-3 \
        --target rows --learn-scale --epochs 2 --batch-size 64 --warmup-steps 200 --schedule linear --val-rows 2000 --eval-every 500 \
        --log-every 100 --device cuda --seed 42 > "$LOGS/static_train_r2v12b_$name.log" 2>&1
    [ -f "runs/static-r2v12b-$name/final/static_query.json" ] || { log "train $name FAILED"; record "static_train:r2v12b:$name" FAILED "-" "$((SECONDS - s))s"; exit 1; }
    cp -r "runs/static-r2v12b-$name/final" "$SM/$name"; cp "runs/static-r2v12b-$name/static_summary.json" "$RESULTS_ROOT/static-r2v12b/$name.json"
    log "train $name done in $((SECONDS - s))s"; record "static_train:r2v12b:$name" ok "-" "$((SECONDS - s))s"
fi

PIRB_QUERY_MODEL="$SM/$name" regime quantized "asym-r2v12b-q-static-$name" "$DOC_MODEL" tiny10 16
log "static $name over r2v12-bias005: tiny10 $(lbv "asym-r2v12b-q-static-$name" tiny10)"

SUM="$RESULTS_ROOT/summary_s6.md"
"$PY" - "$RESULTS_ROOT/static-r2v12b/$name.json" "$POOL/pool.json" "$(lbv "asym-r2v12b-q-static-$name" tiny10)" > "$SUM" <<'EOF'
import json, sys
s, pool, t = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3] or "-"
v, ref = s.get("val_final", {}), s.get("val_splade_reference") or {}
print("# s6 (8.11): static query table credit-k8-kd over the r2v12-bias005 index (quantized regime, nDCG@10)\n")
print(f"Pool: {pool['rows']} kd_v1 queries over {pool['docs']} kd_v1 passages, {pool['candidates_mean']} candidates per query; the s5 recipe (KL, learned scale, lr 5e-3, 2 epochs).\n")
print("| query side | index | tiny10 | val top1 | dims/q | postings fraction |\n|---|---|---:|---:|---:|---:|")
print("| credit-k8-kd (s5) | n5kj | 59.65 | | 83.3 | 10.945 |")
print(f"| credit-k8-kd (s6) | r2v12-bias005 | {t} | {v.get('top1_agreement', 0):.3f} | {v.get('dims_per_query', 0):.1f} | {v.get('postings_fraction', 0):.3f} |")
print(f"| r2v12-bias005 symmetric | r2v12-bias005 | 64.21 | {ref.get('top1_agreement', 0):.3f} | {ref.get('dims_per_query', 0):.1f} | {ref.get('postings_fraction', 0):.3f} |")
EOF
log "summary written to $SUM"
git add "$SUM" "$RESULTS_ROOT/static-r2v12b" "$REGIMES_ROOT"/quantized/asym-r2v12b-q-static-* "$EXP_ROOT/queues/queue_s6.sh" 2>/dev/null
git commit -q -m "Add the s6 results: a static query table trained over the r2v12-bias005 index" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_s6.log" 2>&1 && { echo pushed >> "$LOGS/push_s6.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== s6 done ($(date '+%d.%m %H:%M')) ==="
