#!/usr/bin/env bash
# Queue s5 (3060 Ti, 8.11, ~1.7 h, before copy / v43 / onnx): train the static query table by distillation from the
# symmetric SPLADE on a mined pool. The document side stays frozen (models/arm-dense-kl-n5kj-ens400, the s4 index).
# Why: in s4 the training on the n5 + kj rows changed nothing (idf-rows = idf = 55.76 on tiny10): the rows pool is easy
# (val MRR 0.97 at the start) and the lr 5e-3 moved the idf weights (~8) by 1 % on average.
#   0  unit tests of the static modules (tests/test_distill_static*.py); then the code is committed
#   1  smoke mining on the GPU (3 000 passages, 500 queries): the sparse CSR x dense search runs on CUDA
#   2  mined pool: 300 k kd_v1 passages and 120 k kd_v1 queries encoded by n5kj; per query the teacher's top 32 and the
#      top 16 of the idf and credit-k32 tables (the students' own false positives), teacher scores q_splade . d as targets
#   3  credit-k8 = credit-k32 cut to the 8 strongest slots per token (fewer dims per query, cheaper search)
#   4  training, KL over each query's candidates against the teacher scores, a learned global score scale:
#      idf-kd (k = 1, lr 5e-2, the weights are ~8), credit-k8-kd and credit-k32-kd (lr 5e-3)
#   5  tiny10 in asymmetric mode for the three trained tables and the untrained credit-k8 (no heldout8: tiny10 screens)
#   6  summary_s5.md, commit, push
# Reading (tiny10; symmetric n5kj 64.14; s4: idf 55.76, credit-k32 59.20; leaderboard BM25 pyserini 50.07, bm25s 53.75):
#   idf-kd - idf >= 1 = the own-token weights were the bottleneck (cheapest query side, ~13 dims);
#   credit-k32-kd >= 60 = the frozen index is usable with a static query side; credit-k8-kd within 0.5 of it = take k8;
#   no trained table above its init by 0.3 = the frozen index is the limit, next is the document side (separate model).
#   started by the gate logs/queue_s5.sh once logs/s5.ready exists (it holds the slot after s4)
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=s5 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_s5.pid"
guard_alive
DOC_MODEL=${DOC_MODEL:-models/arm-dense-kl-n5kj-ens400}
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
POOL=${POOL:-cache/static_pools/kd-n5kj}
SMOKE_POOL=cache/static_pools/kd-smoke
SM="models/static"
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }

wait_gpu_free
[ -f "$CORPUS" ] || { log "$CORPUS missing"; exit 1; }
[ -d "$DOC_MODEL" ] || { log "document model $DOC_MODEL missing"; exit 1; }
for t in idf credit-k32; do [ -f "$SM/$t/static_query.json" ] || { log "table $SM/$t missing (built by s4)"; exit 1; }; done
mkdir -p "$SM" "$RESULTS_ROOT/static"
log "=== s5 start: static query tables trained on a mined pool over $DOC_MODEL; $(free_gb) GB free on / ==="

log "=== s5 step 0: unit tests ==="
wait_ram
if ! timeout -k 30 900 "$PY" -m pytest -q tests/test_distill_static.py tests/test_distill_static_mine.py > "$LOGS/s5_pytest.log" 2>&1; then
    log "unit tests FAILED (logs/s5_pytest.log); stopping"; record "pytest:s5" FAILED "-"; exit 1
fi
log "unit tests passed: $(grep -E 'passed' "$LOGS/s5_pytest.log" | tr '\n' ' ')"
git add distill/static_mine.py distill/train_static.py tests/test_distill_static_mine.py docs/distillation.md "$EXP_ROOT/queues/queue_s5.sh" 2>/dev/null
git commit -q -m "Add mined candidate pools and a learned score scale for training static query tables" 2>/dev/null && log "code committed $(git log --oneline -1)"

mine() {  # mine OUT LOGNAME [static_mine args]
    local out="$1" name="$2"; shift 2
    [ -f "$out/pool.json" ] && return 0
    need_space || return 1; wait_gpu_free; wait_ram; log "mine $name ($*)"; local s=$SECONDS
    timeout -k 60 7200 "$PY" -m distill.static_mine --doc-model "$DOC_MODEL" --corpus "$CORPUS" --out "$out" \
        --student "$SM/idf" --student "$SM/credit-k32" --device cuda --dtype fp16 "$@" > "$LOGS/static_mine_$name.log" 2>&1
    if [ -f "$out/pool.json" ]; then
        log "mine $name done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$out/pool.json'));print(d['rows'], 'queries', d['docs'], 'passages,', d['candidates_mean'], 'cands/query (', d['student_only_candidates_mean'], 'from the students only ), recall of the teacher top 10 in the student top', d['student_top'], d['student_recall_of_teacher_top10'], '; encode docs', d['encode_docs_seconds'], 's, search', d['search_seconds'], 's')")"
        record "static_mine:$name" ok "-" "$((SECONDS - s))s"
    else log "mine $name FAILED (logs/static_mine_$name.log)"; record "static_mine:$name" FAILED "-" "$((SECONDS - s))s"; return 1; fi
}
strain() {  # strain NAME INIT [train_static args]: runs/static-NAME, final copied to models/static/NAME
    local name="$1" init="$2"; shift 2
    [ -f "$SM/$name/static_query.json" ] && return 0
    [ -f "$SM/$init/static_query.json" ] || { log "train $name: init $init missing"; return 1; }
    wait_gpu_free; wait_ram; log "train $name from $init ($*)"; local s=$SECONDS
    timeout -k 60 3600 "$PY" -m distill.train_static --pool "$POOL" --init "$SM/$init" --out "runs/static-$name" "$@" > "$LOGS/static_train_$name.log" 2>&1
    if [ -f "runs/static-$name/final/static_query.json" ]; then
        cp -r "runs/static-$name/final" "$SM/$name"; cp "runs/static-$name/static_summary.json" "$RESULTS_ROOT/static/$name.json"
        log "train $name done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$RESULTS_ROOT/static/$name.json'));i,f=d.get('val_init',{}),d.get('val_final',{});print('val top1 %.3f -> %.3f, mrr %.4f -> %.4f, tau %.3f -> %.3f, dims/q %.1f, scale %s -> %s' % (i.get('top1_agreement',0), f.get('top1_agreement',0), i.get('mrr',0), f.get('mrr',0), i.get('kendall_tau',0), f.get('kendall_tau',0), f.get('dims_per_query',0), d.get('scale_init'), d.get('scale_final')))")"
        record "static_train:$name" ok "-" "$((SECONDS - s))s"
    else log "train $name FAILED (logs/static_train_$name.log)"; record "static_train:$name" FAILED "-" "$((SECONDS - s))s"; return 1; fi
}
asym() {  # asym NAME TASKSET: the static table NAME as the query encoder over DOC_MODEL's index
    [ -f "$SM/$1/static_query.json" ] || return 1
    PIRB_QUERY_MODEL="$SM/$1" regime quantized "asym-n5kj-q-static-$1" "$DOC_MODEL" "$2" 16
    log "  static $1 over n5kj: $2 $(lbv "asym-n5kj-q-static-$1" "$2")"
}

log "=== s5 step 1: smoke mining on the GPU ==="
mine "$SMOKE_POOL" smoke --n-docs 3000 --n-queries 500 --log-every 50 || { log "smoke mining failed; stopping"; exit 1; }

log "=== s5 step 2: mined pool ==="
mine "$POOL" kd-n5kj --n-docs 300000 --n-queries 120000 --teacher-top 32 --student-top 16 --max-candidates 64 --batch-size 16 --search-batch 128 \
    || { log "mining failed; stopping"; exit 1; }

log "=== s5 step 3: credit-k8 ==="
if [ ! -f "$SM/credit-k8/static_query.json" ]; then
    "$PY" -m distill.static_query prune --model "$SM/credit-k32" --out "$SM/credit-k8" --top-k 8 > "$LOGS/static_prune_credit-k8.log" 2>&1 \
        && log "credit-k8 written: $("$PY" -c "import json;d=json.load(open('$SM/credit-k8/static_query.json'));print('k', d['k'], 'active slots', d['active_slots'])")" \
        || log "credit-k8 FAILED (logs/static_prune_credit-k8.log)"
fi

log "=== s5 step 4: training ==="
T_COMMON=(--target rows --learn-scale --epochs 2 --batch-size 64 --warmup-steps 200 --schedule linear --val-rows 2000 --eval-every 500 --log-every 100 --device cuda --seed 42)
strain idf-kd idf --lr 5e-2 "${T_COMMON[@]}"
strain credit-k32-kd credit-k32 --lr 5e-3 "${T_COMMON[@]}"
strain credit-k8-kd credit-k8 --lr 5e-3 "${T_COMMON[@]}"

log "=== s5 step 5: tiny10 in asymmetric mode ==="
TRAINED=(idf-kd credit-k32-kd credit-k8-kd)
for name in "${TRAINED[@]}" credit-k8; do asym "$name" tiny10; done
best=$(for name in "${TRAINED[@]}"; do v=$(lbv "asym-n5kj-q-static-$name" tiny10); [ -n "$v" ] && echo "$v $name"; done | sort -rn | head -1 | awk '{print $2}')
# no heldout8 here (8.11): tiny10 screens the techniques; heldout8 only later, for a few shortlisted tables
[ -n "$best" ] && log "best trained static table on tiny10: $best"

log "=== s5 step 6: summary ==="
SUM="$RESULTS_ROOT/summary_s5.md"
{
    echo "# s5 ($(date '+%d.%m.%Y %H:%M'), quantized regime, nDCG@10): static query tables trained on a mined pool, index of $(basename "$DOC_MODEL") frozen"
    echo
    echo "Pool ($POOL): $( "$PY" -c "import json;d=json.load(open('$POOL/pool.json'));print(f\"{d['rows']} kd_v1 queries over {d['docs']} kd_v1 passages; per query the teacher's top {d['teacher_top']} plus the top {d['student_top']} of each of {', '.join(p.split('/')[-1] for p in d['students'])} ({d['candidates_mean']} candidates, {d['student_only_candidates_mean']} found by the students only); recall of the teacher's top 10 in the student's top {d['student_top']}: {d['student_recall_of_teacher_top10']}\")" 2>/dev/null )."
    echo "Training: KL(teacher || student) over each query's candidates, teacher = q_splade . d of n5kj (no labels), a learned global score scale, 2 epochs; -kd = trained on this pool. Validation: 2 000 held-out queries, top-1 agreement = the student's top candidate is the teacher's, quantized query weights."
    echo
    echo '```'
    "$PY" - "$RESULTS_ROOT/static" "$REGIMES_ROOT/quantized" <<'EOF' 2>> "$LOGS/queue_summary.log"
import json, sys
from pathlib import Path
static, regimes = Path(sys.argv[1]), Path(sys.argv[2])
def nd(label, ts):
    f = regimes / label / f"{ts}.json"
    return json.load(open(f))["mean_ndcg@10"] if f.exists() else None
fmt = lambda v, p=2: f"{v:.{p}f}" if isinstance(v, (int, float)) else "-"
print(f"{'query side':26s} {'tiny10':>7s} {'heldout8':>9s} {'val top1':>9s} {'val tau':>8s} {'dims/q':>7s} {'post.frac':>10s} {'scale':>7s}")
print(f"{'BM25 pyserini (leaderb.)':26s} {'50.07':>7s} {'33.34':>9s}")
print(f"{'BM25 bm25s (leaderb.)':26s} {'53.75':>7s} {'36.17':>9s}")
print(f"{'polish-splade (leaderb.)':26s} {'61.53':>7s} {'41.06':>9s}")
print(f"{'n5kj symmetric':26s} {'64.14':>7s} {'43.01':>9s}")
ref = None
for name, init in [("idf", None), ("idf-kd", "idf"), ("credit-k8", None), ("credit-k8-kd", "credit-k8"), ("credit-k32", None), ("credit-k32-rows", None), ("credit-k32-kd", "credit-k32")]:
    label = f"asym-n5kj-q-static-{name}"
    s = json.load(open(static / f"{name}.json")) if (static / f"{name}.json").exists() else {}
    if not s and init is None:
        # an untrained table: its validation is the val_init of the run that starts from it
        kd = static / f"{name}-kd.json"
        s = {"val_final": json.load(open(kd)).get("val_init", {})} if kd.exists() else {}
    v = s.get("val_final", {})
    if s.get("pool", {}).get("kind") == "mined":
        ref = ref or s.get("val_splade_reference")
    print(f"{name:26s} {fmt(nd(label, 'tiny10')):>7s} {fmt(nd(label, 'heldout8')):>9s} {fmt(v.get('top1_agreement'), 3):>9s} {fmt(v.get('kendall_tau'), 3):>8s} {fmt(v.get('dims_per_query'), 1):>7s} {fmt(v.get('postings_fraction'), 3):>10s} {fmt(s.get('scale_final'), 3):>7s}")
if ref:
    print(f"{'n5kj symmetric (val)':26s} {'':>7s} {'':>9s} {fmt(ref.get('top1_agreement'), 3):>9s} {fmt(ref.get('kendall_tau'), 3):>8s} {fmt(ref.get('dims_per_query'), 1):>7s} {fmt(ref.get('postings_fraction'), 3):>10s}")
EOF
    echo '```'
    echo; echo "credit-k32-rows is s4's table (trained on the n5 + kj rows); its val numbers come from the rows pool, the others from the mined pool. Postings fraction = sum of df / N over the query's dims on the pool's passages (x corpus size = postings per query)."
    echo; echo "## Queue state (s5)"; echo; echo '```'
    grep -E "^(static_mine|static_train:(idf|credit-k8|credit-k32)-kd|regime:quantized:asym-n5kj-q-static-(idf-kd|credit-k8|credit-k32-kd))" "$STATE"
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_s5.out"
log "summary written to $SUM"
git add "$SUM" "$RESULTS_ROOT"/static "$REGIMES_ROOT"/quantized/asym-n5kj-q-static-* "$EXP_ROOT/queues/queue_s5.sh" 2>/dev/null
git commit -q -m "Add the s5 results: static query tables distilled from the symmetric SPLADE on a mined pool" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_s5.log" 2>&1 && { echo pushed >> "$LOGS/push_s5.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== s5 done ($(date '+%d.%m %H:%M')) ==="
