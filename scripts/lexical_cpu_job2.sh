#!/usr/bin/env bash
# Second CPU-only lemma job: Stanza metrics for the two arms the first job did
# not cover (trivia-2073, wikisynth-2073), so every arm has a lexical file for
# the HF upload and the coverage analysis. Same idf pool and cache as
# scripts/lexical_cpu_job.sh. Meant to be started by queue_v4 under a cgroup
# memory cap; it also refuses to start when less than 6 GB RAM is available.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=""
DPY=/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/doc2query/.venv/bin/python
OUT=results/lexical_difficulty/full
CACHE=results/lexical_difficulty/lemma_cache.jsonl
LOG=logs/lexical_cpu_job2.log
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_ram() {
    while true; do
        other=$(ps -eo pid,args | awk -v me="$$" '/lexical_difficulty\.py/ && $1 != me {c++} END {print c+0}')
        avail=$(free -g | awk '/^Mem:/ {print $7}')
        if [ "$other" -eq 0 ] && [ "${avail:-0}" -ge 6 ]; then return 0; fi
        log "waiting: other lexical jobs=$other, available RAM=${avail}GB"
        sleep 120
    done
}

[ -f datasets/synth/idf_pool.jsonl ] || { log "idf pool missing (run scripts/lexical_cpu_job.sh first)"; exit 1; }
for arm in trivia-2073 wikisynth-2073; do
    if [ -s "$OUT/$arm.jsonl" ]; then log "$arm: already present"; continue; fi
    log "$arm"
    wait_ram
    nice -n 10 "$DPY" scripts/lexical_difficulty.py --input "$arm=datasets/arms/$arm.jsonl" \
        --output "$OUT/$arm.jsonl" --idf-pool datasets/synth/idf_pool.jsonl --cache "$CACHE" --threads 2 >> "$LOG" 2>&1 \
        && log "$arm done ($(wc -l < "$OUT/$arm.jsonl") rows)" || log "$arm FAILED"
done
log "lexical CPU job 2 finished"
