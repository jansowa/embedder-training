#!/usr/bin/env bash
# Regenerates the per-task tables of docs/benchmark-vs-leaderboard.md (between
# the tables:start / tables:end markers) from results-regimes/old-pyserini.
set -euo pipefail
cd "$(dirname "$0")/.."
DOC=docs/benchmark-vs-leaderboard.md
TMP=$(mktemp)
python3 scripts/leaderboard_compare.py --regime old-pyserini \
    --model base-splade-wrapped --model soup-of-ft --model soup-v2b \
    --taskset tiny10 --taskset heldout8 --taskset heldout2 > "$TMP"
python3 - "$DOC" "$TMP" <<'EOF'
import sys
doc, tables = sys.argv[1:3]
text = open(doc).read()
start, end = "<!-- tables:start -->", "<!-- tables:end -->"
a, b = text.index(start) + len(start), text.index(end)
text = text[:a] + "\n" + open(tables).read().strip("\n") + "\n" + text[b:]
open(doc, "w").write(text)
print(f"tables updated in {doc}")
EOF
rm -f "$TMP"
