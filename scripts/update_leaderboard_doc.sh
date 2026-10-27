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
SPARSITY=$(mktemp)
python3 scripts/sparsity_table.py --model base-splade-wrapped=base --model soup-of-ft --model soup-v2b > "$SPARSITY"
python3 - "$DOC" "$TMP" "$SPARSITY" <<'EOF'
import sys
doc, tables, sparsity = sys.argv[1:4]
text = open(doc).read()
for marker, path in (("tables", tables), ("sparsity", sparsity)):
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    a, b = text.index(start) + len(start), text.index(end)
    text = text[:a] + "\n" + open(path).read().strip("\n") + "\n" + text[b:]
open(doc, "w").write(text)
print(f"tables and sparsity updated in {doc}")
EOF
rm -f "$TMP" "$SPARSITY"
