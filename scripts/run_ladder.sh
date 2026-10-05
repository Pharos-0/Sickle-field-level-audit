#!/usr/bin/env bash
# The full ladder, SEQUENTIALLY. Never run two of these at once: the earlier
# gate sweep was discarded because concurrent loops at different batch sizes
# overwrote each other's output (docs/REPRODUCTION_GATE.md section 4a).
set -u
PY="${SICKLE_PY:-/d/Final Project M.Tech/.venv/Scripts/python.exe}"
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0}"
for v in v1 v2 v3 v4 v5 v6 v7 v8 v9 v10; do
  for s in $SEEDS; do
    echo "=== $v seed $s  $(date -Is) ==="
    PYTHONIOENCODING=utf-8 "$PY" -m src.train --config "$v" --seed "$s" || \
      echo "   $v seed $s FAILED"
  done
done
echo "LADDER COMPLETE"
