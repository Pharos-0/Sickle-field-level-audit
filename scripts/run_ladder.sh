#!/usr/bin/env bash
# The full ladder, SEQUENTIALLY. Never run two of these at once: the earlier
# gate sweep was discarded because concurrent loops at different batch sizes
# overwrote each other's output (docs/REPRODUCTION_GATE.md section 4a).
set -u
# Interpreter resolution, in order: $SICKLE_PY, the project's own .venv,
# then whatever `python` is on PATH. The project is self-contained; it must not
# depend on an environment belonging to a different project.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SICKLE_PY:-$ROOT/.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
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
