#!/usr/bin/env bash
# Prove the pipeline moves before committing hours to it.
# Two training batches and a full validation pass per rung. NOT a result.
set -euo pipefail
# Interpreter resolution, in order: $SICKLE_PY, the project's own .venv,
# then whatever `python` is on PATH. The project is self-contained; it must not
# depend on an environment belonging to a different project.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SICKLE_PY:-$ROOT/.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
cd "$(dirname "$0")/.."
PYTHONIOENCODING=utf-8 "$PY" scripts/check_configs.py
for v in "${@:-v1 v2}"; do
  echo "=== smoke $v ==="
  PYTHONIOENCODING=utf-8 "$PY" -m src.train --config "$v" --seed 0 --smoke --max-batches 2 \
    2>&1 | tail -6
done
