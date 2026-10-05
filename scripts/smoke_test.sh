#!/usr/bin/env bash
# Prove the pipeline moves before committing hours to it.
# Two training batches and a full validation pass per rung. NOT a result.
set -euo pipefail
PY="${SICKLE_PY:-/d/Final Project M.Tech/.venv/Scripts/python.exe}"
cd "$(dirname "$0")/.."
PYTHONIOENCODING=utf-8 "$PY" scripts/check_configs.py
for v in "${@:-v1 v2}"; do
  echo "=== smoke $v ==="
  PYTHONIOENCODING=utf-8 "$PY" -m src.train --config "$v" --seed 0 --smoke --max-batches 2 \
    2>&1 | tail -6
done
