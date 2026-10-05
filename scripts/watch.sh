#!/usr/bin/env bash
# Progress of any rung, from trainlog.json (written every epoch) rather than
# best.json, so it works while a run is still going.
#   bash scripts/watch.sh          # v1 seed 0
#   bash scripts/watch.sh v2 0
cd "$(dirname "$0")/.."
V="${1:-v1}"; S="${2:-0}"
L="outputs/logs/${V}_seed${S}.log"
D="outputs/crop_type/${V}_seed${S}"
PY="${SICKLE_PY:-/d/Final Project M.Tech/.venv/Scripts/python.exe}"
echo "=== $V seed $S ==="
if [ ! -f "$L" ]; then echo "  NOT RUN (no log at $L)"; exit 0; fi
echo "  epochs done : $(grep -cE '^ep ' "$L")"
echo "  NaN batches : $(grep -c 'non-finite' "$L")"
grep -E '^ep ' "$L" | tail -3 | sed 's/^/  /'
[ -f "$D/best.json" ] && PYTHONIOENCODING=utf-8 "$PY" -c "
import json,sys
d=json.load(open(r'$D/best.json'))
print('  best  : epoch %s  pixel %.4f  plot %.4f  paddy %.4f  %s' % (
  d.get('best_epoch'), d.get('macro_iou',0), d.get('plot_macro_iou',0),
  d.get('paddy_iou',0), 'COMPLETE' if d.get('complete') else 'in progress'))
" 2>/dev/null
S1="outputs/logs/chain_${V}_seed${S}.status"
[ -f "$S1" ] && { echo "  chain :"; tail -3 "$S1" | sed 's/^/    /'; }
exit 0
