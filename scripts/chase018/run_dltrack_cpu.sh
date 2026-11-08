#!/bin/bash
set -euo pipefail
ROOT=/home/chocheng9/kaggle_UMUD
LOG=$ROOT/reports/phase3/chase018/dump_eval_dltrack_fasc.log
: > "$LOG"
{
  echo "START $(date) mode=cpu-forced"
  "$ROOT/.venv_tf/bin/python" -u "$ROOT/scripts/chase018/dltrack_dump_only.py"
  echo "DUMP_EXIT $?"
  ls "$ROOT/reports/phase3/chase018/cache/maps_multexp_dltrack_fasc"/*.npz | wc -l
  "$ROOT/.venv/bin/python" -u "$ROOT/scripts/chase018/dltrack_eval_only.py"
  echo "EVAL_EXIT $?"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/chase018/update_lewis_status.py"
  echo "DONE $(date)"
} >> "$LOG" 2>&1