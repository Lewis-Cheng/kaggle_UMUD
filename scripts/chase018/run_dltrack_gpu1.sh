#!/bin/bash
set -euo pipefail
ROOT=/home/chocheng9/kaggle_UMUD
export CUDA_VISIBLE_DEVICES=1
LOG=$ROOT/reports/phase3/chase018/dump_eval_dltrack_fasc.log
exec > >(tee -a "$LOG") 2>&1
echo "START $(date)"
"$ROOT/.venv_tf/bin/python" -u "$ROOT/scripts/chase018/dltrack_dump_only.py"
echo "DUMP_EXIT $?"
"$ROOT/.venv/bin/python" -u "$ROOT/scripts/chase018/dltrack_eval_only.py"
echo "EVAL_EXIT $?"
"$ROOT/.venv/bin/python" "$ROOT/scripts/chase018/update_lewis_status.py"
echo "DONE $(date)"