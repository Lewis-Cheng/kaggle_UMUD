#!/bin/bash
set -euo pipefail
ROOT=/home/chocheng9/kaggle_UMUD
VENV=$ROOT/.venv_tf
LOG=$ROOT/reports/phase3/chase018/dump_eval_dltrack_fasc.log
exec > >(tee -a "$LOG") 2>&1
echo "START $(date)"
if [[ ! -x $VENV/bin/python ]]; then
  /home/chocheng9/.pyenv/versions/3.12.0/bin/python -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "tensorflow==2.17.1" opencv-python-headless pandas numpy
echo PIP_OK
"$VENV/bin/python" -u "$ROOT/scripts/chase018/dltrack_dump_only.py"
"$ROOT/.venv/bin/python" -u "$ROOT/scripts/chase018/dltrack_eval_only.py"
"$ROOT/.venv/bin/python" "$ROOT/scripts/chase018/update_lewis_status.py"
echo "DONE $(date)"
