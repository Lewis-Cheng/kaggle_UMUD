#!/bin/bash
set -e
cd ~/kaggle_UMUD
echo "=== find torch ==="
candidates=(
  .venv/bin/python
  ~/miniconda3/bin/python
  ~/anaconda3/bin/python
  /opt/conda/bin/python
  /usr/bin/python3
)
# glob envs
shopt -s nullglob
for p in ~/miniconda3/envs/*/bin/python ~/.conda/envs/*/bin/python /home/chocheng9/*/bin/python; do
  candidates+=("$p")
done
for p in "${candidates[@]}"; do
  [ -x "$p" ] || continue
  ver=$("$p" -c 'import torch; print(torch.__version__)' 2>/dev/null) || continue
  echo "FOUND $p torch=$ver"
done
echo "=== processes ==="
ps -u chocheng9 -o pid,pcpu,pmem,cmd --sort=-pcpu | head -30
echo "=== which from tmux v7 ==="
tmux capture-pane -t umud_fasc_v7 -p -S -30 2>/dev/null | tail -20