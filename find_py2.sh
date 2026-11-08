#!/bin/bash
cd ~/kaggle_UMUD
echo "=== launch scripts ==="
head -40 scripts/baselines/launch_fasc_v910.sh 2>/dev/null
echo "===="
head -40 scripts/baselines/run_fasc_v5_finish.sh 2>/dev/null
echo "==== find python anywhere ==="
type python python3 2>&1
ls -la .venv/bin/python 2>&1
# module system?
module avail 2>&1 | head -5
# common lab locations
ls /usr/local/cuda* 2>&1 | head
find /home/chocheng9 -maxdepth 4 -name 'python' -type f 2>/dev/null | head -40
find /opt -maxdepth 5 -name 'torch' -type d 2>/dev/null | head -20
# who uses gpu 2/3
nvidia-smi 2>&1 | head -60
echo "==== hist ==="
grep -E "python|conda|torch" ~/.bash_history 2>/dev/null | tail -30