#!/bin/bash
set -e
cd ~/kaggle_UMUD
sed -i 's/\r$//' scripts/rl/run_track_b_pipeline.sh || true
sed -i 's|PY="${PY:-/opt/anaconda3/bin/python}"|PY="${PY:-/home/chocheng9/images_reduced/.venv/bin/python}"|' scripts/rl/run_track_b_pipeline.sh
grep -n '^PY=' scripts/rl/run_track_b_pipeline.sh | head
mkdir -p reports/phase3/rl_track_b
pkill -f 'run_track_b_pipeline.sh' 2>/dev/null || true
sleep 1
nohup bash scripts/rl/run_track_b_pipeline.sh > reports/phase3/rl_track_b/pipeline.log 2>&1 &
echo PID:$!
sleep 6
wc -l reports/phase3/rl_track_b/pipeline.log
head -80 reports/phase3/rl_track_b/pipeline.log