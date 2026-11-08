#!/bin/bash
cd ~/kaggle_UMUD
sed -i 's|PY="${PY:-/opt/anaconda3/bin/python}"|PY="${PY:-/home/chocheng9/images_reduced/.venv/bin/python}"|' scripts/rl/run_track_b_pipeline.sh
grep '^PY=' scripts/rl/run_track_b_pipeline.sh
mkdir -p reports/phase3/rl_track_b
nohup bash scripts/rl/run_track_b_pipeline.sh > reports/phase3/rl_track_b/pipeline.log 2>&1 &
echo PID:$!
sleep 3
head -40 reports/phase3/rl_track_b/pipeline.log
