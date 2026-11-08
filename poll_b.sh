#!/bin/bash
cd ~/kaggle_UMUD
# strip NULs from log for readability
tr -d '\000' < reports/phase3/rl_track_b/pipeline.log | tail -40
echo '---'
ps -u chocheng9 -o pid,etime,cmd | grep -E 'et_predict|train_geometry|infer_test|run_track_b|kaggle' | grep -v grep || echo 'no trackB procs'
ls -la reports/phase3/rl_track_b/ | head -30