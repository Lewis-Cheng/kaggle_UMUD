#!/bin/bash
cd ~/kaggle_UMUD
tr -d '\000' < reports/phase3/rl_track_b/pipeline.log | tail -80
echo '---PROCS---'
ps -u chocheng9 -o pid,etime,cmd | grep -E 'train_geometry|infer_test|run_track_b|kaggle competitions' | grep -v grep || echo none
echo '---FILES---'
ls -la reports/phase3/rl_track_b/