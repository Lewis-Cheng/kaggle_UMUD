#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
# avoid stale unauthorized kaggle.json overriding token in some code paths
mv ~/.kaggle/kaggle.json ~/.kaggle/kaggle.json.bak_lewischeng 2>/dev/null || true
cd ~/kaggle_UMUD
OUT=reports/phase3/rl_track_b
echo '=== submit ==='
kaggle competitions submit -c umud-challenge-muscle-architecture-in-ultrasound-data \
  -f $OUT/submission_track_b_v5_rl.csv \
  -m "trackB_v5_rl_hybrid" | tee $OUT/kaggle_submit_stdout.txt
echo '=== poll scores ==='
for i in 1 2 3 4 5 6 7 8; do
  sleep 15
  echo "--- poll $i $(date -R) ---"
  kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>&1 | tee $OUT/kaggle_submissions_list.txt | head -20
  line=$(grep -i 'trackB_v5_rl_hybrid\|trackB' $OUT/kaggle_submissions_list.txt | head -1 || true)
  echo "LINE=$line"
  # break when not pending
  if echo "$line" | grep -Eiq 'Complete|Error|SubmissionError|Finished'; then break; fi
  if echo "$line" | grep -Eiq '[0-9]+\.[0-9]+'; then break; fi
done
date -R