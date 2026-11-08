#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token | tr -d "\n\r ")
# ensure lewischeng json present (combo that listed OK)
if [ -f ~/.kaggle/kaggle.json.bak_lewischeng ]; then
  cp ~/.kaggle/kaggle.json.bak_lewischeng ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi
cd ~/kaggle_UMUD
OUT=reports/phase3/rl_track_b
echo "auth check:"
kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>&1 | head -5
echo "=== SUBMIT ==="
kaggle competitions submit -c umud-challenge-muscle-architecture-in-ultrasound-data \
  -f "$OUT/submission_track_b_v5_rl.csv" \
  -m "trackB_v5_rl_hybrid" 2>&1 | tee "$OUT/kaggle_submit_stdout.txt"
echo SUBMIT_EXIT:$?
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 12
  echo "--- poll $i $(TZ=Asia/Hong_Kong date '+%Y-%m-%d %H:%M:%S %Z') ---"
  kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>&1 | tee "$OUT/kaggle_submissions_list.txt" | head -20
  if grep -qi 'trackB_v5_rl_hybrid' "$OUT/kaggle_submissions_list.txt"; then
    line=$(grep -i 'trackB_v5_rl_hybrid' "$OUT/kaggle_submissions_list.txt" | head -1)
    echo "FOUND=$line"
    echo "$line" | grep -Eiq 'Pending|Submitting|Scanning' || break
  fi
done