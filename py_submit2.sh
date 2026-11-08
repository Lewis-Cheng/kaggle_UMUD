#!/bin/bash
set -e
cd ~/kaggle_UMUD
OUT=reports/phase3/rl_track_b
TOKEN=$(tr -d "\n\r " < ~/.kaggle/access_token)
# Environment method expected by kaggle CLI
export KAGGLE_USERNAME=tokenuser
export KAGGLE_KEY="$TOKEN"
# Also write json with token as key
python3 - <<PY
import json, os
from pathlib import Path
p = Path.home()/".kaggle"/"kaggle.json"
p.write_text(json.dumps({"username":"tokenuser","key":os.environ["KAGGLE_KEY"]}))
os.chmod(p, 0o600)
print("wrote", p)
PY
export PATH="$HOME/.local/bin:$PATH"
echo "list:"
kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>&1 | head -5
echo "submit via CLI:"
kaggle competitions submit -c umud-challenge-muscle-architecture-in-ultrasound-data \
  -f $OUT/submission_track_b_v5_rl.csv \
  -m "trackB_v5_rl_hybrid" 2>&1 | tee $OUT/kaggle_submit_stdout.txt
echo EXIT:$?
sleep 20
kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>&1 | tee $OUT/kaggle_submissions_list.txt | head -20