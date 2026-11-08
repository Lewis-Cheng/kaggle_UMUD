#!/bin/bash
set -e
cd ~/kaggle_UMUD
export PATH="$HOME/.local/bin:$PATH"
PY=/home/chocheng9/images_reduced/.venv/bin/python
OUT=reports/phase3/rl_track_b
$PY - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
sub = pd.read_csv("reports/phase3/rl_track_b/submission_track_b_v5_rl.csv", sep=";")
gh = pd.read_csv("reports/phase3/submission_tier_b_seg_gh_v5best.csv", sep=";")
sid = set(gh["image_id"].astype(str).str.strip())
uid = set(sub["image_id"].astype(str).str.strip())
print("n_sub", len(sub), "n_gh", len(gh), "missing", len(sid-uid), "extra", len(uid-sid))
assert sid == uid
assert list(sub.columns)[:4] == ["image_id","pa_deg","fl_mm","mt_mm"]
assert np.isfinite(sub[["pa_deg","fl_mm","mt_mm"]].to_numpy()).all()
print("SOFT_VALIDATE_OK")
PY
echo '=== kaggle submit ==='
kaggle competitions submit -c umud-challenge-muscle-architecture-in-ultrasound-data \
  -f $OUT/submission_track_b_v5_rl.csv \
  -m "trackB_v5_rl_hybrid" | tee $OUT/kaggle_submit_stdout.txt
echo '=== wait for score ==='
for i in 1 2 3 4 5 6; do
  sleep 20
  echo "--- poll $i ---"
  kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>/dev/null | head -12 | tee $OUT/kaggle_submissions_list.txt
  if grep -E 'trackB_v5_rl_hybrid' $OUT/kaggle_submissions_list.txt | head -1 | grep -Eiq 'complete|pending|error|submission'; then
    # check if score filled (not just pending)
    line=$(grep 'trackB_v5_rl_hybrid' $OUT/kaggle_submissions_list.txt | head -1)
    echo "LINE=$line"
    echo "$line" | grep -Eiq 'Pending|Submitting' || break
  fi
done