#!/bin/bash
set -e
cd ~/kaggle_UMUD
export PATH="$HOME/.local/bin:$PATH"
PY=/home/chocheng9/images_reduced/.venv/bin/python
OUT=reports/phase3/rl_track_b
echo '=== head submission ==='
head -3 $OUT/submission_track_b_v5_rl.csv
echo '=== compare sample ==='
head -2 data/raw/umud/sample_submission.csv | od -c | head -5
wc -l $OUT/submission_track_b_v5_rl.csv data/raw/umud/sample_submission.csv
# soft validate: exact id set vs sample
$PY - <<'PY'
from pathlib import Path
import pandas as pd
sub = pd.read_csv("reports/phase3/rl_track_b/submission_track_b_v5_rl.csv", sep=";")
sample_raw = Path("data/raw/umud/sample_submission.csv").read_text(encoding="utf-8-sig")
delim = ";" if sample_raw.splitlines()[0].count(";") >= sample_raw.splitlines()[0].count(",") else ","
sample = pd.read_csv("data/raw/umud/sample_submission.csv", sep=delim, encoding="utf-8-sig")
sid = set(sample["image_id"].astype(str).str.strip())
uid = set(sub["image_id"].astype(str).str.strip())
print("n_sub", len(sub), "n_sample", len(sample))
print("missing", len(sid-uid), "extra", len(uid-sid))
print("cols", list(sub.columns))
print("finite", sub[["pa_deg","fl_mm","mt_mm"]].apply(lambda c: c.isfinite().all()).to_dict())
assert sid == uid
assert list(sub.columns)[:4] == ["image_id","pa_deg","fl_mm","mt_mm"]
print("SOFT_VALIDATE_OK")
PY
echo '=== kaggle submit ==='
kaggle competitions submit -c umud-challenge-muscle-architecture-in-ultrasound-data \
  -f $OUT/submission_track_b_v5_rl.csv \
  -m "trackB_v5_rl_hybrid" | tee $OUT/kaggle_submit_stdout.txt
echo '=== wait for score ==='
sleep 25
kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data | tee $OUT/kaggle_submissions_list.txt | head -15