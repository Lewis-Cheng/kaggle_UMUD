#!/usr/bin/env bash
# Track B: frozen v5 + RL refiner → Kaggle submit (real public score).
set -euo pipefail
ROOT="${ROOT:-$HOME/kaggle_UMUD}"
cd "$ROOT"
PY="${PY:-/home/chocheng9/images_reduced/.venv/bin/python}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

DATA_ROOT="$ROOT/data/raw/umud"
SPLIT="$ROOT/reports/phase2/split_assignments.jsonl"
if [[ ! -f "$SPLIT" ]]; then
  SPLIT=$(find "$ROOT" -name 'split_assignments.jsonl' 2>/dev/null | head -1)
fi
export PATH="$HOME/.local/bin:/opt/anaconda3/bin:$PATH"
MULTEXP_IMG="$ROOT/data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
CONS="$ROOT/data/external/umud_expert_arch_v0.1.0/expert_consensus.csv"
OUT="$ROOT/reports/phase3/rl_track_b"
GH_ROWS="$ROOT/reports/phase3/multexp_v5_baseline/multexp_v5_rows.csv"
GH_SUB="$ROOT/reports/phase3/submission_tier_b_seg_gh_v5best.csv"
ET_SUB="$ROOT/reports/phase3/submission_extratrees_mask_proxy.csv"
mkdir -p "$OUT"

echo "[1] ExtraTrees preds on MultExpData"
$PY scripts/rl/et_predict_multexp.py \
  --data-root "$DATA_ROOT" \
  --split-assignments "$SPLIT" \
  --multexp-images "$MULTEXP_IMG" \
  --consensus-csv "$CONS" \
  --out-csv "$OUT/multexp_et_preds.csv"

echo "[2] Train geometry bandit LOOCV"
$PY scripts/rl/train_geometry_bandit.py \
  --multexp-gh-csv "$GH_ROWS" \
  --multexp-et-csv "$OUT/multexp_et_preds.csv" \
  --consensus-csv "$CONS" \
  --out-dir "$OUT" \
  --epochs 80

echo "[3] Infer RL refiner on official test"
# detect whether GH sub uses .tif ids
ID_FLAG=""
if head -2 "$GH_SUB" | grep -q '\.tif'; then ID_FLAG="--id-with-tif"; fi
DELIM=";"
if head -1 "$GH_SUB" | grep -q ',' && ! head -1 "$GH_SUB" | grep -q ';'; then DELIM=","; fi
$PY scripts/rl/infer_test_rl_refiner.py \
  --policy "$OUT/geometry_bandit.npz" \
  --gh-submission "$GH_SUB" \
  --et-submission "$ET_SUB" \
  --out-csv "$OUT/submission_track_b_v5_rl.csv" \
  --out-rows "$OUT/submission_track_b_v5_rl_rows.csv" \
  --delimiter "$DELIM" \
  $ID_FLAG

echo "[4] Validate submission id set"
EXPECTED="$OUT/expected_test_ids.txt"
$PY - << PY
from pathlib import Path
import pandas as pd
gh = Path("$GH_SUB")
raw = gh.read_text().splitlines()[0]
delim = ";" if raw.count(";") >= raw.count(",") else ","
df = pd.read_csv(gh, sep=delim)
col = "image_id" if "image_id" in df.columns else "id"
ids = [str(x).strip() for x in df[col].tolist()]
Path("$EXPECTED").write_text("\n".join(ids) + "\n")
print("wrote", len(ids), "expected ids")
PY
$PY scripts/validate_submission.py "$OUT/submission_track_b_v5_rl.csv" --expected-ids "$EXPECTED"

echo "[5] Kaggle submit"
COMP="umud-challenge-muscle-architecture-in-ultrasound-data"
kaggle competitions submit -c "$COMP" \
  -f "$OUT/submission_track_b_v5_rl.csv" \
  -m "trackB_v5_rl_hybrid" | tee "$OUT/kaggle_submit_stdout.txt"

echo "[6] Fetch public score (wait briefly then list)"
sleep 15
kaggle competitions submissions -c "$COMP" | tee "$OUT/kaggle_submissions_list.txt" | head -20

echo "DONE Track B → $OUT"
