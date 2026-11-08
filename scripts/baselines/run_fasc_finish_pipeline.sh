#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/kaggle_UMUD"
OUT="$ROOT/reports/phase3"
LOG="$OUT/run_fasc_finish_pipeline.log"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Iseconds)] waiter v3: wait for v4b only (v4c optional/slow)"
while pgrep -f 'train_seg_fasc_v4b\.py' >/dev/null; do
  echo "[$(date -Iseconds)] v4b still training"
  tail -1 "$OUT/run_fasc_v4b.log" 2>/dev/null || true
  sleep 40
done
echo "[$(date -Iseconds)] v4b finished"
# snapshot v4c status but do not block
if pgrep -f 'train_seg_fasc_v4c\.py' >/dev/null; then
  echo "v4c still running; postprocess will use current best among finished ckpts"
  tail -2 "$OUT/run_fasc_v4c.log" 2>/dev/null || true
fi
# skip long v4 letterbox resume (known weak ~0.15); go straight to fair eval + GH
bash "$ROOT/scripts/baselines/run_fasc_postprocess.sh"