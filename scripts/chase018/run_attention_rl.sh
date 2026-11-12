#!/bin/bash
# Spatial-attention + RL line agent on the lab. One GPU. No Kaggle submit.
# Spec: reports/phase3/chase018/attention_rl/SPEC.md
set -euo pipefail
cd /home/chocheng9/kaggle_UMUD
export PYTHONPATH=src
PY=.venv/bin/python
DEV="${DEV:-cuda:0}"
OUT=reports/phase3/chase018/attention_rl/lab
mkdir -p "$OUT"

if ! command -v lake >/dev/null && [[ ! -x "$HOME/.elan/bin/lake" ]]; then
  echo "[attn-rl] Lean is not installed; lean_build will be BLOCKED and the run will not be cleared."
  echo "[attn-rl] Install once: curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none"
fi

run() {
  echo "[attn-rl] $*"
  $PY scripts/chase018/train_spatial_attention_rl.py "$@" --device "$DEV" 2>&1 | tee -a "$OUT/run.log"
}

run prepare
run train
run eval-multexp
run infer-test
$PY scripts/chase018/fv_attention_rl_stages.py --lab "$OUT" 2>&1 | tee -a "$OUT/run.log"
echo "[attn-rl] gates: reports/phase3/chase018/attention_rl/fv_status.json"
