#!/usr/bin/env bash
# Run Phase 3 floors A/B/C (+ optional scale hunt). Does NOT claim LB scores.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT=""
SPLIT=""
OUT_DIR="${ROOT}/reports/phase3"
MM_PER_PX="0.07225"
SCALE_PROVENANCE="unknown"
DELIMITER=","
SEED=0
MAX_TRAIN=0
MAX_ROWS=0
RUN_SCALE_HUNT=1
WEB_NOTES=""
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

usage() {
  cat <<USAGE
Usage: bash scripts/baselines/run_phase3_floors.sh \\
  --data-root PATH --split-assignments PATH [options]

Options:
  --data-root PATH              UMUD data root (required)
  --split-assignments PATH      Phase 2 split_assignments.jsonl (required)
  --out-dir PATH                output dir (default: reports/phase3)
  --mm-per-px F                 isotropic scale candidate (default: 0.07225)
  --scale-provenance LABEL      required semantics (default: unknown)
  --delimiter CHAR              submission delimiter (default: ,)
  --seed N                      ExtraTrees seed (default: 0)
  --max-train N                 ExtraTrees train cap for smoke (0=all)
  --max-rows N                  geometry_head smoke cap (0=all)
  --skip-scale-hunt             do not run hunt_physical_scale.py
  --web-notes PATH              optional organizer/EDA notes for scale hunt
  -h, --help

Example (SSH):
  export PYTHONPATH=~/kaggle_UMUD/src
  bash scripts/baselines/run_phase3_floors.sh \\
    --data-root ~/kaggle_UMUD/data/raw/umud \\
    --split-assignments ~/kaggle_UMUD/reports/phase2/split_assignments.jsonl \\
    --out-dir ~/kaggle_UMUD/reports/phase3 \\
    --mm-per-px 0.07225 \\
    --scale-provenance unknown
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --split-assignments) SPLIT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --mm-per-px) MM_PER_PX="$2"; shift 2 ;;
    --scale-provenance) SCALE_PROVENANCE="$2"; shift 2 ;;
    --delimiter) DELIMITER="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-train) MAX_TRAIN="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --skip-scale-hunt) RUN_SCALE_HUNT=0; shift ;;
    --web-notes) WEB_NOTES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${DATA_ROOT}" || -z "${SPLIT}" ]]; then
  echo "ERROR: --data-root and --split-assignments are required" >&2
  usage
  exit 1
fi

mkdir -p "${OUT_DIR}"

if [[ "${RUN_SCALE_HUNT}" -eq 1 ]]; then
  echo "== Scale hunt (no invented scale; no TIFF dpi) =="
  HUNT_ARGS=(
    --data-root "${DATA_ROOT}"
    --out "${OUT_DIR}/scale_hunt.json"
    --sample-size 40
    --seed "${SEED}"
  )
  if [[ -n "${WEB_NOTES}" ]]; then
    HUNT_ARGS+=(--web-notes "${WEB_NOTES}")
  fi
  "${PYTHON_BIN}" scripts/audits/hunt_physical_scale.py "${HUNT_ARGS[@]}"
fi

echo "== Floor A: median prior =="
"${PYTHON_BIN}" scripts/baselines/median_prior.py \
  --data-root "${DATA_ROOT}" \
  --split-assignments "${SPLIT}" \
  --out-dir "${OUT_DIR}" \
  --mm-per-px "${MM_PER_PX}" \
  --scale-provenance "${SCALE_PROVENANCE}" \
  --delimiter "${DELIMITER}"

echo "== Floor B: ExtraTrees mask-proxy =="
"${PYTHON_BIN}" scripts/baselines/extratrees_mask_proxy.py \
  --data-root "${DATA_ROOT}" \
  --split-assignments "${SPLIT}" \
  --out-dir "${OUT_DIR}" \
  --mm-per-px "${MM_PER_PX}" \
  --scale-provenance "${SCALE_PROVENANCE}" \
  --delimiter "${DELIMITER}" \
  --seed "${SEED}" \
  --max-train "${MAX_TRAIN}"

echo "== Floor C: GeometryHead mask proxy =="
"${PYTHON_BIN}" scripts/baselines/geometry_head_proxy.py \
  --data-root "${DATA_ROOT}" \
  --split-assignments "${SPLIT}" \
  --out-dir "${OUT_DIR}" \
  --mm-per-px "${MM_PER_PX}" \
  --scale-provenance "${SCALE_PROVENANCE}" \
  --delimiter "${DELIMITER}" \
  --max-rows "${MAX_ROWS}"

echo "Phase 3 floors finished under ${OUT_DIR}"
echo "Reports are Tier A / proxy only — do NOT claim leaderboard scores."
