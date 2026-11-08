#!/usr/bin/env bash
# Run Phase 2 audits against a local UMUD data-root and freeze summaries into
# configs/protocol.yaml. Does NOT claim competition scores or mark Phase 2 passed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT=""
OUT_DIR="${ROOT}/reports/phase2"
PROTOCOL="${ROOT}/configs/protocol.yaml"
SAMPLE_SIZE=200
VAL_FRACTION=0.2
SEED=0
MAX_FILES=0
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<USAGE
Usage: bash scripts/audits/run_phase2.sh --data-root PATH [options]

Options:
  --data-root PATH     UMUD folder with apo_*/fasc_*/test_images_v2 (required)
  --out-dir PATH       JSON report directory (default: reports/phase2)
  --protocol PATH      protocol.yaml to update (default: configs/protocol.yaml)
  --sample-size N      scale-audit pair sample size (default: 200)
  --val-fraction F     duplicate-aware val fraction (default: 0.2)
  --seed N             RNG seed (default: 0)
  --max-files N        cap images hashed per folder for smoke runs (default: 0=all)
  -h, --help           show this help

Example (SSH):
  bash scripts/audits/run_phase2.sh \\
    --data-root ~/kaggle_UMUD/data/raw/umud \\
    --out-dir ~/kaggle_UMUD/reports/phase2
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --protocol) PROTOCOL="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --val-fraction) VAL_FRACTION="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-files) MAX_FILES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${DATA_ROOT}" ]]; then
  echo "ERROR: --data-root is required" >&2
  usage
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "== Phase 2 inventory =="
"${PYTHON_BIN}" scripts/audits/phase2_inventory.py \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}"

echo "== Phase 2 scale audit (PROXY only) =="
"${PYTHON_BIN}" scripts/audits/phase2_scale_audit.py \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --sample-size "${SAMPLE_SIZE}" \
  --seed "${SEED}"

echo "== Phase 2 duplicate-aware split =="
set +e
"${PYTHON_BIN}" scripts/audits/phase2_duplicate_split.py \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --val-fraction "${VAL_FRACTION}" \
  --seed "${SEED}" \
  --max-files "${MAX_FILES}"
SPLIT_RC=$?
set -e

echo "== Freeze phase2_audit into protocol.yaml =="
export PHASE2_ROOT="$ROOT"
export PHASE2_PROTOCOL="$PROTOCOL"
export PHASE2_OUT_DIR="$OUT_DIR"
export PHASE2_DATA_ROOT="$DATA_ROOT"
"${PYTHON_BIN}" <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["PHASE2_ROOT"])
sys.path.insert(0, str(root / "scripts" / "audits"))
sys.path.insert(0, str(root / "src"))
from _common import load_json, update_protocol_phase2, utc_now_iso

protocol_path = Path(os.environ["PHASE2_PROTOCOL"])
out_dir = Path(os.environ["PHASE2_OUT_DIR"])
data_root = os.environ["PHASE2_DATA_ROOT"]

inv = load_json(out_dir / "inventory.json")
scale = load_json(out_dir / "scale_audit.json")
dup = load_json(out_dir / "duplicate_split.json")

block = {
    "timestamp_utc": utc_now_iso(),
    "data_root_hint": data_root,
    "inventory_counts": inv.get("counts"),
    "stem_overlap": {
        "aponeurosis_n": inv.get("stem_overlap", {}).get("aponeurosis", {}).get("n_stem_overlap"),
        "fascicle_n": inv.get("stem_overlap", {}).get("fascicle", {}).get("n_stem_overlap"),
    },
    "sample_submission_delimiter": (inv.get("sample_submission") or {}).get("delimiter"),
    "calibration_provenance": scale.get("calibration_provenance"),
    "recommended_scale": scale.get("recommended_scale"),
    "scale_candidates_tried": scale.get("isotropic_scales_tried"),
    "scale_audit_label": scale.get("label"),
    "split_method": "sha256_exact_file + assign_duplicate_aware_split",
    "split_val_fraction": dup.get("val_fraction"),
    "split_seed": dup.get("seed"),
    "our_split_gate_pass": dup.get("our_split_gate_pass"),
    "rival_leak_content_group_count": dup.get("rival_leak_content_group_count"),
    "report_paths": {
        "inventory": str(out_dir / "inventory.json"),
        "scale_audit": str(out_dir / "scale_audit.json"),
        "duplicate_split": str(out_dir / "duplicate_split.json"),
        "split_assignments": str(out_dir / "split_assignments.jsonl"),
    },
    "status": "reports_written_not_phase2_passed",
    "note": (
        "Freeze records audit outputs only. Phase 2 is NOT marked passed here; "
        "human review of calibration_provenance and gates is required. "
        "No competition scores invented."
    ),
}
update_protocol_phase2(protocol_path, phase2_block=block)
print(f"Updated {protocol_path} phase2_audit block")
PY

if [[ "${SPLIT_RC}" -ne 0 ]]; then
  echo "WARNING: duplicate_split exited ${SPLIT_RC} (gate may have failed). Reports + protocol freeze still written." >&2
  exit "${SPLIT_RC}"
fi

echo "Phase 2 audit scripts finished. Review JSON under ${OUT_DIR} — do not treat as passed."
