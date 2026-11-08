#!/usr/bin/env python3
"""Single-contrast fault blocks. Does not retune constants or submit.

Image-level apo/fasc blocks stay blocked when maps and masks are absent.
The coordinate block reads the formal-verification witness. The measurement
block reports three aggregators on one synthetic scene and does not pick one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "reports/phase3/chase018/stage_fault_dump.json"
FV = ROOT / "reports/phase3/chase018/fv_umud_contracts.json"
AFFINE = ROOT / "reports/phase3/chase018/multexp_chase_affine.json"
IDS = ROOT / "reports/phase3/official_test_file_names.txt"
INFER = ROOT / "scripts/chase018/infer_rl_refiner.py"


def _load_snapshot():
    path = ROOT / "tests/properties/test_umud_contracts.py"
    spec = importlib.util.spec_from_file_location("fv_umud_contracts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._aggregator_snapshot()


def _suffix_risk() -> dict[str, object]:
    names = [
        Path(line.strip()).name
        for line in IDS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().endswith(".csv")
    ]
    png = [name for name in names if name.lower().endswith(".png")]
    tif = [name for name in names if name.lower().endswith(".tif")]
    # Same expression as infer_rl_refiner.main's row writer.
    rewritten = [name for name in names if not str(name).endswith(".tif")]
    source = INFER.read_text(encoding="utf-8")
    expression_present = 'f"{iid}.tif" if not str(iid).endswith(".tif")' in source
    return {
        "n_ids": len(names),
        "n_png": len(png),
        "n_tif": len(tif),
        "n_rewritten_to_tif_by_infer_expression": len(rewritten),
        "infer_expression_present": expression_present,
        "submitted_csv_present": False,
    }


def _calibration() -> dict[str, object]:
    aff = json.loads(AFFINE.read_text(encoding="utf-8"))
    raw = {"pa_deg": 20.0, "fl_mm": 70.0, "mt_mm": 20.0}
    # Deploy path in infer_rl_refiner: PA and MT affine, FL unchanged.
    deployed = {
        "pa_deg": aff["pa_deg"]["a"] * raw["pa_deg"] + aff["pa_deg"]["b"],
        "fl_mm": raw["fl_mm"],
        "mt_mm": aff["mt_mm"]["a"] * raw["mt_mm"] + aff["mt_mm"]["b"],
    }
    return {
        "example_not_a_test_image": raw,
        "deployed_cal_pa_mt": deployed,
        "affine_file_fl_applied": aff["fl_mm"]["applied"],
        "deploy_path_applies_fl_affine": False,
        "note": (
            "multexp_chase_affine.json marks fl_mm applied=true, but "
            "infer_rl_refiner sets cal_fl to the raw fascicle length. "
            "No new affine was fit. Test-image residuals are unavailable."
        ),
    }


def main() -> int:
    fv = json.loads(FV.read_text(encoding="utf-8"))
    pullback = fv["integration_coordinate_pullback"]
    snapshot = _load_snapshot()
    suffix = _suffix_risk()
    calibration = _calibration()
    blocks = [
        {
            "block": "scale_anchor",
            "hypothesis": "Millimetre scale is not identified per image.",
            "contrast": "measured vs family_prior vs unknown",
            "evidence": {
                "protocol_counts": {"measured": 2318, "family_prior": 427, "unknown": 1373},
                "per_image_rows_present": False,
                "q_invented": False,
            },
            "verdict": "inconclusive",
            "reason": "Scale-row CSV is not in this workspace. Counts are the v2 audit summary, not a new measurement.",
        },
        {
            "block": "coordinate_pullback",
            "hypothesis": "Resize into 384x576 drops the spacing Jacobian.",
            "contrast": "synthetic scene before vs after INTER_AREA, spacing scaled by pixel ratio",
            "evidence": {
                "verdict": pullback["verdict"],
                "abs_delta": pullback["abs_delta"],
                "tolerance": pullback["tolerance"],
                "resize": pullback.get("resize"),
            },
            "verdict": pullback["verdict"],
            "reason": pullback["note"],
        },
        {
            "block": "apo_geometry",
            "hypothesis": "Predicted aponeurosis curves differ from manual apo masks.",
            "contrast": "GT apo vs predicted apo, fasc frozen",
            "evidence": {"images_present": False, "checkpoints_present": False},
            "verdict": "blocked_missing_inputs",
            "reason": "No competition images, masks, or apo checkpoint in this workspace.",
        },
        {
            "block": "fasc_orientation",
            "hypothesis": "Mask overlap can be high while fascicle direction is wrong.",
            "contrast": "GT fasc direction vs predicted direction, apo frozen",
            "evidence": {"images_present": False, "dice_used_as_decision": False},
            "verdict": "blocked_missing_inputs",
            "reason": "No fascicle images or masks in this workspace. Dice was not computed.",
        },
        {
            "block": "measurement_definition",
            "hypothesis": "The aggregator, not the map, changes pennation.",
            "contrast": "weighted median vs unweighted median vs depth-fit on one synthetic scene",
            "evidence": snapshot,
            "verdict": "synthetic_contrast_not_a_selection",
            "reason": "Three readouts are reported. None is selected. This is not a leaderboard score.",
        },
        {
            "block": "calibration_transfer",
            "hypothesis": "The frozen PA+MT affine does not transfer.",
            "contrast": "raw example vs deployed cal_pa_mt; FL affine in the JSON vs FL raw in infer",
            "evidence": calibration,
            "verdict": "inconclusive",
            "reason": "The affine file and the deploy path disagree on FL. Test predictions are absent, so no domain verdict.",
        },
        {
            "block": "deployment_parity",
            "hypothesis": "The CSV writer renames png ids to tif.",
            "contrast": "official test basenames vs infer_rl_refiner id expression",
            "evidence": suffix,
            "verdict": "inconclusive",
            "reason": (
                "The infer expression would rewrite every non-.tif id. "
                f"{suffix['n_rewritten_to_tif_by_infer_expression']} official ids do not end in .tif. "
                "The submitted CSV is not here, so the public file is not shown to contain that rewrite."
            ),
        },
    ]
    payload = {
        "constants_retuned": False,
        "kaggle_submit": False,
        "multiexp_used_to_select": False,
        "blocks": blocks,
    }
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({row["block"]: row["verdict"] for row in blocks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
