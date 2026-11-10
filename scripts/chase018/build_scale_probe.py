#!/usr/bin/env python3
"""One FL/MT scale probe from submission_texture_rl.csv. PA is copied unchanged."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_official_submission_ids  # noqa: E402
from umud.calibration.scale_probe import probe_factor_from_median_ratios  # noqa: E402
from umud.metric.submission import validate_submission  # noqa: E402


def _median_ratio(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(gt) & (pred > 0)
    if mask.sum() < 5:
        raise SystemExit("not enough finite positive predictions to choose a scale sign")
    return float(np.median(gt[mask] / pred[mask]))


def main() -> int:
    base_csv = ROOT / "reports/phase3/chase018/submission_texture_rl.csv"
    multiexp_csv = ROOT / "reports/phase3/chase018/multexp_texture_pa_tex0.5_flsin0.25.csv"
    affine_json = ROOT / "reports/phase3/chase018/multexp_chase_affine.json"
    official = ROOT / "reports/phase3/official_test_file_names.txt"
    ids = load_official_submission_ids(official)

    ch = pd.read_csv(multiexp_csv)
    aff = json.loads(affine_json.read_text())
    if aff["fl_mm"]["function"] != "identity" or aff["fl_mm"]["applied"] is not False:
        raise SystemExit("scale probe abort: FL manifest is not identity")
    mt_a = float(aff["mt_mm"]["a"])
    mt_b = float(aff["mt_mm"]["b"])
    fl_ratio = _median_ratio(ch["pred_fl"].to_numpy(), ch["gt_fl"].to_numpy())
    mt_pred = mt_a * ch["pred_mt"].to_numpy() + mt_b
    mt_ratio = _median_ratio(mt_pred, ch["gt_mt"].to_numpy())
    factor = probe_factor_from_median_ratios(fl_ratio, mt_ratio)

    raw = pd.read_csv(base_csv, dtype=str)
    if list(raw.columns)[:4] != ["image_id", "pa_deg", "fl_mm", "mt_mm"]:
        raise SystemExit(f"unexpected base columns: {list(raw.columns)}")
    base_ids = [str(x).strip() for x in raw["image_id"].tolist()]
    if base_ids != ids:
        raise SystemExit("texture_rl ids are not the official list in order")

    rows = []
    lines = ["image_id,pa_deg,fl_mm,mt_mm"]
    for _, row in raw.iterrows():
        image_id = str(row["image_id"]).strip()
        pa_text = str(row["pa_deg"]).strip()
        fl = float(row["fl_mm"]) * factor
        mt = float(row["mt_mm"]) * factor
        rows.append(
            {
                "image_id": image_id,
                "pa_deg": float(pa_text),
                "fl_mm": fl,
                "mt_mm": mt,
            }
        )
        lines.append(f"{image_id},{pa_text},{fl:.16g},{mt:.16g}")
    tag = f"{factor:.2f}".replace(".", "p")
    out_csv = ROOT / "reports/phase3/chase018" / f"submission_scale_probe_flmt_{tag}.csv"
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    text = out_csv.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        raise SystemExit("scale probe CSV acquired a BOM")
    check = pd.DataFrame(rows).rename(columns={"image_id": "id"})
    validate_submission(check, ids)

    meta = {
        "base_csv": str(base_csv),
        "factor": factor,
        "targets_scaled": ["fl_mm", "mt_mm"],
        "pa_deg": "copied",
        "fl_median_gt_over_pred": fl_ratio,
        "mt_median_gt_over_pred_after_affine": mt_ratio,
        "n": len(rows),
        "official_ids": str(official),
        "claims_kaggle_lb": False,
        "reference": "MultExp median sign only; not the reference model to train against",
        "note": "Single factor from the shared MultExp residual sign. Not a scale grid and not a claim that MultExp is the right anatomy.",
    }
    out_json = out_csv.with_suffix(".json")
    out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("wrote", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
