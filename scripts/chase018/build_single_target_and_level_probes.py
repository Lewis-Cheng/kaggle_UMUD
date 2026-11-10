#!/usr/bin/env python3
"""Single-target splits of v2, plus one-parameter FL and MT level probes.

S is additive across targets, so each file changes one column and copies the
other two from texture_rl. The FL-only v2 file is the submission to try first.
Level probes use the existing FL x0.9 geometry as the knob and never scale
FL and MT together.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from umud.baselines.io import load_official_submission_ids, write_submission_csv

ROOT = Path(__file__).resolve().parents[2]
CHASE = ROOT / "reports/phase3/chase018"
IDS = load_official_submission_ids(ROOT / "reports/phase3/official_test_file_names.txt")
RL = pd.read_csv(CHASE / "submission_texture_rl.csv").set_index("image_id").loc[IDS]
V2 = pd.read_csv(CHASE / "submission_ref_geometry_v2.csv").set_index("image_id").loc[IDS]
OUT_DIR = CHASE / "probes_016"
# Current texture_rl FL already includes the geometry factor 0.9.
BAKED_FL_FACTOR = 0.9


def write_rows(name: str, pa, fl, mt, note: str) -> dict:
    rows = [
        {"image_id": i, "pa_deg": float(p), "fl_mm": float(f), "mt_mm": float(m)}
        for i, p, f, m in zip(IDS, pa, fl, mt)
    ]
    path = OUT_DIR / name
    write_submission_csv(path, rows, validate_against=IDS)
    frame = pd.DataFrame(rows).set_index("image_id").loc[IDS]
    meta = {
        "file": str(path),
        "note": note,
        "mae_vs_texture_rl": {
            "pa_deg": float(np.abs(frame.pa_deg - RL.pa_deg).mean()),
            "fl_mm": float(np.abs(frame.fl_mm - RL.fl_mm).mean()),
            "mt_mm": float(np.abs(frame.mt_mm - RL.mt_mm).mean()),
        },
        "median_ratio_vs_texture_rl": {
            "fl_mm": float((frame.fl_mm / RL.fl_mm).median()),
            "mt_mm": float((frame.mt_mm / RL.mt_mm).median()),
        },
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    # One column from v2, the other two copied from the public winner.
    for column, label in (
        ("fl_mm", "FL-only v2. Tests whether the new FL ordering is more concordant than texture_rl. Submit this first."),
        ("pa_deg", "PA-only v2. Do not submit until the FL-only score is back."),
        ("mt_mm", "MT-only v2. Do not submit until the FL-only score is back."),
    ):
        pa = V2.pa_deg.to_numpy() if column == "pa_deg" else RL.pa_deg.to_numpy()
        fl = V2.fl_mm.to_numpy() if column == "fl_mm" else RL.fl_mm.to_numpy()
        mt = V2.mt_mm.to_numpy() if column == "mt_mm" else RL.mt_mm.to_numpy()
        written.append(write_rows(f"submission_v2_{column}_only.csv", pa, fl, mt, label))

    # Absolute geometry factors around the baked 0.9. Each file scales FL only.
    for factor in (0.85, 0.95, 1.0):
        scale = factor / BAKED_FL_FACTOR
        written.append(
            write_rows(
                f"submission_fl_factor_{factor:.2f}.csv",
                RL.pa_deg.to_numpy(),
                RL.fl_mm.to_numpy() * scale,
                RL.mt_mm.to_numpy(),
                f"FL-only level. Absolute geometry factor {factor} (texture_rl is 0.90). MT and PA copied.",
            )
        )
    # MT-only level. These do not touch FL.
    for factor in (0.98, 1.02, 1.05):
        written.append(
            write_rows(
                f"submission_mt_factor_{factor:.2f}.csv",
                RL.pa_deg.to_numpy(),
                RL.fl_mm.to_numpy(),
                np.clip(RL.mt_mm.to_numpy() * factor, 10.0, 50.0),
                f"MT-only level x{factor}. FL and PA copied from texture_rl.",
            )
        )
    index = {
        "submit_first": str(OUT_DIR / "submission_v2_fl_mm_only.csv"),
        "do_not_submit_together": "One file per public score. FL-only v2 is the attribution test.",
        "files": written,
    }
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"submit_first": index["submit_first"], "n_files": len(written)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
