#!/usr/bin/env python3
"""Apply chase018 SoftmaxBandit to test CSVs → submission_texture_rl.csv."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_official_submission_ids, write_submission_csv  # noqa: E402
from umud.calibration.executable import (  # noqa: E402
    CalibrationSpecError,
    require_executable_calibration,
)
from umud.rl.chase018_bandit import (  # noqa: E402
    ACTION_NAMES,
    GeometrySample,
    SoftmaxBandit,
    apply_action,
    build_state,
)


def _literal_id(x: str) -> str:
    """Keep the id string as written. Do not strip or invent a suffix."""
    return str(x).strip()


def _require_unique_ids(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    indexed = frame.set_index("image_id")
    if not indexed.index.is_unique:
        raise SystemExit(f"{name} image_id collision")
    return indexed


def _read_sub(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    delim = ";" if raw.splitlines()[0].count(";") >= raw.splitlines()[0].count(",") else ","
    df = pd.read_csv(path, sep=delim)
    cols = {c.lower().strip(): c for c in df.columns}
    id_col = cols.get("image_id") or cols.get("id") or cols.get("imageid")
    pa = cols.get("pa_deg") or cols.get("pred_pa")
    fl = cols.get("fl_mm") or cols.get("pred_fl")
    mt = cols.get("mt_mm") or cols.get("pred_mt")
    out = pd.DataFrame(
        {
            "image_id": df[id_col].map(_literal_id),
            "pa_deg": df[pa].astype(float),
            "fl_mm": df[fl].astype(float),
            "mt_mm": df[mt].astype(float),
        }
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--ch-submission", type=Path, required=True)
    ap.add_argument("--v5-submission", type=Path, required=True)
    ap.add_argument("--et-submission", type=Path, required=True)
    ap.add_argument("--scratch-submission", type=Path, default=None)
    ap.add_argument("--affine-json", type=Path, default=ROOT / "reports/phase3/chase018/multexp_chase_affine.json")
    ap.add_argument("--ids", type=Path, default=ROOT / "reports/phase3/official_test_file_names.txt")
    ap.add_argument("--median-pa", type=float, default=5.445698064952495)
    ap.add_argument("--median-fl", type=float, default=50.163316237970605)
    ap.add_argument("--median-mt", type=float, default=13.004999999999999)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-rows", type=Path, default=None)
    args = ap.parse_args()

    pol = SoftmaxBandit.load(args.policy)
    ch = _require_unique_ids(_read_sub(args.ch_submission), "ch")
    v5 = _require_unique_ids(_read_sub(args.v5_submission), "v5")
    et = _require_unique_ids(_read_sub(args.et_submission), "et")
    scratch = None
    if args.scratch_submission and args.scratch_submission.is_file():
        scratch = _require_unique_ids(_read_sub(args.scratch_submission), "scratch")

    if not args.affine_json.is_file():
        raise SystemExit(f"calibration manifest missing: {args.affine_json}")
    aff = json.loads(args.affine_json.read_text())
    try:
        coeff = require_executable_calibration(aff)
    except CalibrationSpecError as exc:
        raise SystemExit(f"calibration spec abort: {exc}") from exc
    pa_coeff = coeff["pa_deg"]
    mt_coeff = coeff["mt_mm"]
    if pa_coeff is None or mt_coeff is None or coeff["fl_mm"] is not None:
        raise SystemExit("calibration spec abort: FL must be identity and PA/MT affine")
    pa_a, pa_b = pa_coeff
    mt_a, mt_b = mt_coeff

    if not args.ids.is_file():
        raise SystemExit(f"official id list missing: {args.ids}")
    try:
        ids = load_official_submission_ids(args.ids)
    except ValueError as exc:
        raise SystemExit(f"official id abort: {exc}") from exc

    rows = []
    action_hist = {n: 0 for n in ACTION_NAMES}
    for iid in ids:
        if iid not in ch.index:
            raise SystemExit(f"official id missing from ch submission: {iid}")
        c = ch.loc[iid]
        cpa, cfl, cmt = float(c.pa_deg), float(c.fl_mm), float(c.mt_mm)
        if iid in v5.index:
            v = v5.loc[iid]
            vpa, vfl, vmt = float(v.pa_deg), float(v.fl_mm), float(v.mt_mm)
        else:
            vpa, vfl, vmt = args.median_pa, args.median_fl, args.median_mt
        if iid in et.index:
            e = et.loc[iid]
            epa, efl, emt = float(e.pa_deg), float(e.fl_mm), float(e.mt_mm)
        else:
            epa, efl, emt = args.median_pa, args.median_fl, args.median_mt
        spa = sfl = smt = None
        has_sc = 0.0
        if scratch is not None and iid in scratch.index:
            s = scratch.loc[iid]
            spa, sfl, smt = float(s.pa_deg), float(s.fl_mm), float(s.mt_mm)
            has_sc = 1.0
        sample = GeometrySample(
            image_id=iid,
            ch_pa=cpa,
            ch_fl=cfl,
            ch_mt=cmt,
            v5_pa=vpa,
            v5_fl=vfl,
            v5_mt=vmt,
            et_pa=epa,
            et_fl=efl,
            et_mt=emt,
            med_pa=args.median_pa,
            med_fl=args.median_fl,
            med_mt=args.median_mt,
            cal_pa=pa_a * cpa + pa_b,
            cal_fl=cfl,
            cal_mt=mt_a * cmt + mt_b,
            scratch_pa=spa,
            scratch_fl=sfl,
            scratch_mt=smt,
            has_scratch=has_sc,
            apo_fg_frac=0.05,
            fasc_fg_frac=0.02,
            ok_geometry=1.0,
        )
        a = pol.act(build_state(sample), greedy=True)
        pred = apply_action(sample, a)
        action_hist[ACTION_NAMES[a]] += 1
        rows.append(
            {
                "image_id": iid,
                "pa_deg": pred["pa_deg"],
                "fl_mm": pred["fl_mm"],
                "mt_mm": pred["mt_mm"],
                "action": a,
                "action_name": pred["action_name"],
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_submission_csv(args.out_csv, rows, delimiter=",", validate_against=ids)
    # ensure no BOM
    text = args.out_csv.read_text(encoding="utf-8")
    args.out_csv.write_text(text.lstrip("\ufeff"), encoding="utf-8", newline="\n")
    if args.out_rows:
        pd.DataFrame(rows).to_csv(args.out_rows, index=False)
    meta = {
        "n": len(rows),
        "action_hist": action_hist,
        "policy": str(args.policy),
        "ch_submission": str(args.ch_submission),
        "v5_submission": str(args.v5_submission),
        "et_submission": str(args.et_submission),
        "scratch_submission": str(args.scratch_submission) if args.scratch_submission else None,
        "pipeline": "chase018 texture_pa → SoftmaxBandit refine (frozen seg)",
        "claims_kaggle_lb": False,
    }
    print(json.dumps(meta, indent=2))
    args.out_csv.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
