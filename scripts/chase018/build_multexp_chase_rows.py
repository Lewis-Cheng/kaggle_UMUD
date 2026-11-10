#!/usr/bin/env python3
"""Join chase018 texture MultExp + v5 GH + ET + consensus (+ optional scratch) → multexp_chase_rows.csv."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _stem(x: str) -> str:
    x = str(x).strip()
    for suf in (".tif", ".png", ".jpg"):
        if x.lower().endswith(suf):
            return Path(x).stem
    return x


def fit_affine(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float]:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    m = np.isfinite(pred) & np.isfinite(gt)
    if m.sum() < 5:
        return 1.0, 0.0, float("nan"), float("nan")
    a, b = np.polyfit(pred[m], gt[m], 1)
    mae0 = float(np.mean(np.abs(pred[m] - gt[m])))
    mae1 = float(np.mean(np.abs(a * pred[m] + b - gt[m])))
    return float(a), float(b), mae0, mae1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--chase-csv",
        type=Path,
        default=ROOT / "reports/phase3/chase018/multexp_texture_pa_tex0.5_flsin0.25.csv",
    )
    ap.add_argument(
        "--v5-csv",
        type=Path,
        default=ROOT / "reports/phase3/multexp_v5_baseline/multexp_v5_rows.csv",
    )
    ap.add_argument(
        "--et-csv",
        type=Path,
        default=ROOT / "reports/phase3/rl_track_b/multexp_et_preds.csv",
    )
    ap.add_argument(
        "--consensus-csv",
        type=Path,
        default=ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv",
    )
    ap.add_argument("--scratch-csv", type=Path, default=None, help="multexp rows for scratch fasc+texture")
    ap.add_argument(
        "--affine-json",
        type=Path,
        default=ROOT / "reports/phase3/chase018/multexp_affine.json",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "reports/phase3/chase018/multexp_chase_rows.csv",
    )
    ap.add_argument(
        "--out-affine",
        type=Path,
        default=ROOT / "reports/phase3/chase018/multexp_chase_affine.json",
    )
    args = ap.parse_args()

    ch = pd.read_csv(args.chase_csv)
    ch["ImageID"] = ch["image_file"].map(_stem)
    v5 = pd.read_csv(args.v5_csv)
    if "ImageID" not in v5.columns and "image_file" in v5.columns:
        v5["ImageID"] = v5["image_file"].map(_stem)
    et = pd.read_csv(args.et_csv)
    cons = pd.read_csv(args.consensus_csv)

    v5 = v5.set_index("ImageID")
    et = et.set_index("ImageID")
    cons = cons.set_index("ImageID")

    # Fit PA/MT affine on chase texture preds. FL coefficients stay diagnostic.
    # The executable spec is FL identity even when the fit MAE is slightly lower.
    pa_a, pa_b, pa0, pa1 = fit_affine(ch["pred_pa"].values, ch["gt_pa"].values)
    fl_a, fl_b, fl0, fl1 = fit_affine(ch["pred_fl"].values, ch["gt_fl"].values)
    mt_a, mt_b, mt0, mt1 = fit_affine(ch["pred_mt"].values, ch["gt_mt"].values)
    affine = {
        "pa_deg": {
            "a": pa_a,
            "b": pa_b,
            "mae_before": pa0,
            "mae_after": pa1,
            "function": "affine",
            "applied": True,
        },
        "fl_mm": {
            "a": fl_a,
            "b": fl_b,
            "mae_before": fl0,
            "mae_after": fl1,
            "function": "identity",
            "applied": False,
        },
        "mt_mm": {
            "a": mt_a,
            "b": mt_b,
            "mae_before": mt0,
            "mae_after": mt1,
            "function": "affine",
            "applied": True,
        },
        "note": (
            "Executable spec: PA and MT affine, FL identity. "
            "FL coefficients are a diagnostic fit and are not applied."
        ),
    }
    # Prefer existing floor affine for PA/MT if chase fit is worse? Spec: MultExp affine on chase018.
    # Keep chase-fit. Also stash legacy floor affine path for reference.
    if args.affine_json.is_file():
        affine["legacy_floor_affine"] = json.loads(args.affine_json.read_text())

    scratch = None
    if args.scratch_csv and Path(args.scratch_csv).is_file():
        scratch = pd.read_csv(args.scratch_csv)
        scratch["ImageID"] = scratch["image_file"].map(_stem) if "image_file" in scratch.columns else scratch["ImageID"]
        scratch = scratch.set_index("ImageID")

    med_pa, med_fl, med_mt = 5.445698064952495, 50.163316237970605, 13.004999999999999
    if len(et):
        med_pa = float(et["et_pa"].median())
        med_fl = float(et["et_fl"].median())
        med_mt = float(et["et_mt"].median())
    # Prefer known train medians used by Track B
    med_pa, med_fl, med_mt = 5.445698064952495, 50.163316237970605, 13.004999999999999

    rows = []
    for _, r in ch.iterrows():
        iid = r["ImageID"]
        if iid not in v5.index or iid not in et.index or iid not in cons.index:
            print("skip missing join", iid, file=sys.stderr)
            continue
        v = v5.loc[iid]
        e = et.loc[iid]
        c = cons.loc[iid]
        ch_pa, ch_fl, ch_mt = float(r["pred_pa"]), float(r["pred_fl"]), float(r["pred_mt"])
        cal_pa = pa_a * ch_pa + pa_b
        cal_mt = mt_a * ch_mt + mt_b
        cal_fl = ch_fl
        row = {
            "ImageID": iid,
            "image_file": r.get("image_file", f"{iid}.tif"),
            "ch_pa": ch_pa,
            "ch_fl": ch_fl,
            "ch_mt": ch_mt,
            "v5_pa": float(v["pred_PA"]),
            "v5_fl": float(v["pred_FL"]),
            "v5_mt": float(v["pred_MT"]),
            "et_pa": float(e["et_pa"]),
            "et_fl": float(e["et_fl"]),
            "et_mt": float(e["et_mt"]),
            "med_pa": med_pa,
            "med_fl": med_fl,
            "med_mt": med_mt,
            "cal_pa": cal_pa,
            "cal_fl": cal_fl,
            "cal_mt": cal_mt,
            "gt_pa": float(c["median_PA"]),
            "gt_fl": float(c["median_FL"]),
            "gt_mt": float(c["median_MT"]),
            "iqr_pa": float(c.get("iqr_PA", 0.0) or 0.0),
            "iqr_fl": float(c.get("iqr_FL", 0.0) or 0.0),
            "iqr_mt": float(c.get("iqr_MT", 0.0) or 0.0),
            "apo_fg_frac": float(v.get("apo_fg_frac", 0.05) or 0.05),
            "fasc_fg_frac": float(v.get("fasc_fg_frac", 0.02) or 0.02),
            "ok": bool(r.get("ok", True)),
            "has_scratch": 0,
            "scratch_pa": np.nan,
            "scratch_fl": np.nan,
            "scratch_mt": np.nan,
        }
        if scratch is not None and iid in scratch.index:
            s = scratch.loc[iid]
            spa = float(s["pred_pa"] if "pred_pa" in s.index else s.get("pa_deg", np.nan))
            sfl = float(s["pred_fl"] if "pred_fl" in s.index else s.get("fl_mm", np.nan))
            smt = float(s["pred_mt"] if "pred_mt" in s.index else s.get("mt_mm", np.nan))
            if np.isfinite(spa):
                row.update(
                    has_scratch=1,
                    scratch_pa=spa,
                    scratch_fl=sfl,
                    scratch_mt=smt,
                )
        rows.append(row)

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    args.out_affine.write_text(json.dumps(affine, indent=2))
    print("wrote", args.out_csv, "n", len(out), "scratch", int(out.has_scratch.sum()))
    print("wrote", args.out_affine)
    print(json.dumps(affine, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
