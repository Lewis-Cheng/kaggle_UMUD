#!/usr/bin/env python3
"""Apply trained RL refiner to frozen v5 GH + ExtraTrees test preds → submission CSV.

Does not touch fasc_v5 weights. No test labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import write_submission_csv  # noqa: E402
from umud.rl.bandit_policy import SoftmaxBandit  # noqa: E402
from umud.rl.geometry_env import ACTION_NAMES, GeometrySample, apply_action, build_state  # noqa: E402


def _norm_id(x: str) -> str:
    x = str(x).strip()
    if x.lower().endswith(".tif") or x.lower().endswith(".png"):
        return Path(x).stem
    return x


def _read_sub(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8", errors="replace")
    delim = ";" if raw.splitlines()[0].count(";") >= raw.splitlines()[0].count(",") else ","
    df = pd.read_csv(path, sep=delim)
    # normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    id_col = cols.get("image_id") or cols.get("id")
    pa = cols.get("pa_deg")
    fl = cols.get("fl_mm")
    mt = cols.get("mt_mm")
    out = pd.DataFrame(
        {
            "image_id": df[id_col].map(_norm_id),
            "pa_deg": df[pa].astype(float),
            "fl_mm": df[fl].astype(float),
            "mt_mm": df[mt].astype(float),
        }
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--gh-submission", type=Path, required=True, help="frozen v5 GH test CSV")
    ap.add_argument("--et-submission", type=Path, required=True)
    ap.add_argument("--median-pa", type=float, default=5.445698064952495)
    ap.add_argument("--median-fl", type=float, default=50.163316237970605)
    ap.add_argument("--median-mt", type=float, default=13.004999999999999)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-rows", type=Path, default=None)
    ap.add_argument("--delimiter", type=str, default=";")
    ap.add_argument("--id-with-tif", action="store_true", help="write image_id with .tif suffix")
    args = ap.parse_args()

    pol = SoftmaxBandit.load(args.policy)
    gh = _read_sub(args.gh_submission).set_index("image_id")
    et = _read_sub(args.et_submission).set_index("image_id")
    ids = sorted(set(gh.index) & set(et.index))
    if len(ids) < 100:
        # allow union with fallback
        ids = sorted(set(gh.index) | set(et.index))
    print(f"n_test_ids={len(ids)} overlap={len(set(gh.index)&set(et.index))}")

    rows = []
    action_hist = {n: 0 for n in ACTION_NAMES}
    for iid in ids:
        if iid in gh.index:
            g = gh.loc[iid]
            gpa, gfl, gmt = float(g.pa_deg), float(g.fl_mm), float(g.mt_mm)
        else:
            gpa, gfl, gmt = args.median_pa, args.median_fl, args.median_mt
        if iid in et.index:
            e = et.loc[iid]
            epa, efl, emt = float(e.pa_deg), float(e.fl_mm), float(e.mt_mm)
        else:
            epa, efl, emt = args.median_pa, args.median_fl, args.median_mt
        # State features for test: no MultExp masks → use disagreement + magnitude proxies
        # Approximate fg fracs as mid defaults; unknown_scale=0 (AmbrosM used in GH path)
        sample = GeometrySample(
            image_id=iid,
            gh_pa=gpa,
            gh_fl=gfl,
            gh_mt=gmt,
            et_pa=epa,
            et_fl=efl,
            et_mt=emt,
            med_pa=args.median_pa,
            med_fl=args.median_fl,
            med_mt=args.median_mt,
            apo_fg_frac=0.05,
            fasc_fg_frac=0.02,
            ok_geometry=1.0,
            unknown_scale=0.0,
        )
        st = build_state(sample)
        a = pol.act(st, greedy=True)
        pred = apply_action(sample, a)
        action_hist[ACTION_NAMES[a]] += 1
        out_id = f"{iid}.tif" if args.id_with_tif else iid
        rows.append(
            {
                "image_id": out_id,
                "pa_deg": pred["pa_deg"],
                "fl_mm": pred["fl_mm"],
                "mt_mm": pred["mt_mm"],
                "action": a,
                "action_name": ACTION_NAMES[a],
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_submission_csv(args.out_csv, rows, delimiter=args.delimiter)
    if args.out_rows:
        pd.DataFrame(rows).to_csv(args.out_rows, index=False)
    meta = {
        "n": len(rows),
        "action_hist": action_hist,
        "policy": str(args.policy),
        "gh_submission": str(args.gh_submission),
        "et_submission": str(args.et_submission),
        "pipeline": "frozen_v5_GH + ExtraTrees → RL discrete refiner",
        "claims_self_eval_as_goal": False,
    }
    print(json.dumps(meta, indent=2))
    (args.out_csv.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
