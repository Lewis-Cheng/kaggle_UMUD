#!/usr/bin/env python3
"""Train discrete RL geometry refiner on MultExpData (LOOCV gate).

Inputs: multexp_v5_rows.csv (frozen v5 GH) + multexp_et_preds.csv + expert_consensus.csv
Does NOT update seg_fasc_v5 weights. Reward = S_expert_NOT_LB.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.rl.bandit_policy import BanditConfig, train_reinforce_loocv  # noqa: E402
from umud.rl.geometry_env import GeometrySample  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multexp-gh-csv", type=Path, required=True)
    ap.add_argument("--multexp-et-csv", type=Path, required=True)
    ap.add_argument("--consensus-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gh = pd.read_csv(args.multexp_gh_csv)
    et = pd.read_csv(args.multexp_et_csv)
    cons = pd.read_csv(args.consensus_csv)
    et = et.set_index("ImageID")
    cons = cons.set_index("ImageID")

    # Global medians from ExtraTrees train report / ET file meta fallback
    med_pa = float(et["et_pa"].median())
    med_fl = float(et["et_fl"].median())
    med_mt = float(et["et_mt"].median())
    # Prefer known train medians from phase3 ET report if present
    med_pa, med_fl, med_mt = 5.445698064952495, 50.163316237970605, 13.004999999999999

    samples: list[GeometrySample] = []
    for _, row in gh.iterrows():
        iid = str(row["ImageID"])
        if iid not in et.index or iid not in cons.index:
            continue
        e = et.loc[iid]
        c = cons.loc[iid]
        samples.append(
            GeometrySample(
                image_id=iid,
                gh_pa=float(row["pred_PA"]),
                gh_fl=float(row["pred_FL"]),
                gh_mt=float(row["pred_MT"]),
                et_pa=float(e["et_pa"]),
                et_fl=float(e["et_fl"]),
                et_mt=float(e["et_mt"]),
                med_pa=med_pa,
                med_fl=med_fl,
                med_mt=med_mt,
                gt_pa=float(c["median_PA"]),
                gt_fl=float(c["median_FL"]),
                gt_mt=float(c["median_MT"]),
                apo_fg_frac=float(row.get("apo_fg_frac", 0.0)),
                fasc_fg_frac=float(row.get("fasc_fg_frac", 0.0)),
                ok_geometry=1.0 if bool(row.get("ok", True)) else 0.0,
                unknown_scale=0.0,
                iqr_pa=float(c.get("iqr_PA", 0.0)),
                iqr_fl=float(c.get("iqr_FL", 0.0)),
                iqr_mt=float(c.get("iqr_MT", 0.0)),
            )
        )

    print(f"n_samples={len(samples)}")
    if len(samples) < 10:
        print("ERROR: too few MultExpData samples", file=sys.stderr)
        return 1

    cfg = BanditConfig(epochs=args.epochs, seed=args.seed, lr=0.05, l2=0.08)
    result = train_reinforce_loocv(samples, cfg=cfg)
    policy = result.pop("policy")
    fold_rows = result.pop("fold_rows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy.save(args.out_dir / "geometry_bandit.npz")
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "loocv_fold_rows.csv", index=False)
    (args.out_dir / "train_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in result if k != "fold_rows"}, indent=2))
    if not result["gate_beat_gh"]:
        print("GATE FAIL: LOOCV did not beat GH-only; still saving policy for inspection.")
    else:
        print("GATE PASS: LOOCV beat frozen GH-only on MultExpData.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
