#!/usr/bin/env python3
"""LOOCV SoftmaxBandit on chase018 MultExp rows. Gate: loocv S < ch_only S (0.17178)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.rl.chase018_bandit import (  # noqa: E402
    BanditConfig,
    GeometrySample,
    train_reinforce_loocv,
)


def rows_to_samples(df: pd.DataFrame) -> list[GeometrySample]:
    samples: list[GeometrySample] = []
    for _, r in df.iterrows():
        has_sc = float(r.get("has_scratch", 0) or 0) > 0.5
        spa = r.get("scratch_pa", np.nan)
        samples.append(
            GeometrySample(
                image_id=str(r["ImageID"]),
                ch_pa=float(r["ch_pa"]),
                ch_fl=float(r["ch_fl"]),
                ch_mt=float(r["ch_mt"]),
                v5_pa=float(r["v5_pa"]),
                v5_fl=float(r["v5_fl"]),
                v5_mt=float(r["v5_mt"]),
                et_pa=float(r["et_pa"]),
                et_fl=float(r["et_fl"]),
                et_mt=float(r["et_mt"]),
                med_pa=float(r["med_pa"]),
                med_fl=float(r["med_fl"]),
                med_mt=float(r["med_mt"]),
                cal_pa=float(r["cal_pa"]),
                cal_fl=float(r["cal_fl"]),
                cal_mt=float(r["cal_mt"]),
                scratch_pa=float(spa) if has_sc and np.isfinite(spa) else None,
                scratch_fl=float(r["scratch_fl"]) if has_sc else None,
                scratch_mt=float(r["scratch_mt"]) if has_sc else None,
                gt_pa=float(r["gt_pa"]),
                gt_fl=float(r["gt_fl"]),
                gt_mt=float(r["gt_mt"]),
                apo_fg_frac=float(r.get("apo_fg_frac", 0.05) or 0.05),
                fasc_fg_frac=float(r.get("fasc_fg_frac", 0.02) or 0.02),
                ok_geometry=1.0 if bool(r.get("ok", True)) else 0.0,
                iqr_pa=float(r.get("iqr_pa", 0.0) or 0.0),
                iqr_fl=float(r.get("iqr_fl", 0.0) or 0.0),
                iqr_mt=float(r.get("iqr_mt", 0.0) or 0.0),
                has_scratch=1.0 if has_sc else 0.0,
            )
        )
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rows-csv",
        type=Path,
        default=ROOT / "reports/phase3/chase018/multexp_chase_rows.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports/phase3/chase018/rl_refiner",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-s", type=float, default=0.17178)
    args = ap.parse_args()

    df = pd.read_csv(args.rows_csv)
    samples = rows_to_samples(df)
    print(f"n_samples={len(samples)} scratch={sum(s.has_scratch > 0.5 for s in samples)}")
    if len(samples) < 10:
        print("ERROR: too few samples", file=sys.stderr)
        return 1

    cfg = BanditConfig(epochs=args.epochs, seed=args.seed, lr=0.05, l2=0.08)
    result = train_reinforce_loocv(samples, cfg=cfg, gate_baseline_s=args.gate_s)
    policy = result.pop("policy")
    fold_rows = result.pop("fold_rows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy.save(args.out_dir / "chase018_bandit.npz")
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "loocv_fold_rows.csv", index=False)
    # also write the public MultExp gate summary
    summary = {
        "ch_only_S": result["baseline_ch_only"]["S_expert_NOT_LB"],
        "cal_pa_mt_S": result["baseline_cal_pa_mt"]["S_expert_NOT_LB"],
        "rl_loocv_S": result["loocv"]["S_expert_NOT_LB"],
        "gate_pass": result["gate_beat_ch"],
        "gate_rule": result["gate_rule"],
        "gate_baseline_s": result["gate_baseline_s"],
        "delta_vs_ch": result["loocv"]["S_expert_NOT_LB"] - result["baseline_ch_only"]["S_expert_NOT_LB"],
        "loocv_action_hist": result["loocv_action_hist"],
        "MAE_ch": {
            "PA": result["baseline_ch_only"]["MAE_PA"],
            "FL": result["baseline_ch_only"]["MAE_FL"],
            "MT": result["baseline_ch_only"]["MAE_MT"],
        },
        "MAE_rl_loocv": {
            "PA": result["loocv"]["MAE_PA"],
            "FL": result["loocv"]["MAE_FL"],
            "MT": result["loocv"]["MAE_MT"],
        },
        "action_baselines_S": {
            k: v["S_expert_NOT_LB"] for k, v in result["action_baselines"].items()
        },
    }
    (args.out_dir / "train_report.json").write_text(json.dumps(result, indent=2))
    (ROOT / "reports/phase3/chase018/multexp_rl_boost_scores.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    if result["gate_beat_ch"]:
        print("GATE PASS: LOOCV beat ch_only / texture_pa MultExp S.")
        return 0
    print("GATE FAIL: keep texture_pa as champ; do not prefer RL CSV.")
    return 0  # still success exit; gate recorded


if __name__ == "__main__":
    raise SystemExit(main())
