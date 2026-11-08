#!/usr/bin/env python3
"""Blend LabTrain(/FT) fasc maps with Frundles fasc; Frundles apo; frozen LOOCV."""
from __future__ import annotations
import json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))
from adapt_multexp_fasc import FROZEN, loocv_cal_pa_mt, predict_one  # noqa: E402

HK = timezone(timedelta(hours=8))
HOLD = 0.13195144580335014
GATE = 0.12
OUT = ROOT / "reports/phase3/chase018"

def now():
    return datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z")

def eval_blend(fasc_a: Path, fasc_b: Path, apo_dir: Path, alphas, thrs, tag):
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    results = []
    best = None
    for a in alphas:
        for thr in thrs:
            kw = dict(FROZEN)
            kw["fasc_thr"] = float(thr)
            rows = []
            for _, er in expert.iterrows():
                name = er["image_file"]
                g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
                za = fasc_a / f"{name}.npz"
                zb = fasc_b / f"{name}.npz"
                zapo = apo_dir / f"{name}.npz"
                if g is None or not za.exists() or not zb.exists() or not zapo.exists():
                    continue
                fa = np.load(za)["fasc"].astype(np.float32)
                fb = np.load(zb)["fasc"].astype(np.float32)
                apo = np.load(zapo)["apo"].astype(np.float32)
                if fa.shape != fb.shape:
                    fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]), interpolation=cv2.INTER_LINEAR)
                fasc = (1.0 - a) * fa + a * fb
                if fasc.shape != apo.shape:
                    fasc = cv2.resize(fasc, (apo.shape[1], apo.shape[0]), interpolation=cv2.INTER_LINEAR)
                pred = predict_one(
                    g, apo, fasc, float(er["Scale_pixel_per_cm"]) / 10.0, kw
                )
                rows.append(
                    dict(
                        image_file=name,
                        ok=pred["ok"],
                        pred_pa=pred["pred_pa"],
                        pred_fl=pred["pred_fl"],
                        pred_mt=pred["pred_mt"],
                        gt_pa=float(er["median_PA"]),
                        gt_fl=float(er["median_FL"]),
                        gt_mt=float(er["median_MT"]),
                    )
                )
            df = pd.DataFrame(rows)
            comps, mae, *_ = loocv_cal_pa_mt(df)
            summary = {
                "tag": tag,
                "alpha_b": a,
                "thr": thr,
                "S": float(comps["S"]),
                "MAE": mae,
                "n": len(df),
                "gate": float(comps["S"]) < GATE,
                "beats_hold": float(comps["S"]) < HOLD,
            }
            results.append(summary)
            print(
                "blend tag={} a={} thr={} S={:.5f}".format(tag, a, thr, summary["S"]),
                flush=True,
            )
            if best is None or summary["S"] < best["S"]:
                best = summary
                best["_df"] = df
    return results, best

def main():
    frundles = OUT / "cache/maps_multexp"
    lab = OUT / "cache/maps_multexp_labtrain_fasc"
    ft = OUT / "cache/maps_multexp_labtrain_comp_ft"
    alphas = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 1.0]
    thrs = [0.25, 0.30, 0.35, 0.40, 0.45]
    all_res = []
    bests = []
    for tag, bdir in [("labtrain", lab), ("labtrain_comp_ft", ft)]:
        if not bdir.exists():
            print("skip missing", bdir)
            continue
        # blend: (1-a)*frundles + a*other ; apo always frundles
        res, best = eval_blend(frundles, bdir, frundles, alphas, thrs, tag)
        all_res.extend(res)
        bests.append({k: v for k, v in best.items() if k != "_df"})
        if best.get("beats_hold") and "_df" in best:
            best["_df"].to_csv(OUT / f"multexp_blend_{tag}_best_rows.csv", index=False)
    overall = min(bests, key=lambda x: x["S"]) if bests else None
    payload = {
        "ts": now(),
        "hold": HOLD,
        "gate": GATE,
        "results": [{k: v for k, v in r.items() if k != "_df"} for r in all_res],
        "best_per_source": bests,
        "best": overall,
        "promoted": bool(overall and overall["S"] < HOLD),
        "gate_pass": bool(overall and overall["S"] < GATE),
        "note": "Blend Frundles fasc with LabTrain(/FT) fasc; Frundles apo; MultExp never in train",
    }
    (OUT / "multexp_nonfrundles_fasc_blend.json").write_text(json.dumps(payload, indent=2))
    print("OVERALL", json.dumps(overall, indent=2))
    print("promoted", payload["promoted"], "gate", payload["gate_pass"])

if __name__ == "__main__":
    main()
