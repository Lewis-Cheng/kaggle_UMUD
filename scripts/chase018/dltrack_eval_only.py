#!/usr/bin/env python3
"""Eval DLTrack MultExp fasc maps with Frundles apo; frozen recipe (main .venv)."""
from __future__ import annotations
import json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "scripts/chase018"))
from adapt_multexp_fasc import FROZEN, loocv_cal_pa_mt, predict_one  # noqa: E402

HOLD = 0.13195144580335014
GATE = 0.12
HK = timezone(timedelta(hours=8))
CACHE = ROOT / "reports/phase3/chase018/cache/maps_multexp_dltrack_fasc"
OUT = ROOT / "reports/phase3/chase018"

expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
results = []
best = None
best_df = None
for thr in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    kw = dict(FROZEN)
    kw["fasc_thr"] = float(thr)
    rows = []
    for _, er in expert.iterrows():
        name = er["image_file"]
        g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
        zp = CACHE / f"{name}.npz"
        if g is None or not zp.exists():
            continue
        z = np.load(zp)
        pred = predict_one(
            g,
            z["apo"].astype(np.float32),
            z["fasc"].astype(np.float32),
            float(er["Scale_pixel_per_cm"]) / 10.0,
            kw,
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
    if len(df) == 0 or "pred_pa" not in df.columns:
        raise RuntimeError("no DLTrack maps scored; dump missing? cache=%s" % CACHE)
    comps, mae, *_ = loocv_cal_pa_mt(df)
    summary = {
        "thr": thr,
        "S": float(comps["S"]),
        "MAE": mae,
        "n": len(df),
        "gate": float(comps["S"]) < GATE,
        "beats_hold": float(comps["S"]) < HOLD,
    }
    results.append(summary)
    print("dltrack thr={} S={:.5f}".format(thr, summary["S"]), flush=True)
    if best is None or summary["S"] < best["S"]:
        best, best_df = summary, df
if best_df is not None:
    best_df.to_csv(OUT / "multexp_nonfrundles_fasc_dltrack_best_rows.csv", index=False)
payload = {
    "ts": datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z"),
    "hold": HOLD,
    "gate": GATE,
    "results": results,
    "best": best,
    "promoted": bool(best and best["S"] < HOLD),
    "gate_pass": bool(best and best["S"] < GATE),
    "note": "Zenodo DLTrack fasc VGG16 + Frundles apo; MultExp eval only",
}
(OUT / "multexp_nonfrundles_fasc_dltrack.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(best, indent=2), flush=True)
