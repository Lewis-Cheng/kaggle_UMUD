#!/usr/bin/env python3
"""Dump MultExp fasc with Zenodo DLTrack VGG16-BCE-512 h5 + Frundles apo; frozen eval.

Requires tensorflow. MultExp never used for training (pretrained public weights).
"""
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

HK = timezone(timedelta(hours=8))
HOLD = 0.13195144580335014
GATE = 0.12
H5 = ROOT / "data/external/osf_xbawc/labtrain_DLTrack_models/DLTrack_models/model-fasc-VGG16-BCE-512.h5"
CACHE = ROOT / "reports/phase3/chase018/cache/maps_multexp_dltrack_fasc"
APO = ROOT / "reports/phase3/chase018/cache/maps_multexp"
OUT = ROOT / "reports/phase3/chase018"

def now():
    return datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z")

def main():
    import tensorflow as tf
    print("tf", tf.__version__, flush=True)
    model = tf.keras.models.load_model(str(H5), compile=False)
    print("loaded", H5, "input", model.input_shape, flush=True)
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    CACHE.mkdir(parents=True, exist_ok=True)
    # DLTrack typically 512x512 RGB or gray expanded
    in_shape = model.input_shape
    # expect (None, 512, 512, 1) or (None, 512, 512, 3)
    th, tw = in_shape[1], in_shape[2]
    ch = in_shape[3] if len(in_shape) > 3 else 1
    n = 0
    for _, er in expert.iterrows():
        name = er["image_file"]
        g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
        zapo = APO / f"{name}.npz"
        if g is None or not zapo.exists():
            continue
        apo = np.load(zapo)["apo"].astype(np.float32)
        x = cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if ch == 3:
            x = np.stack([x, x, x], axis=-1)
        else:
            x = x[..., None]
        pred = model.predict(x[None], verbose=0)[0]
        if pred.ndim == 3:
            fasc = pred[..., 0]
        else:
            fasc = pred
        fasc = fasc.astype(np.float32)
        if fasc.max() > 1.5:  # logits
            fasc = 1.0 / (1.0 + np.exp(-fasc))
        if fasc.shape != apo.shape:
            fasc = cv2.resize(fasc, (apo.shape[1], apo.shape[0]), interpolation=cv2.INTER_LINEAR)
        np.savez_compressed(CACHE / f"{name}.npz", apo=apo.astype(np.float16), fasc=fasc.astype(np.float16))
        n += 1
    print("dumped", n, flush=True)

    thrs = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    results = []
    best = None
    best_df = None
    for thr in thrs:
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
                g, z["apo"].astype(np.float32), z["fasc"].astype(np.float32),
                float(er["Scale_pixel_per_cm"]) / 10.0, kw,
            )
            rows.append(
                dict(
                    image_file=name, ok=pred["ok"],
                    pred_pa=pred["pred_pa"], pred_fl=pred["pred_fl"], pred_mt=pred["pred_mt"],
                    gt_pa=float(er["median_PA"]), gt_fl=float(er["median_FL"]), gt_mt=float(er["median_MT"]),
                )
            )
        df = pd.DataFrame(rows)
        comps, mae, *_ = loocv_cal_pa_mt(df)
        summary = {
            "thr": thr, "S": float(comps["S"]), "MAE": mae, "n": len(df),
            "gate": float(comps["S"]) < GATE, "beats_hold": float(comps["S"]) < HOLD,
        }
        results.append(summary)
        print("dltrack thr={} S={:.5f}".format(thr, summary["S"]), flush=True)
        if best is None or summary["S"] < best["S"]:
            best, best_df = summary, df
    if best_df is not None:
        best_df.to_csv(OUT / "multexp_nonfrundles_fasc_dltrack_best_rows.csv", index=False)
    payload = {
        "ts": now(), "hold": HOLD, "gate": GATE, "h5": str(H5),
        "results": results, "best": best,
        "promoted": bool(best and best["S"] < HOLD),
        "gate_pass": bool(best and best["S"] < GATE),
        "note": "Zenodo DLTrack fasc VGG16 + Frundles apo; MultExp eval only",
    }
    (OUT / "multexp_nonfrundles_fasc_dltrack.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)

if __name__ == "__main__":
    main()
