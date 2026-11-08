#!/usr/bin/env python3
"""FL post-process sweeps on Frundles maps + fold-pure FL selector on grid-best."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.chase018.eval_promoted_hough_multexp import (  # noqa: E402
    PROMOTED,
    loocv_cal_pa_mt,
    predict_one,
)
from umud.metric.score import score_components  # noqa: E402

HK = timezone(timedelta(hours=8))
PRIOR_FLOOR = 0.1376082508388217
PRIOR_SELECTOR = 0.13550


def main() -> int:
    expert = pd.read_csv(
        ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv"
    )
    edir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"

    recipes = [
        dict(PROMOTED),
        dict(PROMOTED, name="hough_tex0.3_thr035_apo60", fasc_thr=0.35, apo_thr=0.60, tex_w=0.3),
        dict(PROMOTED, name="hough_tex0.4_thr035_apo60", fasc_thr=0.35, apo_thr=0.60, tex_w=0.4),
    ]

    def run_kw(kw):
        rows = []
        for _, er in expert.iterrows():
            name = er["image_file"]
            g = cv2.imread(str(edir / name), 0)
            z = np.load(cache / f"{name}.npz")
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
                    pred_pa=pred["pred_pa"],
                    pred_fl=pred["pred_fl"],
                    pred_mt=pred["pred_mt"],
                    gt_pa=float(er["median_PA"]),
                    gt_fl=float(er["median_FL"]),
                    gt_mt=float(er["median_MT"]),
                )
            )
        return pd.DataFrame(rows)

    results = []
    for base in recipes:
        print("BASE", base.get("name"), flush=True)
        for fl_factor in (0.85, 0.88, 0.90, 0.92, 0.95):
            for fl_sin_w in (0.0, 0.15, 0.25, 0.35):
                for agree_thr in (None, 10.0, 15.0, 25.0):
                    kw = dict(base)
                    kw.update(fl_factor=fl_factor, fl_sin_w=fl_sin_w, agree_thr=agree_thr)
                    kw["name"] = (
                        f"{base.get('name','prom')}_ff{fl_factor}_sin{fl_sin_w}_ag{agree_thr}"
                    )
                    df = run_kw(kw)
                    comps, mae = loocv_cal_pa_mt(df)
                    rec = dict(
                        name=kw["name"],
                        S=float(comps["S"]),
                        mae=mae,
                        kw={
                            k: kw[k]
                            for k in (
                                "fasc_thr",
                                "apo_thr",
                                "tex_w",
                                "fl_factor",
                                "fl_sin_w",
                                "agree_thr",
                                "pa_mode",
                            )
                        },
                    )
                    results.append(rec)
                    print(
                        f"  {rec['S']:.5f} {rec['name']} fl={mae['fl_mm']:.3f}",
                        flush=True,
                    )

    best_base = dict(
        PROMOTED, name="hough_tex0.3_thr035_apo60", fasc_thr=0.35, apo_thr=0.60, tex_w=0.3
    )
    df0 = run_kw(best_base)
    comps0, mae0 = loocv_cal_pa_mt(df0)
    print("best_base_confirm", float(comps0["S"]), mae0, flush=True)
    df0.to_csv(
        ROOT / "reports/phase3/chase018/multexp_fl_postprocess_gridbest_rows.csv",
        index=False,
    )
    for s in np.linspace(0.90, 1.10, 21):
        df = df0.copy()
        df["pred_fl"] = df0["pred_fl"] * s
        comps, mae = loocv_cal_pa_mt(df)
        results.append(
            dict(
                name=f"thr035_flscale_{s:.3f}",
                S=float(comps["S"]),
                mae=mae,
                kw=dict(fl_scale=float(s)),
            )
        )

    # multi FL-cand selector on best_base PA/MT
    cand_kws = []
    for fl_factor in (0.85, 0.90, 0.95):
        for fl_sin_w in (0.0, 0.25, 0.5):
            for agree in (None, 15.0):
                cand_kws.append(
                    dict(best_base, fl_factor=fl_factor, fl_sin_w=fl_sin_w, agree_thr=agree)
                )
    cand_kws.append(dict(best_base, fl_factor=0.9, fl_sin_w=1.0, agree_thr=None))
    cand_kws.append(dict(best_base, fl_factor=0.9, fl_sin_w=0.0, agree_thr=None))

    store = {}
    pa_mt = {}
    gt = {}
    for ci, kw in enumerate(cand_kws):
        df = run_kw(kw)
        for _, r in df.iterrows():
            store.setdefault(r.image_file, {})[ci] = float(r.pred_fl)
            pa_mt[r.image_file] = (float(r.pred_pa), float(r.pred_mt))
            gt[r.image_file] = (float(r.gt_pa), float(r.gt_fl), float(r.gt_mt))

    imgs = sorted(store)
    n = len(imgs)
    nC = len(cand_kws)
    F = np.array([[store[im][c] for c in range(nC)] for im in imgs], float)
    PA = np.array([pa_mt[im][0] for im in imgs])
    MT = np.array([pa_mt[im][1] for im in imgs])
    GPA = np.array([gt[im][0] for im in imgs])
    GFL = np.array([gt[im][1] for im in imgs])
    GMT = np.array([gt[im][2] for im in imgs])

    def score_flo(flo):
        pa_o = np.zeros(n)
        mt_o = np.zeros(n)
        for i in range(n):
            m = np.ones(n, dtype=bool)
            m[i] = False
            a, b = np.polyfit(PA[m], GPA[m], 1)
            pa_o[i] = a * PA[i] + b
            a, b = np.polyfit(MT[m], GMT[m], 1)
            mt_o[i] = a * MT[i] + b
        err = {
            "pa_deg": np.abs(pa_o - GPA),
            "fl_mm": np.abs(flo - GFL),
            "mt_mm": np.abs(mt_o - GMT),
        }
        return float(score_components(err)["S"]), {
            k: float(np.mean(v)) for k, v in err.items()
        }

    flo_oracle = np.array(
        [F[i, int(np.argmin(np.abs(F[i] - GFL[i])))] for i in range(n)]
    )
    S_oracle, mae_oracle = score_flo(flo_oracle)
    print("oracle", S_oracle, mae_oracle, flush=True)

    flo = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        j = int(np.argmin([np.mean(np.abs(F[m, c] - GFL[m])) for c in range(nC)]))
        flo[i] = F[i, j]
    S_tr, mae_tr = score_flo(flo)
    print("train_best_single", S_tr, mae_tr, flush=True)

    flo = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        order = np.argsort([np.mean(np.abs(F[m, c] - GFL[m])) for c in range(nC)])[:5]
        A = F[m][:, order]
        w = np.linalg.lstsq(A, GFL[m], rcond=None)[0]
        w = np.clip(w, 0, None)
        w = np.ones(5) / 5 if w.sum() <= 1e-9 else w / w.sum()
        flo[i] = float(F[i, order] @ w)
    S_bl, mae_bl = score_flo(flo)
    print("ols_top5_blend", S_bl, mae_bl, flush=True)

    results.sort(key=lambda r: r["S"])
    best = results[0]
    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "cache_dir": str(cache),
        "prior_floor": PRIOR_FLOOR,
        "prior_selector": PRIOR_SELECTOR,
        "grid_best": {"name": "hough_tex0.3_thr035_apo60", "S_ref": 0.13422},
        "best_base_confirm": {"S": float(comps0["S"]), "mae": mae0},
        "best_fl_post": best,
        "top20": results[:20],
        "selector": {
            "oracle": {"S": S_oracle, "MAE": mae_oracle},
            "train_best_single": {"S": S_tr, "MAE": mae_tr},
            "ols_top5_blend": {"S": S_bl, "MAE": mae_bl},
        },
        "gate_0.12": best["S"] < 0.12,
        "beats_floor": best["S"] < PRIOR_FLOOR,
        "beats_selector": best["S"] < PRIOR_SELECTOR,
        "kaggle_submit": False,
        "lab_fasc_ckpt": False,
        "protocol": "LOOCV affine PA+MT only; FL raw",
    }

    out = ROOT / "reports/phase3/chase018/multexp_fl_postprocess_ffsweep.json"
    out.write_text(json.dumps(payload, indent=2))
    print("BEST_OVERALL", best["S"], best["name"], flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
