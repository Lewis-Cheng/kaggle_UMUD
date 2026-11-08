#!/usr/bin/env python3
"""Honest MultExp LOOCV for promoted hough_tex03_thr032_apo60 + variants.

S_mean = mean of (MAE_PA/6, MAE_FL/12, MAE_MT/3) via score_components (S1≈S).
Uses cached apo/fasc maps. LOOCV affine on PA+MT only (FL raw).
No Kaggle submit.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.calibration.anisotropic_scale import mm_per_prob_pixel  # noqa: E402
from umud.geometry.host_faithful import (  # noqa: E402
    apo_curves,
    aponeuroses_rich,
    fascicle_segments,
    line_curve_intersection,
    pa_depth_fit,
    slope_deg,
    weighted_median,
)
from umud.geometry.texture_angle import texture_angle  # noqa: E402
from umud.metric.score import score_components  # noqa: E402

HK = timezone(timedelta(hours=8))

# Promoted floor (honest MultExp LOOCV ~0.13761 on Frundles maps)
PROMOTED = dict(
    name="hough_tex03_thr032_apo60",
    apo_thr=0.60,
    fasc_thr=0.32,
    min_len=3.0,
    depth_at=0.7,
    tex_w=0.3,
    fl_sin_w=0.25,
    fl_factor=0.9,
    agree_thr=15.0,
    tex_lo=0.5,
    tex_hi=0.95,
    pa_mode="hough",  # hough | geo
)

BAND_AGREE_V1 = dict(
    name="band_agree_v1",
    apo_thr=0.55,
    fasc_thr=0.35,
    min_len=3.0,
    depth_at=0.7,
    tex_w=0.6,
    fl_sin_w=0.25,
    fl_factor=0.9,
    agree_thr=15.0,
    tex_lo=0.5,
    tex_hi=0.95,
    pa_mode="geo",
)


def predict_one(g, apo, fasc, px_per_mm, kw):
    H0, W0 = g.shape
    h, w = apo.shape
    mx, my = mm_per_prob_pixel(px_per_mm, px_per_mm, W0, H0, w, h)
    rich = aponeuroses_rich(apo, mx, my, thr=kw["apo_thr"])
    curves = apo_curves(apo, mx, my, thr=kw["apo_thr"])
    if rich is None or len(curves) < 2:
        return dict(ok=False, pred_pa=15.0, pred_fl=70.0, pred_mt=20.0)

    th = lambda b: float(np.median(b["bot"] - b["top"]))
    lo, hi = max(rich[0]["lo"], rich[1]["lo"]), min(rich[0]["hi"], rich[1]["hi"])
    if hi - lo < 3:
        lo, hi = min(rich[0]["lo"], rich[1]["lo"]), max(rich[0]["hi"], rich[1]["hi"])
    Xs = lo + np.array([0.2, 0.5, 0.8]) * (hi - lo)
    mt = float(
        np.mean(
            np.polyval(rich[1]["fit"]["c"], Xs)
            - np.polyval(rich[0]["fit"]["c"], Xs)
            - (th(rich[0]) + th(rich[1])) / 2
        )
    )
    fang, pa_tex = texture_angle(
        g, px_per_mm, px_per_mm, rich, depth_lo=kw["tex_lo"], depth_hi=kw["tex_hi"]
    )

    # hough PA from fasc edges
    edges = cv2.Canny((fasc > kw["fasc_thr"]).astype(np.uint8) * 255, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=8)
    pas_h = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if abs(x2 - x1) + abs(y2 - y1) < 5:
                continue
            ang = (np.degrees(np.arctan2((y2 - y1) * my, (x2 - x1) * mx)) + 90) % 180 - 90
            cx = 0.5 * (x1 + x2) * mx
            cy = 0.5 * (y1 + y2) * my
            ys_sup = float(np.polyval(curves[0][0], cx))
            ys_deep = float(np.polyval(curves[1][0], cx))
            if not (ys_sup - 2 < cy < ys_deep + 2):
                continue
            pa = abs(
                ((ang - float(slope_deg(curves[1][0], np.clip(cx, curves[1][1], curves[1][2])))) + 90)
                % 180
                - 90
            )
            if 2 < pa < 60:
                pas_h.append(pa)
    pa_h = float(np.median(pas_h)) if pas_h else 15.0

    pas, fls, ws, dep = [], [], [], []
    for cx, cy, ang, L in fascicle_segments(
        fasc, mx, my, thr=kw["fasc_thr"], min_len_mm=kw["min_len"]
    ):
        ys_sup = float(np.polyval(curves[0][0], cx))
        ys_deep = float(np.polyval(curves[1][0], cx))
        if not (ys_sup - 2 < cy < ys_deep + 2):
            continue
        pa = abs(
            ((ang - float(slope_deg(curves[1][0], np.clip(cx, curves[1][1], curves[1][2])))) + 90)
            % 180
            - 90
        )
        if not (2 < pa < 60):
            continue
        p1 = line_curve_intersection(cx, cy, ang, curves[0][0])
        p2 = line_curve_intersection(cx, cy, ang, curves[1][0])
        pas.append(pa)
        ws.append(L)
        dep.append((cy - ys_sup) / (ys_deep - ys_sup) if ys_deep != ys_sup else 0.5)
        fls.append(
            float(np.linalg.norm(p1 - p2)) if p1 is not None and p2 is not None else float("nan")
        )
    segs = np.array([dep, pas, ws]).T.reshape(-1, 3) if pas else np.zeros((0, 3))
    pa_geo = weighted_median(np.asarray(pas), np.asarray(ws)) if pas else 15.0
    if len(segs) >= 5:
        v = pa_depth_fit(segs, at=kw["depth_at"])
        if v is not None:
            pa_geo = v

    if kw.get("pa_mode", "hough") == "geo":
        pa = pa_geo
    else:
        pa = pa_h
    if kw["tex_w"] > 0 and np.isfinite(pa_tex):
        pa = (1 - kw["tex_w"]) * pa + kw["tex_w"] * float(pa_tex)

    fl_geo = np.nan
    if pas:
        fls_a, ws_a = np.asarray(fls), np.asarray(ws)
        good = np.isfinite(fls_a) & (fls_a > 10) & (fls_a < 300)
        if good.any():
            fl_geo = weighted_median(fls_a[good], ws_a[good]) * kw["fl_factor"]
    if not np.isfinite(fl_geo):
        fl_geo = 70.0

    fl_tex = np.nan
    if np.isfinite(fang):
        fls_t = []
        for t in (0.3, 0.5, 0.7):
            cx = lo + t * (hi - lo)
            cy = 0.5 * (
                float(np.polyval(rich[0]["fit"]["bot"], cx))
                + float(np.polyval(rich[1]["fit"]["top"], cx))
            )
            p1 = line_curve_intersection(cx, cy, fang, curves[0][0])
            p2 = line_curve_intersection(cx, cy, fang, curves[1][0])
            if p1 is not None and p2 is not None:
                L = float(np.linalg.norm(p1 - p2))
                if 10 < L < 300:
                    fls_t.append(L)
        if fls_t:
            fl_tex = float(np.median(fls_t)) * kw["fl_factor"]

    fl_sin = mt / np.sin(np.radians(np.clip(pa, 5, 45)))
    if kw["agree_thr"] is not None and np.isfinite(fl_tex) and abs(fl_geo - fl_tex) > kw["agree_thr"]:
        fl = 0.5 * fl_geo + 0.5 * fl_sin
    else:
        fl = (1 - kw["fl_sin_w"]) * fl_geo + kw["fl_sin_w"] * fl_sin
    return dict(ok=bool(pas or pas_h), pred_pa=float(pa), pred_fl=float(fl), pred_mt=float(mt))


def loocv_cal_pa_mt(df: pd.DataFrame):
    n = len(df)
    pa, fl, mt = df.pred_pa.values, df.pred_fl.values, df.pred_mt.values
    gpa, gfl, gmt = df.gt_pa.values, df.gt_fl.values, df.gt_mt.values
    pa_o = np.zeros(n)
    mt_o = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        a, b = np.polyfit(pa[m], gpa[m], 1)
        pa_o[i] = a * pa[i] + b
        a, b = np.polyfit(mt[m], gmt[m], 1)
        mt_o[i] = a * mt[i] + b
    err = {
        "pa_deg": np.abs(pa_o - gpa),
        "fl_mm": np.abs(fl - gfl),
        "mt_mm": np.abs(mt_o - gmt),
    }
    return score_components(err), {k: float(np.mean(v)) for k, v in err.items()}


def run_recipe(name, kw, expert, expert_dir, cache_dir):
    rows = []
    for _, er in expert.iterrows():
        name_img = er["image_file"]
        g = cv2.imread(str(expert_dir / name_img), cv2.IMREAD_GRAYSCALE)
        zpath = cache_dir / f"{name_img}.npz"
        if g is None or not zpath.exists():
            continue
        z = np.load(zpath)
        pred = predict_one(
            g,
            z["apo"].astype(np.float32),
            z["fasc"].astype(np.float32),
            float(er["Scale_pixel_per_cm"]) / 10.0,
            kw,
        )
        rows.append(
            dict(
                image_file=name_img,
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
    if len(df) < 5:
        return {"name": name, "n": len(df), "error": "too few rows"}, None
    comps, mae = loocv_cal_pa_mt(df)
    raw_err = {
        "pa_deg": np.abs(df.pred_pa - df.gt_pa).to_numpy(),
        "fl_mm": np.abs(df.pred_fl - df.gt_fl).to_numpy(),
        "mt_mm": np.abs(df.pred_mt - df.gt_mt).to_numpy(),
    }
    comps_raw = score_components(raw_err)
    return {
        "name": name,
        "n": len(df),
        "ok": int(df.ok.sum()),
        "S_loocv_cal_pa_mt": float(comps["S"]),
        "S1_loocv": float(comps["S1"]),
        "S_raw": float(comps_raw["S"]),
        "MAE_loocv": mae,
        "kw": {k: v for k, v in kw.items() if k != "name"},
        "gate_0.12": float(comps["S"]) < 0.12,
        "gap_to_0.12": float(comps["S"]) - 0.12,
    }, df


def variant_grid():
    out = [dict(BAND_AGREE_V1), dict(PROMOTED)]
    # thr / apo sweeps around promoted
    for fasc_thr in (0.28, 0.30, 0.32, 0.35):
        for apo_thr in (0.55, 0.60, 0.65):
            for tex_w in (0.2, 0.3, 0.4):
                name = f"hough_tex{tex_w:.1f}_thr{int(fasc_thr*100):03d}_apo{int(apo_thr*100)}"
                kw = dict(PROMOTED)
                kw.update(name=name, fasc_thr=fasc_thr, apo_thr=apo_thr, tex_w=tex_w)
                # dedupe exact promoted
                if (
                    fasc_thr == PROMOTED["fasc_thr"]
                    and apo_thr == PROMOTED["apo_thr"]
                    and tex_w == PROMOTED["tex_w"]
                ):
                    continue
                out.append(kw)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--expert-csv",
        type=Path,
        default=ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv",
    )
    ap.add_argument(
        "--expert-dir",
        type=Path,
        default=ROOT
        / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "reports/phase3/chase018/cache/maps_multexp",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "reports/phase3/chase018/multexp_promoted_hough_scores.json",
    )
    ap.add_argument("--tag", default="", help="suffix for out files / label")
    ap.add_argument("--promoted-only", action="store_true")
    args = ap.parse_args()

    expert = pd.read_csv(args.expert_csv)
    variants = [dict(PROMOTED), dict(BAND_AGREE_V1)] if args.promoted_only else variant_grid()
    results = []
    best = None
    for kw in variants:
        name = kw["name"]
        print(f"running {name} on {args.cache_dir}", flush=True)
        summary, df = run_recipe(name, kw, expert, args.expert_dir, args.cache_dir)
        results.append(summary)
        if df is not None:
            print(
                f"  S_loocv={summary['S_loocv_cal_pa_mt']:.5f} "
                f"MAE pa={summary['MAE_loocv']['pa_deg']:.3f} "
                f"fl={summary['MAE_loocv']['fl_mm']:.3f} "
                f"mt={summary['MAE_loocv']['mt_mm']:.3f}",
                flush=True,
            )
            if best is None or summary["S_loocv_cal_pa_mt"] < best["S_loocv_cal_pa_mt"]:
                best = summary
                rows_path = args.out_json.with_name(
                    args.out_json.stem + "_best_rows.csv"
                )
                df.to_csv(rows_path, index=False)

    results_sorted = sorted(
        [r for r in results if "S_loocv_cal_pa_mt" in r],
        key=lambda r: r["S_loocv_cal_pa_mt"],
    )
    tag = args.tag or args.cache_dir.name
    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "cache_dir": str(args.cache_dir),
        "tag": tag,
        "n_variants": len(results),
        "best": best,
        "gate_0.12": bool(best and best["S_loocv_cal_pa_mt"] < 0.12),
        "prior_promoted_floor": 0.1376082508388217,
        "kaggle_submit": False,
        "results_ranked": results_sorted,
        "all_results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print("RANK (lower S better):", flush=True)
    for r in results_sorted[:12]:
        print(
            f"  {r['S_loocv_cal_pa_mt']:.5f}  {r['name']}  "
            f"pa={r['MAE_loocv']['pa_deg']:.3f} fl={r['MAE_loocv']['fl_mm']:.3f} "
            f"mt={r['MAE_loocv']['mt_mm']:.3f}",
            flush=True,
        )
    print("wrote", args.out_json, flush=True)
    if best:
        print(
            f"BEST {best['name']} S={best['S_loocv_cal_pa_mt']:.5f} "
            f"gate_0.12={best['gate_0.12']} gap={best['gap_to_0.12']:+.5f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
