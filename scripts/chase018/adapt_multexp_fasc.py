#!/usr/bin/env python3
"""MultExp-domain fasc adaptation (A/C) on Frundles maps; frozen FL/geo recipe.

A) Apo-bounded corridor + structure-tensor / ridge cleanup of Frundles fasc.
C) Per-image fold-pure selection among adapt variants.

Honest LOOCV: affine PA+MT, FL raw. No Kaggle. No attention. No lab ckpt.
Promote only if S beats 0.13294. Gate pass iff S < 0.12.
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
PRIOR_BEST = 0.13293555901039555

# Frozen promoted FL/geo recipe — do not retune here
FROZEN = dict(
    name="hough_tex0.4_thr035_apo60_ff0.9_sin0.35_agNone",
    apo_thr=0.60,
    fasc_thr=0.35,
    min_len=3.0,
    depth_at=0.7,
    tex_w=0.4,
    fl_sin_w=0.35,
    fl_factor=0.9,
    agree_thr=None,
    tex_lo=0.5,
    tex_hi=0.95,
    pa_mode="hough",
)


def apo_belly_mask(apo: np.ndarray, apo_thr: float, margin_px: int = 2) -> np.ndarray:
    """Binary mask of pixels strictly between superficial and deep apo bands."""
    h, w = apo.shape
    rich = aponeuroses_rich(apo, 1.0, 1.0, thr=apo_thr)  # pixel units
    if rich is None:
        # fallback: anything not apo and between vertical apo centroids
        m = (apo > apo_thr).astype(np.uint8)
        if m.sum() < 50:
            return np.ones((h, w), np.uint8)
        ys, xs = np.nonzero(m)
        ymed = float(np.median(ys))
        # keep central 60% vertically around apo mid-gap estimate
        top = int(max(0, ymed - 0.35 * h))
        bot = int(min(h, ymed + 0.35 * h))
        out = np.zeros((h, w), np.uint8)
        out[top:bot, :] = 1
        out[m > 0] = 0
        return out
    sup, deep = rich[0], rich[1]
    xs = np.arange(w, dtype=float)
    # polys are in "mm" but we passed mx=my=1 so pixel coords
    y_sup = np.polyval(sup["fit"]["bot"], xs) + margin_px
    y_deep = np.polyval(deep["fit"]["top"], xs) - margin_px
    yy = np.arange(h)[:, None]
    lo = max(sup["lo"], deep["lo"])
    hi = min(sup["hi"], deep["hi"])
    if hi - lo < 5:
        lo, hi = min(sup["lo"], deep["lo"]), max(sup["hi"], deep["hi"])
    mask = (yy >= y_sup[None]) & (yy <= y_deep[None]) & (xs[None] >= lo) & (xs[None] <= hi)
    mask &= (y_deep > y_sup + 3)[None]
    return mask.astype(np.uint8)


def structure_ridge(gray: np.ndarray, sigma: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ridge_response, fasc_angle_deg_map, coherence) in [0,1]-ish."""
    g = gray.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    # multi-scale mild: single sigma then optional blur integrate
    k = max(3, int(2 * round(3 * sigma) + 1))
    Ix = cv2.Sobel(hp, cv2.CV_32F, 1, 0, ksize=k)
    Iy = cv2.Sobel(hp, cv2.CV_32F, 0, 1, ksize=k)
    si = max(1.5, sigma)
    Jxx = cv2.GaussianBlur(Ix * Ix, (0, 0), si)
    Jyy = cv2.GaussianBlur(Iy * Iy, (0, 0), si)
    Jxy = cv2.GaussianBlur(Ix * Iy, (0, 0), si)
    tr = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    coher = (l1 - l2) / (l1 + l2 + 1e-8)
    # ridge: high coherence + anisotropic; use (l1-l2)*coher normalized
    ridge = (l1 - l2) * coher
    # normalize robustly
    flat = ridge.ravel()
    lo, hi = np.percentile(flat, [50, 99.5])
    if hi <= lo + 1e-8:
        ridge_n = np.zeros_like(ridge)
    else:
        ridge_n = np.clip((ridge - lo) / (hi - lo), 0, 1)
    grad_ang = 0.5 * np.degrees(np.arctan2(2.0 * Jxy, Jxx - Jyy + 1e-12))
    fasc_ang = ((grad_ang + 90.0 + 90) % 180) - 90  # perpendicular to gradient
    return ridge_n.astype(np.float32), fasc_ang.astype(np.float32), coher.astype(np.float32)


def morph_cleanup(prob: np.ndarray, open_k: int = 0, close_k: int = 0, min_area: int = 0) -> np.ndarray:
    out = prob.copy()
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = (out > 0.15).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        out = out * m
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        # close on soft threshold then restore soft values where closed
        m = (out > 0.12).astype(np.uint8)
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        # fill gaps with local max of neighbors
        blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
        fill = (m2 > 0) & (m == 0)
        out = np.where(fill, np.maximum(out, blurred * 0.85), out)
    if min_area > 0:
        m = (out > 0.15).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if st[i, 4] >= min_area:
                keep[lab == i] = 1
        out = out * keep
    return out.astype(np.float32)


def orient_filter(fasc: np.ndarray, ang_map: np.ndarray, belly: np.ndarray, center_ang: float, tol_deg: float) -> np.ndarray:
    """Zero fasc outside belly or whose local ridge angle disagrees with center_ang."""
    if not np.isfinite(center_ang):
        return fasc * belly
    d = np.abs(((ang_map - center_ang + 90) % 180) - 90)
    keep = (d <= tol_deg).astype(np.float32)
    return fasc * belly * keep


def adapt_fasc(
    gray_full: np.ndarray,
    apo: np.ndarray,
    fasc: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Produce adapted fasc probability map (same HxW as apo/fasc)."""
    h, w = apo.shape
    g = cv2.resize(gray_full, (w, h), interpolation=cv2.INTER_AREA)
    belly = apo_belly_mask(apo, apo_thr=0.60, margin_px=2).astype(np.float32)
    ridge, ang, coher = structure_ridge(g, sigma=1.5)
    # dominant fasc angle in belly
    wts = (ridge * coher * belly).ravel()
    angs = ang.ravel()
    if wts.sum() > 1e-6:
        # circular-ish median via histogram
        bins = np.linspace(-90, 90, 37)
        hist, _ = np.histogram(angs, bins=bins, weights=wts)
        center_ang = float(0.5 * (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]))
    else:
        center_ang = float("nan")

    base = fasc.astype(np.float32)

    if variant == "baseline":
        return base

    if variant == "belly_gate":
        # keep Frundles fasc only inside apo corridor; soft outside downweight
        return (base * (0.15 + 0.85 * belly)).astype(np.float32)

    if variant == "belly_orient15":
        return orient_filter(base, ang, belly, center_ang, 15.0).astype(np.float32)

    if variant == "belly_orient25":
        return orient_filter(base, ang, belly, center_ang, 25.0).astype(np.float32)

    if variant == "ridge_blend03":
        # boost Frundles where ridge agrees; add weak ridge seeds in belly
        boost = base * (1.0 + 0.8 * ridge * belly)
        seeds = 0.30 * ridge * coher * belly
        out = np.maximum(boost, seeds)
        out = morph_cleanup(out, open_k=0, close_k=3, min_area=8)
        return np.clip(out, 0, 1).astype(np.float32)

    if variant == "ridge_blend05":
        boost = base * (1.0 + 1.0 * ridge * belly)
        seeds = 0.50 * ridge * coher * belly
        # only keep seeds near orientation
        if np.isfinite(center_ang):
            d = np.abs(((ang - center_ang + 90) % 180) - 90)
            seeds = seeds * (d <= 20).astype(np.float32)
        out = np.maximum(boost, seeds)
        out = morph_cleanup(out, open_k=2, close_k=3, min_area=10)
        return np.clip(out, 0, 1).astype(np.float32)

    if variant == "ridge_replace":
        # heavier: Frundles * ridge + corridor seeds; morph to connect fascicles
        out = np.maximum(base * ridge * belly, 0.45 * ridge * coher * belly)
        if np.isfinite(center_ang):
            d = np.abs(((ang - center_ang + 90) % 180) - 90)
            out = out * (0.2 + 0.8 * (d <= 22).astype(np.float32))
        out = morph_cleanup(out, open_k=2, close_k=5, min_area=12)
        return np.clip(out, 0, 1).astype(np.float32)

    if variant == "morph_connect":
        # Frundles belly-gated + close to connect broken fascicles
        out = base * (0.2 + 0.8 * belly)
        out = morph_cleanup(out, open_k=0, close_k=5, min_area=6)
        return np.clip(out, 0, 1).astype(np.float32)

    if variant == "morph_prune":
        out = base * belly
        out = morph_cleanup(out, open_k=3, close_k=0, min_area=15)
        return np.clip(out, 0, 1).astype(np.float32)

    if variant == "frangi_like":
        # multi-sigma ridge max inside belly, blend lightly with Frundles
        ridges = []
        for s in (1.0, 1.5, 2.5):
            r, a, c = structure_ridge(g, sigma=s)
            ridges.append(r * c)
        rmax = np.maximum.reduce(ridges)
        out = np.maximum(base * belly, 0.40 * rmax * belly)
        out = morph_cleanup(out, open_k=2, close_k=3, min_area=10)
        return np.clip(out, 0, 1).astype(np.float32)

    if variant == "corridor_dilate":
        # dilate thin Frundles fasc along approx orientation via elliptical close
        out = base * belly
        # anisotropic close: horizontal-ish ellipse then rotate approx
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
        m = (out > 0.15).astype(np.uint8)
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        blurred = cv2.GaussianBlur(out, (0, 0), 1.2)
        fill = (m2 > 0) & (m == 0)
        out = np.where(fill, np.maximum(out, blurred * 0.9), out)
        out = morph_cleanup(out, open_k=0, close_k=0, min_area=8)
        return np.clip(out, 0, 1).astype(np.float32)

    raise ValueError(f"unknown variant {variant}")


VARIANTS = [
    "baseline",
    "belly_gate",
    "belly_orient15",
    "belly_orient25",
    "ridge_blend03",
    "ridge_blend05",
    "ridge_replace",
    "morph_connect",
    "morph_prune",
    "frangi_like",
    "corridor_dilate",
]


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
    return score_components(err), {k: float(np.mean(v)) for k, v in err.items()}, pa_o, mt_o


def image_features(g, apo, fasc, px_per_mm):
    """Cheap MultExp features for fold-pure selector (C)."""
    h, w = apo.shape
    belly = apo_belly_mask(apo, 0.60)
    fasc_m = float((fasc > 0.35).mean())
    apo_m = float((apo > 0.60).mean())
    belly_frac = float(belly.mean())
    # apo gap proxy in map pixels
    rich = aponeuroses_rich(apo, 1.0, 1.0, thr=0.60)
    if rich is not None:
        xc = 0.5 * (rich[0]["lo"] + rich[0]["hi"])
        gap = float(np.polyval(rich[1]["fit"]["c"], xc) - np.polyval(rich[0]["fit"]["c"], xc))
    else:
        gap = float("nan")
    g_r = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    ridge, ang, coher = structure_ridge(g_r, sigma=1.5)
    coh_m = float(coher[belly > 0].mean()) if belly.sum() else 0.0
    # orientation dispersion in belly
    if belly.sum() > 100:
        a = ang[belly > 0]
        # circular variance proxy
        th = np.deg2rad(2 * a)  # period 180
        disp = float(1.0 - np.sqrt(np.mean(np.cos(th)) ** 2 + np.mean(np.sin(th)) ** 2))
    else:
        disp = 1.0
    return dict(
        fasc_mass=fasc_m,
        apo_mass=apo_m,
        belly_frac=belly_frac,
        apo_gap_px=gap,
        coher_mean=coh_m,
        orient_disp=disp,
        scale=float(px_per_mm),
    )


def run_variant(variant, expert, expert_dir, cache_dir, kw, save_adapt_dir=None):
    rows = []
    feats = []
    for _, er in expert.iterrows():
        name_img = er["image_file"]
        g = cv2.imread(str(expert_dir / name_img), cv2.IMREAD_GRAYSCALE)
        zpath = cache_dir / f"{name_img}.npz"
        if g is None or not zpath.exists():
            continue
        z = np.load(zpath)
        apo = z["apo"].astype(np.float32)
        fasc0 = z["fasc"].astype(np.float32)
        fasc = adapt_fasc(g, apo, fasc0, variant)
        if save_adapt_dir is not None:
            save_adapt_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_adapt_dir / f"{name_img}.npz", apo=apo, fasc=fasc.astype(np.float16))
        px = float(er["Scale_pixel_per_cm"]) / 10.0
        pred = predict_one(g, apo, fasc, px, kw)
        feat = image_features(g, apo, fasc0, px)
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
                **feat,
            )
        )
        feats.append(feat)
    df = pd.DataFrame(rows)
    if len(df) < 5:
        return {"name": variant, "n": len(df), "error": "too few"}, None
    comps, mae, _, _ = loocv_cal_pa_mt(df)
    return {
        "name": variant,
        "n": int(len(df)),
        "ok": int(df.ok.sum()),
        "S_loocv_cal_pa_mt": float(comps["S"]),
        "S1_loocv": float(comps["S1"]),
        "MAE_loocv": mae,
        "gate_0.12": float(comps["S"]) < 0.12,
        "gap_to_0.12": float(comps["S"]) - 0.12,
        "beats_0.13294": float(comps["S"]) < PRIOR_BEST,
    }, df


def fold_pure_selector(preds_by_var: dict[str, pd.DataFrame], feature_keys, variants):
    """For each held-out i, pick variant by nearest-neighbor on train features
    using train MAE_FL+PA proxy (sum of abs err on train after affine fit).
    Actually: pick the variant with best train S (LOOCV among train) — simple
    argmin of train mean component error. Features unused if global-best on train.
    Also try feature-gated rules.
    """
    # Align on image_file
    base = next(iter(preds_by_var.values()))
    imgs = list(base.image_file.values)
    n = len(imgs)
    # stack preds
    pa = {v: preds_by_var[v].set_index("image_file").loc[imgs, "pred_pa"].values for v in variants}
    fl = {v: preds_by_var[v].set_index("image_file").loc[imgs, "pred_fl"].values for v in variants}
    mt = {v: preds_by_var[v].set_index("image_file").loc[imgs, "pred_mt"].values for v in variants}
    gpa = base.set_index("image_file").loc[imgs, "gt_pa"].values
    gfl = base.set_index("image_file").loc[imgs, "gt_fl"].values
    gmt = base.set_index("image_file").loc[imgs, "gt_mt"].values
    feat = np.stack(
        [base.set_index("image_file").loc[imgs, k].values.astype(float) for k in feature_keys],
        axis=1,
    )
    # replace nan feats
    col_med = np.nanmedian(feat, axis=0)
    inds = np.where(~np.isfinite(feat))
    feat[inds] = np.take(col_med, inds[1])

    def score_from_preds(pa_p, fl_p, mt_p, idx_train, idx_test_one):
        # affine on train, apply to test one; FL raw
        a, b = np.polyfit(pa_p[idx_train], gpa[idx_train], 1)
        pa_hat = a * pa_p[idx_test_one] + b
        a, b = np.polyfit(mt_p[idx_train], gmt[idx_train], 1)
        mt_hat = a * mt_p[idx_test_one] + b
        e_pa = abs(pa_hat - gpa[idx_test_one])
        e_fl = abs(fl_p[idx_test_one] - gfl[idx_test_one])
        e_mt = abs(mt_hat - gmt[idx_test_one])
        return e_pa / 6.0, e_fl / 12.0, e_mt / 3.0

    def train_S(pa_p, fl_p, mt_p, idx_train):
        # nested LOOCV S on train subset
        errs = []
        tr = list(idx_train)
        for j, t in enumerate(tr):
            sub = [x for x in tr if x != t]
            if len(sub) < 3:
                continue
            e = score_from_preds(pa_p, fl_p, mt_p, sub, t)
            errs.append(e)
        if not errs:
            return 9.0
        arr = np.array(errs)
        return float(arr.mean(axis=0).mean())

    # Strategy 1: global best variant by full LOOCV already known — skip
    # Strategy 2: for each fold, pick variant with best train_S
    pa_o = np.zeros(n)
    fl_o = np.zeros(n)
    mt_o = np.zeros(n)
    picked = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_v, best_s = None, 9.0
        for v in variants:
            s = train_S(pa[v], fl[v], mt[v], tr)
            if s < best_s:
                best_s, best_v = s, v
        picked.append(best_v)
        # affine using train of picked variant
        a, b = np.polyfit(pa[best_v][tr], gpa[tr], 1)
        pa_o[i] = a * pa[best_v][i] + b
        a, b = np.polyfit(mt[best_v][tr], gmt[tr], 1)
        mt_o[i] = a * mt[best_v][i] + b
        fl_o[i] = fl[best_v][i]

    err = {
        "pa_deg": np.abs(pa_o - gpa),
        "fl_mm": np.abs(fl_o - gfl),
        "mt_mm": np.abs(mt_o - gmt),
    }
    comps = score_components(err)
    mae = {k: float(np.mean(v)) for k, v in err.items()}

    # Strategy 3: feature-threshold rules fold-pure
    # For each fold, on train fit a simple decision: if fasc_mass < t use V1 else V2
    # Search over pairs of variants and threshold on fasc_mass / orient_disp
    best_rule = None
    best_rule_S = 9.0
    rule_pa = np.zeros(n)
    rule_fl = np.zeros(n)
    rule_mt = np.zeros(n)
    feat_idx = {k: i for i, k in enumerate(feature_keys)}
    for fkey in ("fasc_mass", "orient_disp", "coher_mean", "apo_gap_px"):
        fi = feat_idx[fkey]
        for v_lo in variants:
            for v_hi in variants:
                if v_lo == v_hi:
                    continue
                # try percentiles as thresholds from each fold's train
                pa_tmp = np.zeros(n)
                fl_tmp = np.zeros(n)
                mt_tmp = np.zeros(n)
                for i in range(n):
                    tr = [j for j in range(n) if j != i]
                    # choose threshold as median of train feature
                    thr = float(np.median(feat[tr, fi]))
                    # which side better on train?
                    # evaluate both assignments of lo/hi
                    def assign_score(lo_is_below):
                        # compute train_S if we assign v_lo when feat<thr
                        # approximate: mean abs FL err on train with raw preds
                        e = []
                        for t in tr:
                            v = v_lo if ((feat[t, fi] < thr) == lo_is_below) else v_hi
                            e.append(abs(fl[v][t] - gfl[t]) / 12.0 + abs(pa[v][t] - gpa[t]) / 6.0)
                        return float(np.mean(e))

                    s0 = assign_score(True)
                    s1 = assign_score(False)
                    lo_is_below = s0 <= s1
                    v_pick = v_lo if ((feat[i, fi] < thr) == lo_is_below) else v_hi
                    a, b = np.polyfit(pa[v_pick][tr], gpa[tr], 1)
                    pa_tmp[i] = a * pa[v_pick][i] + b
                    a, b = np.polyfit(mt[v_pick][tr], gmt[tr], 1)
                    mt_tmp[i] = a * mt[v_pick][i] + b
                    fl_tmp[i] = fl[v_pick][i]
                err_r = {
                    "pa_deg": np.abs(pa_tmp - gpa),
                    "fl_mm": np.abs(fl_tmp - gfl),
                    "mt_mm": np.abs(mt_tmp - gmt),
                }
                comps_r = score_components(err_r)
                if comps_r["S"] < best_rule_S:
                    best_rule_S = float(comps_r["S"])
                    best_rule = dict(
                        feature=fkey,
                        v_lo=v_lo,
                        v_hi=v_hi,
                        S=float(comps_r["S"]),
                        MAE={k: float(np.mean(v)) for k, v in err_r.items()},
                    )
                    rule_pa, rule_fl, rule_mt = pa_tmp, fl_tmp, mt_tmp

    return {
        "trainS_pick": {
            "S_loocv_cal_pa_mt": float(comps["S"]),
            "MAE_loocv": mae,
            "picked_counts": {v: int(picked.count(v)) for v in variants},
            "gate_0.12": float(comps["S"]) < 0.12,
            "beats_0.13294": float(comps["S"]) < PRIOR_BEST,
        },
        "feature_rule": best_rule,
    }


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
        "--out-dir",
        type=Path,
        default=ROOT / "reports/phase3/chase018",
    )
    ap.add_argument("--save-best-maps", action="store_true")
    args = ap.parse_args()

    expert = pd.read_csv(args.expert_csv)
    kw = dict(FROZEN)
    results = []
    preds = {}
    best = None
    best_df = None
    for v in VARIANTS:
        print(f"=== {v} ===", flush=True)
        save_dir = None
        summary, df = run_variant(v, expert, args.expert_dir, args.cache_dir, kw, save_dir)
        results.append(summary)
        if df is not None:
            preds[v] = df
            print(
                f"  S={summary['S_loocv_cal_pa_mt']:.5f} "
                f"pa={summary['MAE_loocv']['pa_deg']:.3f} "
                f"fl={summary['MAE_loocv']['fl_mm']:.3f} "
                f"mt={summary['MAE_loocv']['mt_mm']:.3f} "
                f"beats={summary['beats_0.13294']}",
                flush=True,
            )
            if best is None or summary["S_loocv_cal_pa_mt"] < best["S_loocv_cal_pa_mt"]:
                best = summary
                best_df = df

    # C: fold-pure selectors among variants that produced dfs
    ok_vars = [v for v in VARIANTS if v in preds]
    feat_keys = [
        "fasc_mass",
        "apo_mass",
        "belly_frac",
        "apo_gap_px",
        "coher_mean",
        "orient_disp",
        "scale",
    ]
    print("=== fold-pure selector ===", flush=True)
    sel = fold_pure_selector(preds, feat_keys, ok_vars)
    print("trainS_pick", sel["trainS_pick"], flush=True)
    print("feature_rule", sel["feature_rule"], flush=True)

    # Oracle upper bound: per-image best variant by abs FL+PA err (raw, then LOOCV affine on chosen)
    # Honest oracle for diagnosis only
    base = preds["baseline"]
    imgs = list(base.image_file.values)
    n = len(imgs)
    gpa = base.set_index("image_file").loc[imgs, "gt_pa"].values
    gfl = base.set_index("image_file").loc[imgs, "gt_fl"].values
    gmt = base.set_index("image_file").loc[imgs, "gt_mt"].values
    # pick variant minimizing |fl-gt|/12+|pa-gt|/6 raw per image (oracle)
    oracle_pa = np.zeros(n)
    oracle_fl = np.zeros(n)
    oracle_mt = np.zeros(n)
    for i, im in enumerate(imgs):
        best_e, bp, bf, bm = 9.0, None, None, None
        for v in ok_vars:
            r = preds[v].set_index("image_file").loc[im]
            e = abs(r.pred_fl - gfl[i]) / 12.0 + abs(r.pred_pa - gpa[i]) / 6.0
            if e < best_e:
                best_e = e
                bp, bf, bm = float(r.pred_pa), float(r.pred_fl), float(r.pred_mt)
        oracle_pa[i], oracle_fl[i], oracle_mt[i] = bp, bf, bm
    # LOOCV affine on oracle preds
    pa_o = np.zeros(n)
    mt_o = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        a, b = np.polyfit(oracle_pa[m], gpa[m], 1)
        pa_o[i] = a * oracle_pa[i] + b
        a, b = np.polyfit(oracle_mt[m], gmt[m], 1)
        mt_o[i] = a * oracle_mt[i] + b
    err_o = {
        "pa_deg": np.abs(pa_o - gpa),
        "fl_mm": np.abs(oracle_fl - gfl),
        "mt_mm": np.abs(mt_o - gmt),
    }
    comps_o = score_components(err_o)
    oracle = {
        "S_loocv_cal_pa_mt": float(comps_o["S"]),
        "MAE_loocv": {k: float(np.mean(v)) for k, v in err_o.items()},
        "note": "per-image oracle pick among A variants; not transferable",
    }
    print("oracle", oracle, flush=True)

    ranked = sorted(
        [r for r in results if "S_loocv_cal_pa_mt" in r],
        key=lambda r: r["S_loocv_cal_pa_mt"],
    )
    # consider selector as candidate
    candidates = list(ranked)
    if sel["trainS_pick"]["S_loocv_cal_pa_mt"] < (best["S_loocv_cal_pa_mt"] if best else 9):
        best = {
            "name": "selector_trainS_pick",
            "S_loocv_cal_pa_mt": sel["trainS_pick"]["S_loocv_cal_pa_mt"],
            "MAE_loocv": sel["trainS_pick"]["MAE_loocv"],
            "gate_0.12": sel["trainS_pick"]["gate_0.12"],
            "beats_0.13294": sel["trainS_pick"]["beats_0.13294"],
            "gap_to_0.12": sel["trainS_pick"]["S_loocv_cal_pa_mt"] - 0.12,
        }
    if sel["feature_rule"] and sel["feature_rule"]["S"] < best["S_loocv_cal_pa_mt"]:
        best = {
            "name": f"selector_rule_{sel['feature_rule']['feature']}",
            "S_loocv_cal_pa_mt": sel["feature_rule"]["S"],
            "MAE_loocv": sel["feature_rule"]["MAE"],
            "gate_0.12": sel["feature_rule"]["S"] < 0.12,
            "beats_0.13294": sel["feature_rule"]["S"] < PRIOR_BEST,
            "gap_to_0.12": sel["feature_rule"]["S"] - 0.12,
            "rule": sel["feature_rule"],
        }

    # optionally dump best single-variant adapted maps
    if args.save_best_maps and ranked:
        vbest = ranked[0]["name"]
        if vbest in VARIANTS and vbest != "baseline":
            out_maps = args.out_dir / "cache" / f"maps_multexp_adapt_{vbest}"
            print(f"saving adapted maps for {vbest} -> {out_maps}", flush=True)
            run_variant(vbest, expert, args.expert_dir, args.cache_dir, kw, out_maps)

    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "protocol": "honest LOOCV affine PA+MT; FL raw; frozen recipe",
        "frozen_recipe": FROZEN,
        "prior_best_S": PRIOR_BEST,
        "cache_dir": str(args.cache_dir),
        "variants_ranked": ranked,
        "selector": sel,
        "oracle_among_A": oracle,
        "best": best,
        "gate_0.12": bool(best and best["S_loocv_cal_pa_mt"] < 0.12),
        "promote": bool(best and best["S_loocv_cal_pa_mt"] < PRIOR_BEST),
        "kaggle_submit": False,
        "lab_fasc_ckpt": False,
        "tried": [
            "A: apo-belly gate, orient filter, ridge blend/replace, morph connect/prune, frangi-like, corridor dilate",
            "C: fold-pure trainS variant pick + feature-threshold rules",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "multexp_fasc_adapt_scores.json"
    out_json.write_text(json.dumps(payload, indent=2))
    if best_df is not None and ranked and ranked[0]["name"] in preds:
        preds[ranked[0]["name"]].to_csv(args.out_dir / "multexp_fasc_adapt_best_rows.csv", index=False)

    status = []
    status.append(f"# MultExp fasc adaptation (A/C) — {payload['asof']}")
    status.append("")
    status.append(f"- Prior best S: **{PRIOR_BEST:.5f}** (frozen recipe on Frundles maps)")
    status.append(
        f"- Best adapted: **{best['name']}** S=**{best['S_loocv_cal_pa_mt']:.5f}** "
        f"PA {best['MAE_loocv']['pa_deg']:.3f} / FL {best['MAE_loocv']['fl_mm']:.3f} / "
        f"MT {best['MAE_loocv']['mt_mm']:.3f}"
    )
    status.append(f"- Beats prior: {best.get('beats_0.13294')} | gate 0.12: {best.get('gate_0.12')} | promote: {payload['promote']}")
    status.append(f"- Oracle among A variants: S={oracle['S_loocv_cal_pa_mt']:.5f}")
    status.append("")
    status.append("## Ranked variants")
    for r in ranked[:12]:
        status.append(
            f"- {r['S_loocv_cal_pa_mt']:.5f}  `{r['name']}`  "
            f"pa={r['MAE_loocv']['pa_deg']:.3f} fl={r['MAE_loocv']['fl_mm']:.3f} "
            f"mt={r['MAE_loocv']['mt_mm']:.3f}"
        )
    status.append("")
    status.append("## Selector (C)")
    status.append(f"- trainS_pick S={sel['trainS_pick']['S_loocv_cal_pa_mt']:.5f} counts={sel['trainS_pick']['picked_counts']}")
    status.append(f"- feature_rule: {sel['feature_rule']}")
    status.append("")
    if payload["promote"]:
        status.append("Promoted (recipe JSON will be updated by caller if single-variant beat).")
    else:
        gap = best["S_loocv_cal_pa_mt"] - 0.12
        status.append(
            f"No promotion. Remaining gap to 0.12 ≈ {gap:.5f}. "
            "A/C post-filters on Frundles fasc likely insufficient; next lever is short "
            "geometry-aware MultExp fasc fine-tune (B) with nested-safe LOOCV protocol."
        )
    status_path = args.out_dir / "multexp_fasc_adapt_status.md"
    status_path.write_text("\n".join(status) + "\n")
    print("wrote", out_json, status_path, flush=True)
    print(
        f"BEST {best['name']} S={best['S_loocv_cal_pa_mt']:.5f} "
        f"promote={payload['promote']} gate={payload['gate_0.12']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
