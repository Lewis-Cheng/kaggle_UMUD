#!/usr/bin/env python3
"""FL post-process + transferable FL-source selector on Frundles MultExp maps.

Keeps promoted PA/MT (hough_tex03_thr032_apo60). Explores FL candidates only.
Honest LOOCV: affine on PA+MT only; FL raw. Nested fold-pure selector.
No Kaggle. No lab fasc ckpt.
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
PRIOR_FLOOR = 0.1376082508388217
PRIOR_SELECTOR = 0.13550

PROMOTED = dict(
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
    pa_mode="hough",
)


def _intersect_endpoints(cx, cy, ang, curves, protocol: str):
    """protocol: centre | deep_insert | shallow_insert | both_inner"""
    c0, c1 = curves[0], curves[1]
    if protocol == "centre":
        p1 = line_curve_intersection(cx, cy, ang, c0[0])
        p2 = line_curve_intersection(cx, cy, ang, c1[0])
    elif protocol == "deep_insert":
        # shift start toward deep apo slightly along normal-ish (use deep curve)
        p1 = line_curve_intersection(cx, cy, ang, c0[0])
        p2 = line_curve_intersection(cx, cy, ang, c1[0])
        if p2 is not None:
            # pull deep endpoint 10% of span toward fascicle center (insertion gate)
            p2 = 0.9 * p2 + 0.1 * np.array([cx, cy])
    elif protocol == "shallow_insert":
        p1 = line_curve_intersection(cx, cy, ang, c0[0])
        p2 = line_curve_intersection(cx, cy, ang, c1[0])
        if p1 is not None:
            p1 = 0.9 * p1 + 0.1 * np.array([cx, cy])
    elif protocol == "both_inner":
        p1 = line_curve_intersection(cx, cy, ang, c0[0])
        p2 = line_curve_intersection(cx, cy, ang, c1[0])
        if p1 is not None:
            p1 = 0.85 * p1 + 0.15 * np.array([cx, cy])
        if p2 is not None:
            p2 = 0.85 * p2 + 0.15 * np.array([cx, cy])
    else:
        p1 = line_curve_intersection(cx, cy, ang, c0[0])
        p2 = line_curve_intersection(cx, cy, ang, c1[0])
    return p1, p2


def _ill_conditioned(p1, p2, ang, max_cond: float = 25.0) -> bool:
    if p1 is None or p2 is None:
        return True
    v = p2 - p1
    L = float(np.linalg.norm(v))
    if not (10 < L < 300):
        return True
    # angle between fascicle direction and chord
    k = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    u = v / (L + 1e-9)
    cos = abs(float(np.dot(k, u)))
    if cos < 1.0 / max_cond:
        return True
    # long horizontal extrap relative to vertical span
    if abs(v[0]) > max_cond * (abs(v[1]) + 1e-3):
        return True
    return False


def _fl_from_segs(fls, ws, factor):
    fls_a, ws_a = np.asarray(fls, dtype=float), np.asarray(ws, dtype=float)
    good = np.isfinite(fls_a) & (fls_a > 10) & (fls_a < 300)
    if not good.any():
        return float("nan")
    return weighted_median(fls_a[good], ws_a[good]) * factor


def predict_bundle(g, apo, fasc, px_per_mm, kw=None):
    kw = dict(PROMOTED if kw is None else kw)
    H0, W0 = g.shape
    h, w = apo.shape
    mx, my = mm_per_prob_pixel(px_per_mm, px_per_mm, W0, H0, w, h)
    rich = aponeuroses_rich(apo, mx, my, thr=kw["apo_thr"])
    curves = apo_curves(apo, mx, my, thr=kw["apo_thr"])
    if rich is None or len(curves) < 2:
        base = dict(ok=False, pred_pa=15.0, pred_mt=20.0, fl_candidates={})
        return base

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

    # collect segments once
    segs_raw = []
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
        segs_raw.append((cx, cy, ang, L, ys_sup, ys_deep, pa))

    pas = [s[6] for s in segs_raw]
    ws = [s[3] for s in segs_raw]
    dep = [
        (s[1] - s[4]) / (s[5] - s[4]) if s[5] != s[4] else 0.5 for s in segs_raw
    ]
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

    # FL candidates by endpoint protocol
    protocols = ("centre", "deep_insert", "shallow_insert", "both_inner")
    fl_by_proto = {}
    for proto in protocols:
        fls = []
        wsl = []
        n_reject = 0
        for cx, cy, ang, L, ys_sup, ys_deep, pa_s in segs_raw:
            p1, p2 = _intersect_endpoints(cx, cy, ang, curves, proto)
            if _ill_conditioned(p1, p2, ang):
                n_reject += 1
                continue
            fls.append(float(np.linalg.norm(p1 - p2)))
            wsl.append(L)
        fl_by_proto[proto] = dict(fls=fls, ws=wsl, n_reject=n_reject, n=len(segs_raw))

    # texture FL
    fl_tex_raw = float("nan")
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
            if p1 is not None and p2 is not None and not _ill_conditioned(p1, p2, fang):
                L = float(np.linalg.norm(p1 - p2))
                if 10 < L < 300:
                    fls_t.append(L)
        if fls_t:
            fl_tex_raw = float(np.median(fls_t))

    fl_sin = mt / np.sin(np.radians(np.clip(pa, 5, 45)))

    cands = {}
    # promoted-like baseline (centre + agree)
    fl_geo0 = _fl_from_segs(fl_by_proto["centre"]["fls"], fl_by_proto["centre"]["ws"], 0.9)
    if not np.isfinite(fl_geo0):
        fl_geo0 = 70.0
    fl_tex0 = fl_tex_raw * 0.9 if np.isfinite(fl_tex_raw) else float("nan")
    if np.isfinite(fl_tex0) and abs(fl_geo0 - fl_tex0) > 15.0:
        cands["promoted"] = 0.5 * fl_geo0 + 0.5 * fl_sin
    else:
        cands["promoted"] = 0.75 * fl_geo0 + 0.25 * fl_sin

    # fl_factor grid on centre geo
    for ff in (0.80, 0.85, 0.90, 0.95, 1.00):
        fg = _fl_from_segs(fl_by_proto["centre"]["fls"], fl_by_proto["centre"]["ws"], ff)
        if not np.isfinite(fg):
            fg = 70.0
        cands[f"geo_ff{ff:.2f}"] = fg
        cands[f"geo_sin25_ff{ff:.2f}"] = 0.75 * fg + 0.25 * fl_sin
        cands[f"geo_sin40_ff{ff:.2f}"] = 0.60 * fg + 0.40 * fl_sin
        cands[f"geo_sin50_ff{ff:.2f}"] = 0.50 * fg + 0.50 * fl_sin
        if np.isfinite(fl_tex_raw):
            ft = fl_tex_raw * ff
            cands[f"tex_ff{ff:.2f}"] = ft
            cands[f"geotex50_ff{ff:.2f}"] = 0.5 * fg + 0.5 * ft
            cands[f"agree_sin_ff{ff:.2f}"] = (
                0.5 * fg + 0.5 * fl_sin
                if abs(fg - ft) > 15.0
                else 0.75 * fg + 0.25 * fl_sin
            )

    # endpoint protocols
    for proto in protocols:
        for ff in (0.85, 0.90, 0.95):
            fg = _fl_from_segs(fl_by_proto[proto]["fls"], fl_by_proto[proto]["ws"], ff)
            if not np.isfinite(fg):
                # reject→fallback to centre
                fg = _fl_from_segs(
                    fl_by_proto["centre"]["fls"], fl_by_proto["centre"]["ws"], ff
                )
            if not np.isfinite(fg):
                fg = 70.0
            cands[f"{proto}_ff{ff:.2f}"] = fg
            cands[f"{proto}_sin25_ff{ff:.2f}"] = 0.75 * fg + 0.25 * fl_sin
            cands[f"{proto}_sin50_ff{ff:.2f}"] = 0.50 * fg + 0.50 * fl_sin

    # pure sin + blends
    cands["sin_only"] = float(fl_sin)
    cands["sin_clip_pa"] = float(mt / np.sin(np.radians(np.clip(pa, 8, 40))))
    for w in (0.15, 0.25, 0.35, 0.45, 0.60):
        cands[f"blend_sin{w:.2f}"] = (1 - w) * fl_geo0 + w * fl_sin

    # reject-heavy: if many ill-conditioned, prefer sin
    n_tot = max(1, fl_by_proto["centre"]["n"])
    reject_rate = fl_by_proto["centre"]["n_reject"] / n_tot
    cands["reject_gate_sin"] = fl_sin if reject_rate > 0.4 else fl_geo0
    cands["reject_gate_blend"] = (
        0.4 * fl_geo0 + 0.6 * fl_sin if reject_rate > 0.3 else 0.75 * fl_geo0 + 0.25 * fl_sin
    )

    # features for selector
    feats = dict(
        pa=float(pa),
        mt=float(mt),
        fl_geo=float(fl_geo0),
        fl_sin=float(fl_sin),
        fl_tex=float(fl_tex0) if np.isfinite(fl_tex0) else float(fl_geo0),
        n_segs=float(len(segs_raw)),
        reject_rate=float(reject_rate),
        agree_abs=float(abs(fl_geo0 - fl_tex0)) if np.isfinite(fl_tex0) else 0.0,
        pa_tex=float(pa_tex) if np.isfinite(pa_tex) else float(pa),
        fang=float(fang) if np.isfinite(fang) else float(pa),
    )

    return dict(
        ok=bool(pas or pas_h),
        pred_pa=float(pa),
        pred_mt=float(mt),
        fl_candidates={k: float(v) for k, v in cands.items()},
        feats=feats,
    )


def loocv_cal_pa_mt_with_fl(df: pd.DataFrame, fl_col: str):
    n = len(df)
    pa, fl, mt = df.pred_pa.values, df[fl_col].values, df.pred_mt.values
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


def oracle_pick(df: pd.DataFrame, cand_cols: list[str]):
    """Per-row best FL source (optimistic upper bound; not for promotion)."""
    n = len(df)
    fl_o = np.zeros(n)
    picks = []
    for i in range(n):
        best_c, best_e = None, 1e9
        for c in cand_cols:
            e = abs(float(df.iloc[i][c]) - float(df.iloc[i]["gt_fl"]))
            if e < best_e:
                best_e, best_c = e, c
        fl_o[i] = df.iloc[i][best_c]
        picks.append(best_c)
    tmp = df.copy()
    tmp["fl_oracle"] = fl_o
    comps, mae = loocv_cal_pa_mt_with_fl(tmp, "fl_oracle")
    return float(comps["S"]), mae, picks


def fold_pure_selector(df: pd.DataFrame, cand_cols: list[str], feat_cols: list[str]):
    """LOOCV: fit L1 residual SoftmaxBandit-style weights on train fold only.

    Softmax over candidates with features → logits = W @ feats + b, pick argmax
    or soft blend. Use leave-one-out; train uses ridge multinomial via
    per-candidate absolute residual regression (transferable).
    """
    n = len(df)
    X = df[feat_cols].to_numpy(dtype=float)
    # standardize with train stats inside loop
    fl_o = np.zeros(n)
    picks = []
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        Xtr, Xte = X[m], X[i]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr_n = (Xtr - mu) / sd
        Xte_n = (Xte - mu) / sd
        # residual target: which candidate closest to GT FL on train
        # fit linear score: -|pred-gt| ≈ w·x + b  via OLS per candidate, pick max score
        scores = []
        for c in cand_cols:
            y = -np.abs(df.loc[m, c].to_numpy() - df.loc[m, "gt_fl"].to_numpy())
            # ridge OLS
            A = np.concatenate([Xtr_n, np.ones((Xtr_n.shape[0], 1))], 1)
            reg = 1e-2
            w = np.linalg.solve(A.T @ A + reg * np.eye(A.shape[1]), A.T @ y)
            scores.append(float(w[:-1] @ Xte_n + w[-1]))
        # softmax blend of top candidates (temperature)
        sc = np.asarray(scores)
        sc = sc - sc.max()
        p = np.exp(sc / 0.5)
        p = p / p.sum()
        # hard pick
        j = int(np.argmax(scores))
        fl_hard = float(df.iloc[i][cand_cols[j]])
        # soft blend
        fl_soft = float(sum(p[k] * df.iloc[i][cand_cols[k]] for k in range(len(cand_cols))))
        # residual corrector: train L1 residual of hard pick → shrink toward sin/geo
        fl_o[i] = fl_soft  # soft is more transferable historically
        picks.append(cand_cols[j])
    tmp = df.copy()
    tmp["fl_sel"] = fl_o
    comps, mae = loocv_cal_pa_mt_with_fl(tmp, "fl_sel")
    # also evaluate hard
    tmp["fl_hard"] = [float(df.iloc[i][picks[i]]) for i in range(n)]
    comps_h, mae_h = loocv_cal_pa_mt_with_fl(tmp, "fl_hard")
    return {
        "soft": {"S": float(comps["S"]), "MAE": mae, "picks": picks},
        "hard": {"S": float(comps_h["S"]), "MAE": mae_h},
    }


def residual_l1_selector(df: pd.DataFrame, cand_cols: list[str], feat_cols: list[str]):
    """Pick candidate minimizing predicted |FL-GT| via per-cand ridge, then
    apply LOOCV L1 residual shrink: fl' = fl + a*(fl_sin-fl) style.
    """
    n = len(df)
    X = df[feat_cols].to_numpy(dtype=float)
    fl_o = np.zeros(n)
    picks = []
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        Xtr, Xte = X[m], X[i]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr_n = (Xtr - mu) / sd
        Xte_n = (Xte - mu) / sd
        pred_err = []
        for c in cand_cols:
            y = np.abs(df.loc[m, c].to_numpy() - df.loc[m, "gt_fl"].to_numpy())
            A = np.concatenate([Xtr_n, np.ones((Xtr_n.shape[0], 1))], 1)
            w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ y)
            pred_err.append(float(w[:-1] @ Xte_n + w[-1]))
        j = int(np.argmin(pred_err))
        fl = float(df.iloc[i][cand_cols[j]])
        # residual shrink toward sin using train-optimal alpha
        fl_sin = float(df.iloc[i]["feat_fl_sin"])
        # fit alpha on train: minimize |fl_c + a*(fl_sin_c - fl_c) - gt|
        best_a, best_e = 0.0, 1e9
        for a in np.linspace(0, 0.8, 17):
            fl_tr = df.loc[m, cand_cols[j]].to_numpy()
            sin_tr = df.loc[m, "feat_fl_sin"].to_numpy()
            e = np.mean(np.abs(fl_tr + a * (sin_tr - fl_tr) - df.loc[m, "gt_fl"].to_numpy()))
            if e < best_e:
                best_e, best_a = e, a
        fl_o[i] = fl + best_a * (fl_sin - fl)
        picks.append(cand_cols[j])
    tmp = df.copy()
    tmp["fl_sel"] = fl_o
    comps, mae = loocv_cal_pa_mt_with_fl(tmp, "fl_sel")
    return {"S": float(comps["S"]), "MAE": mae, "picks": picks}


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
        default=ROOT / "reports/phase3/chase018/multexp_fl_postprocess_scores.json",
    )
    args = ap.parse_args()

    expert = pd.read_csv(args.expert_csv)
    rows = []
    cand_names = None
    for _, er in expert.iterrows():
        name_img = er["image_file"]
        g = cv2.imread(str(args.expert_dir / name_img), cv2.IMREAD_GRAYSCALE)
        zpath = args.cache_dir / f"{name_img}.npz"
        if g is None or not zpath.exists():
            print(f"SKIP {name_img}", flush=True)
            continue
        z = np.load(zpath)
        pred = predict_bundle(
            g,
            z["apo"].astype(np.float32),
            z["fasc"].astype(np.float32),
            float(er["Scale_pixel_per_cm"]) / 10.0,
        )
        if cand_names is None:
            cand_names = sorted(pred["fl_candidates"].keys())
        row = dict(
            image_file=name_img,
            ok=pred["ok"],
            pred_pa=pred["pred_pa"],
            pred_mt=pred["pred_mt"],
            gt_pa=float(er["median_PA"]),
            gt_fl=float(er["median_FL"]),
            gt_mt=float(er["median_MT"]),
        )
        for k, v in pred["fl_candidates"].items():
            row[k] = v
        for k, v in pred["feats"].items():
            row[f"feat_{k}"] = v
        rows.append(row)
        print(f"  {name_img} pa={pred['pred_pa']:.2f} mt={pred['pred_mt']:.2f} n_cands={len(pred['fl_candidates'])}", flush=True)

    df = pd.DataFrame(rows)
    assert cand_names is not None
    print(f"n={len(df)} candidates={len(cand_names)}", flush=True)

    # score each candidate alone
    ranked = []
    for c in cand_names:
        comps, mae = loocv_cal_pa_mt_with_fl(df, c)
        ranked.append(
            dict(
                name=c,
                S=float(comps["S"]),
                MAE=mae,
                gate_0_12=float(comps["S"]) < 0.12,
            )
        )
    ranked.sort(key=lambda r: r["S"])
    print("TOP single-source FL:", flush=True)
    for r in ranked[:15]:
        print(
            f"  {r['S']:.5f}  {r['name']}  pa={r['MAE']['pa_deg']:.3f} "
            f"fl={r['MAE']['fl_mm']:.3f} mt={r['MAE']['mt_mm']:.3f}",
            flush=True,
        )

    # oracle
    S_oracle, mae_oracle, oracle_picks = oracle_pick(df, cand_names)
    print(f"ORACLE S={S_oracle:.5f} MAE={mae_oracle}", flush=True)

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    # compact candidate shortlist for selector (top-K by single + promoted + key blends)
    shortlist = []
    for r in ranked[:20]:
        shortlist.append(r["name"])
    for must in ("promoted", "sin_only", "blend_sin0.25", "geo_sin25_ff0.90", "centre_ff0.90", "deep_insert_sin25_ff0.90", "reject_gate_blend"):
        if must in cand_names and must not in shortlist:
            shortlist.append(must)

    sel = fold_pure_selector(df, shortlist, feat_cols)
    print(f"SELECTOR soft S={sel['soft']['S']:.5f} hard S={sel['hard']['S']:.5f}", flush=True)

    sel_l1 = residual_l1_selector(df, shortlist, feat_cols)
    print(f"SELECTOR L1-resid S={sel_l1['S']:.5f}", flush=True)

    # second-pass: tighter shortlist of 8
    short8 = [r["name"] for r in ranked[:8]]
    sel8 = fold_pure_selector(df, short8, feat_cols)
    sel8_l1 = residual_l1_selector(df, short8, feat_cols)
    print(f"SELECTOR8 soft S={sel8['soft']['S']:.5f} L1 S={sel8_l1['S']:.5f}", flush=True)

    best_single = ranked[0]
    best_transfer = min(
        [
            ("soft_top20", sel["soft"]["S"], sel["soft"]["MAE"]),
            ("hard_top20", sel["hard"]["S"], sel["hard"]["MAE"]),
            ("l1_top20", sel_l1["S"], sel_l1["MAE"]),
            ("soft_top8", sel8["soft"]["S"], sel8["soft"]["MAE"]),
            ("l1_top8", sel8_l1["S"], sel8_l1["MAE"]),
            ("single_" + best_single["name"], best_single["S"], best_single["MAE"]),
        ],
        key=lambda t: t[1],
    )

    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "cache_dir": str(args.cache_dir),
        "n": len(df),
        "n_candidates": len(cand_names),
        "prior_promoted_floor": PRIOR_FLOOR,
        "prior_selector": PRIOR_SELECTOR,
        "gate_0.12": best_transfer[1] < 0.12,
        "best_transferable": {
            "name": best_transfer[0],
            "S": best_transfer[1],
            "MAE": best_transfer[2],
            "beats_floor": best_transfer[1] < PRIOR_FLOOR,
            "beats_selector": best_transfer[1] < PRIOR_SELECTOR,
        },
        "best_single": best_single,
        "oracle": {"S": S_oracle, "MAE": mae_oracle},
        "selector_soft_top20": {"S": sel["soft"]["S"], "MAE": sel["soft"]["MAE"]},
        "selector_hard_top20": {"S": sel["hard"]["S"], "MAE": sel["hard"]["MAE"]},
        "selector_l1_top20": {"S": sel_l1["S"], "MAE": sel_l1["MAE"]},
        "selector_soft_top8": {"S": sel8["soft"]["S"], "MAE": sel8["soft"]["MAE"]},
        "selector_l1_top8": {"S": sel8_l1["S"], "MAE": sel8_l1["MAE"]},
        "ranked_single_top30": ranked[:30],
        "kaggle_submit": False,
        "lab_fasc_ckpt": False,
        "protocol": "LOOCV affine PA+MT only; FL raw; fold-pure selector",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    # rows dump
    rows_path = args.out_json.with_name(args.out_json.stem + "_rows.csv")
    df.to_csv(rows_path, index=False)
    print("wrote", args.out_json, rows_path, flush=True)
    print(
        f"BEST_TRANSFER {best_transfer[0]} S={best_transfer[1]:.5f} "
        f"gate={best_transfer[1] < 0.12} vs_floor={best_transfer[1] - PRIOR_FLOOR:+.5f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
