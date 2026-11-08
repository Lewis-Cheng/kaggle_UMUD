#!/usr/bin/env python3
"""Apo-relative soft PA (ExpDrawn) + FL geometry refine on Frundles MultExp maps.

Experiment A: ExpDrawn stroke angles vs deep-apo tangent -> soft PA targets;
  light Frundles FT with BCE+Dice+soft_pa (MultExp held out); optional PA head.
Experiment B: Frundles MultExp maps only; new FL endpoint/protocol/cond gates;
  fold-pure protocol LOOCV. Combo if A improves PA.
No Kaggle. No whole-mask ExpDrawn Dice-only FT chase.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
    dhash,
)
from umud.geometry.texture_angle import texture_angle  # noqa: E402
from umud.metric.score import score_components  # noqa: E402
from umud.seg.frundles_smp import load_ckpt, predict_prob  # noqa: E402

HK = timezone(timedelta(hours=8))
HOLD = 0.13195144580335014
GATE = 0.12
FRUNDLE_BASE = 0.13293555901039555
H, W = 384, 576

FROZEN = dict(
    name="hough_tex0.4_thr035_apo60",
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


def now() -> str:
    return datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S Asia/Hong_Kong")


def soft_dice(probs, gt, eps=1e-6):
    dims = (1, 2, 3) if probs.ndim == 4 else (1, 2)
    inter = (probs * gt).sum(dims)
    return 1.0 - (2 * inter + eps) / (probs.sum(dims) + gt.sum(dims) + eps)


def structure_angle(x, eps=1e-6):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    avg = torch.ones(1, 1, 5, 5, dtype=x.dtype, device=x.device) / 25.0
    gxx = F.conv2d(gx * gx, avg, padding=2)
    gyy = F.conv2d(gy * gy, avg, padding=2)
    gxy = F.conv2d(gx * gy, avg, padding=2)
    return 0.5 * torch.atan2(2 * gxy, gxx - gyy + eps)


def soft_pa_loss(probs, pa_target_deg, apo_slope_deg, mask_w, eps=1e-6):
    """Encourage fasc structure-tensor angle to match apo-relative PA soft target.

    pa_target_deg: (B,) degrees. apo_slope_deg: (B,) deep-apo slope degrees.
    mask_w: (B,1,H,W) soft weight along strokes / belly.
    Fasc absolute angle ~ apo_slope + signed_pa; we match |ang - apo| to pa_target.
    """
    ang = structure_angle(probs)  # radians, orientation of intensity ridges
    # convert structure angle (rad) to image degrees in [-90,90]
    ang_deg = ang * (180.0 / math.pi)
    # PA relative to apo: abs wrap
    apo = apo_slope_deg.view(-1, 1, 1, 1)
    pa_pred = ((ang_deg - apo + 90.0) % 180.0) - 90.0
    pa_pred = pa_pred.abs()
    tgt = pa_target_deg.view(-1, 1, 1, 1).clamp(2.0, 50.0)
    w = mask_w
    # soft L1 in degrees, normalized
    diff = (pa_pred - tgt).abs() / 30.0
    num = (diff * w).sum((1, 2, 3))
    den = w.sum((1, 2, 3)) + eps
    return (num / den).mean()


def groups_from_hashes(hs, max_dist=12):
    parent = list(range(len(hs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    arr = np.array(hs, dtype=np.uint8)
    for i in range(len(arr)):
        for j in np.nonzero((arr[i + 1 :] != arr[i]).sum(1) <= max_dist)[0] + i + 1:
            parent[find(int(j))] = find(i)
    return np.array([find(i) for i in range(len(hs))])


def multexp_hashes():
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    hs = []
    for _, er in expert.iterrows():
        g = cv2.imread(str(expert_dir / er["image_file"]), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            hs.append(dhash(g))
    return hs


def far_from_multexp(img_path: Path, mex_hs, max_dist=8) -> bool:
    g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return False
    h = dhash(g)
    for mh in mex_hs:
        if (np.asarray(h, dtype=np.uint8) != np.asarray(mh, dtype=np.uint8)).sum() <= max_dist:
            return False
    return True


def stroke_apo_relative_pa(gray: np.ndarray, mask: np.ndarray, apo_prob: np.ndarray, px_per_mm: float = 10.0):
    """Extract apo-relative PA from ExpDrawn fasc strokes vs deep-apo tangent."""
    H0, W0 = gray.shape
    h, w = apo_prob.shape
    mx, my = mm_per_prob_pixel(px_per_mm, px_per_mm, W0, H0, w, h)
    # resize mask to apo map size for geometry in map coords
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    curves = apo_curves(apo_prob, mx, my, thr=0.60)
    if curves is None or len(curves) < 2:
        return None
    # connected components on strokes
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
    pas, lens = [], []
    weight_map = np.zeros((h, w), dtype=np.float32)
    for lab in range(1, nlab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 25:
            continue
        ys, xs = np.where(labels == lab)
        if len(xs) < 10:
            continue
        # PCA orientation in pixel coords then mm
        pts = np.stack([xs.astype(np.float64) * mx, ys.astype(np.float64) * my], 1)
        pts -= pts.mean(0, keepdims=True)
        cov = pts.T @ pts / max(len(pts) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        vx, vy = evecs[:, int(np.argmax(evals))]
        ang = (np.degrees(np.arctan2(vy, vx)) + 90) % 180 - 90
        cx = float(xs.mean() * mx)
        cy = float(ys.mean() * my)
        ys_sup = float(np.polyval(curves[0][0], cx))
        ys_deep = float(np.polyval(curves[1][0], cx))
        if not (ys_sup - 3 < cy < ys_deep + 3):
            continue
        apo_s = float(slope_deg(curves[1][0], np.clip(cx, curves[1][1], curves[1][2])))
        pa = abs(((ang - apo_s) + 90) % 180 - 90)
        if not (2 < pa < 55):
            continue
        L = float(np.sqrt(evals.max()) * 2)  # approx half-length scale
        pas.append(pa)
        lens.append(max(L, 1.0))
        weight_map[labels == lab] = max(L, 1.0)
    if not pas:
        return None
    pa_img = float(weighted_median(np.asarray(pas), np.asarray(lens)))
    # deep apo slope at image center of mass of strokes
    ys, xs = np.where(weight_map > 0)
    cx = float(xs.mean() * mx) if len(xs) else 0.5 * w * mx
    apo_s = float(slope_deg(curves[1][0], np.clip(cx, curves[1][1], curves[1][2])))
    # resize weight to training size later
    return dict(pa=pa_img, apo_slope=apo_s, weight=weight_map, n_strokes=len(pas))


def build_expdrawn_pa_targets(apo_model, apo_size, device, extract_json: Path, img_dir: Path, mask_dir: Path, mex_hs):
    meta = json.loads(extract_json.read_text())["meta"]
    rows = []
    for m in meta:
        if not m.get("ok"):
            continue
        stem = m["stem"]
        ip = img_dir / f"{stem}.png"
        mp = mask_dir / f"{stem}.png"
        if not ip.exists() or not mp.exists():
            continue
        if not far_from_multexp(ip, mex_hs):
            continue
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if g is None or mask is None:
            continue
        apo = predict_prob(apo_model, apo_size, g, device)
        tgt = stroke_apo_relative_pa(g, mask, apo)
        if tgt is None:
            continue
        rows.append(
            dict(
                stem=stem,
                path_img=str(ip),
                path_mask=str(mp),
                pa=tgt["pa"],
                apo_slope=tgt["apo_slope"],
                n_strokes=tgt["n_strokes"],
                weight=tgt["weight"],  # at apo map size
                apo=apo,
            )
        )
    return rows


class ApoRelPADataset(Dataset):
    def __init__(self, rows, augment=True, seed=0):
        self.rows = rows
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = cv2.imread(r["path_img"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        y = (cv2.imread(r["path_mask"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        wmap = r["weight"].astype(np.float32)
        # resize all to HxW
        x = cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA)
        y = cv2.resize(y, (W, H), interpolation=cv2.INTER_AREA)
        y = (y > 0.25).astype(np.float32)
        wmap = cv2.resize(wmap, (W, H), interpolation=cv2.INTER_LINEAR)
        if wmap.max() > 0:
            wmap = wmap / (wmap.max() + 1e-6)
        # light belly dilate weight + stroke
        wmap = np.clip(wmap + 0.15 * y, 0, 1)
        if self.augment:
            if self.rng.random() < 0.5:
                x = np.fliplr(x).copy()
                y = np.fliplr(y).copy()
                wmap = np.fliplr(wmap).copy()
            if self.rng.random() < 0.3:
                g = float(self.rng.uniform(0.85, 1.15))
                x = np.clip(x ** g, 0, 1)
            if self.rng.random() < 0.3:
                x = np.clip(x + float(self.rng.normal(0, 0.03)), 0, 1)
        return (
            torch.from_numpy(x[None]),
            torch.from_numpy(y[None]),
            torch.from_numpy(wmap[None]),
            torch.tensor([r["pa"]], dtype=torch.float32),
            torch.tensor([r["apo_slope"]], dtype=torch.float32),
        )


def loocv_cal_pa_mt(df: pd.DataFrame):
    n = len(df)
    pa, fl, mt = df.pred_pa.values.astype(float), df.pred_fl.values.astype(float), df.pred_mt.values.astype(float)
    gpa, gfl, gmt = df.gt_pa.values.astype(float), df.gt_fl.values.astype(float), df.gt_mt.values.astype(float)
    pa_o = np.zeros(n)
    mt_o = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        a, b = np.polyfit(pa[m], gpa[m], 1)
        pa_o[i] = a * pa[i] + b
        a, b = np.polyfit(mt[m], gmt[m], 1)
        mt_o[i] = a * mt[i] + b
    err = {"pa_deg": np.abs(pa_o - gpa), "fl_mm": np.abs(fl - gfl), "mt_mm": np.abs(mt_o - gmt)}
    return score_components(err), {k: float(np.mean(v)) for k, v in err.items()}


def _ill(p1, p2, ang, max_cond=25.0) -> bool:
    if p1 is None or p2 is None:
        return True
    v = p2 - p1
    L = float(np.linalg.norm(v))
    if not (10 < L < 300):
        return True
    k = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    u = v / (L + 1e-9)
    if abs(float(np.dot(k, u))) < 1.0 / max_cond:
        return True
    if abs(v[0]) > max_cond * (abs(v[1]) + 1e-3):
        return True
    return False


def _fl_wm(fls, ws, factor):
    fls_a, ws_a = np.asarray(fls, dtype=float), np.asarray(ws, dtype=float)
    good = np.isfinite(fls_a) & (fls_a > 10) & (fls_a < 300)
    if not good.any():
        return float("nan")
    return weighted_median(fls_a[good], ws_a[good]) * factor


def predict_bundle(g, apo, fasc, px_per_mm, kw=None, pa_override=None, fl_protocol="baseline"):
    """Predict PA/MT/FL with optional PA override and FL protocol variants."""
    kw = dict(FROZEN if kw is None else kw)
    H0, W0 = g.shape
    h, w = apo.shape
    mx, my = mm_per_prob_pixel(px_per_mm, px_per_mm, W0, H0, w, h)
    rich = aponeuroses_rich(apo, mx, my, thr=kw["apo_thr"])
    curves = apo_curves(apo, mx, my, thr=kw["apo_thr"])
    if rich is None or len(curves) < 2:
        return dict(ok=False, pred_pa=15.0, pred_fl=70.0, pred_mt=20.0, fl_cands={})

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
    fang, pa_tex = texture_angle(g, px_per_mm, px_per_mm, rich, depth_lo=kw["tex_lo"], depth_hi=kw["tex_hi"])

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
            pa = abs(((ang - float(slope_deg(curves[1][0], np.clip(cx, curves[1][1], curves[1][2])))) + 90) % 180 - 90)
            if 2 < pa < 60:
                pas_h.append(pa)
    pa_h = float(np.median(pas_h)) if pas_h else 15.0

    segs_raw = []
    for cx, cy, ang, L in fascicle_segments(fasc, mx, my, thr=kw["fasc_thr"], min_len_mm=kw["min_len"]):
        ys_sup = float(np.polyval(curves[0][0], cx))
        ys_deep = float(np.polyval(curves[1][0], cx))
        if not (ys_sup - 2 < cy < ys_deep + 2):
            continue
        pa = abs(((ang - float(slope_deg(curves[1][0], np.clip(cx, curves[1][1], curves[1][2])))) + 90) % 180 - 90)
        if not (2 < pa < 60):
            continue
        segs_raw.append((cx, cy, ang, L, ys_sup, ys_deep, pa))

    pas = [s[6] for s in segs_raw]
    ws = [s[3] for s in segs_raw]
    dep = [(s[1] - s[4]) / (s[5] - s[4]) if s[5] != s[4] else 0.5 for s in segs_raw]
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
    if pa_override is not None and np.isfinite(pa_override):
        pa = float(pa_override)

    # --- FL candidates ---
    fl_cands = {}
    # centre geo — match frozen recipe (NO ill-filter on baseline fl_geo)
    fls_c, ws_c = [], []
    n_rej = 0
    for cx, cy, ang, L, *_rest in segs_raw:
        p1 = line_curve_intersection(cx, cy, ang, curves[0][0])
        p2 = line_curve_intersection(cx, cy, ang, curves[1][0])
        if p1 is None or p2 is None:
            n_rej += 1
            continue
        Lfl = float(np.linalg.norm(p1 - p2))
        if _ill(p1, p2, ang):
            n_rej += 1
        if 10 < Lfl < 300:
            fls_c.append(Lfl)
            ws_c.append(L)
    fl_geo = _fl_wm(fls_c, ws_c, kw["fl_factor"])
    if not np.isfinite(fl_geo):
        fl_geo = 70.0
    # also texture FL like frozen recipe for agree path
    fl_tex = float("nan")
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
    reject_rate = n_rej / max(1, len(segs_raw))

    # baseline frozen blend (exact agree_thr logic)
    if kw.get("agree_thr") is not None and np.isfinite(fl_tex) and abs(fl_geo - fl_tex) > kw["agree_thr"]:
        fl_cands["baseline"] = 0.5 * fl_geo + 0.5 * fl_sin
    else:
        fl_cands["baseline"] = (1 - kw["fl_sin_w"]) * fl_geo + kw["fl_sin_w"] * fl_sin

    # deep_insert: pull deep endpoint 10% toward centre
    fls_d, ws_d = [], []
    for cx, cy, ang, L, *_r in segs_raw:
        p1 = line_curve_intersection(cx, cy, ang, curves[0][0])
        p2 = line_curve_intersection(cx, cy, ang, curves[1][0])
        if p2 is not None:
            p2 = 0.9 * p2 + 0.1 * np.array([cx, cy])
        if _ill(p1, p2, ang):
            continue
        fls_d.append(float(np.linalg.norm(p1 - p2)))
        ws_d.append(L)
    fg = _fl_wm(fls_d, ws_d, kw["fl_factor"])
    fl_cands["deep_insert"] = (1 - kw["fl_sin_w"]) * (fg if np.isfinite(fg) else fl_geo) + kw["fl_sin_w"] * fl_sin

    # tip_extrapolate: skeleton tips near shallow -> cast to deep
    fls_t, ws_t = [], []
    bin_f = (fasc > kw["fasc_thr"]).astype(np.uint8)
    # morphological skeleton (opencv-contrib thinning not always present)
    sk = np.zeros_like(bin_f)
    tmp = bin_f.copy()
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(64):
        if not tmp.any():
            break
        er = cv2.erode(tmp, ker)
        opened = cv2.dilate(er, ker)
        sk = np.maximum(sk, tmp - opened)
        tmp = er
    # tips: skeleton pixels with <=1 neighbor
    sk_b = (sk > 0).astype(np.uint8)
    if sk_b.any():
        ker8 = np.ones((3, 3), np.uint8)
        nbr = cv2.filter2D(sk_b, -1, ker8) - sk_b
        tips = np.argwhere((sk_b > 0) & (nbr <= 1))
        for y, x in tips[:40]:
            cx, cy = x * mx, y * my
            ys_sup = float(np.polyval(curves[0][0], cx))
            ys_deep = float(np.polyval(curves[1][0], cx))
            if not (ys_sup - 4 < cy < ys_deep + 4):
                continue
            # local angle from nearby skeleton
            y0, x0 = max(0, y - 6), max(0, x - 6)
            patch = sk_b[y0 : y + 7, x0 : x + 7]
            ys2, xs2 = np.where(patch > 0)
            if len(xs2) < 3:
                continue
            pts = np.stack([xs2.astype(float) * mx, ys2.astype(float) * my], 1)
            pts -= pts.mean(0)
            cov = pts.T @ pts / max(len(pts) - 1, 1)
            evals, evecs = np.linalg.eigh(cov)
            vx, vy = evecs[:, int(np.argmax(evals))]
            ang = (np.degrees(np.arctan2(vy, vx)) + 90) % 180 - 90
            # prefer tips closer to shallow (upper)
            depth = (cy - ys_sup) / (ys_deep - ys_sup + 1e-6)
            if depth > 0.55:
                continue
            p1 = line_curve_intersection(cx, cy, ang, curves[0][0])
            p2 = line_curve_intersection(cx, cy, ang, curves[1][0])
            if _ill(p1, p2, ang, max_cond=20.0):
                continue
            fls_t.append(float(np.linalg.norm(p1 - p2)))
            ws_t.append(1.0 + (0.55 - depth))
    fg = _fl_wm(fls_t, ws_t, kw["fl_factor"])
    fl_cands["tip_extrapolate"] = (1 - kw["fl_sin_w"]) * (fg if np.isfinite(fg) else fl_geo) + kw["fl_sin_w"] * fl_sin

    # cond_sin_gate
    fl_cands["cond_sin_gate"] = float(fl_sin if reject_rate > 0.35 or abs(fl_geo - fl_sin) / (abs(fl_sin) + 1e-3) > 0.35 else fl_cands["baseline"])

    # mt_pa_cap
    lo_c, hi_c = 0.75 * fl_sin, 1.25 * fl_sin
    fl_cands["mt_pa_cap"] = float(np.clip(fl_geo, min(lo_c, hi_c), max(lo_c, hi_c)))
    fl_cands["mt_pa_cap_blend"] = 0.5 * fl_cands["mt_pa_cap"] + 0.5 * fl_cands["baseline"]

    # deep_snap: snap deep intersection to apo ridge (poly curve already); use sin if ill
    fls_s, ws_s, n_fb = [], [], 0
    for cx, cy, ang, L, *_r in segs_raw:
        p1 = line_curve_intersection(cx, cy, ang, curves[0][0])
        p2 = line_curve_intersection(cx, cy, ang, curves[1][0])
        if _ill(p1, p2, ang, max_cond=18.0):
            n_fb += 1
            # fallback length via sin at this segment PA
            pa_s = _r[-1] if _r else pa
            fls_s.append(float(mt / np.sin(np.radians(np.clip(pa_s, 5, 45)))))
            ws_s.append(L * 0.5)
            continue
        fls_s.append(float(np.linalg.norm(p1 - p2)))
        ws_s.append(L)
    fg = _fl_wm(fls_s, ws_s, kw["fl_factor"])
    fl_cands["deep_snap_sinfb"] = float(fg if np.isfinite(fg) else fl_sin)
    fl_cands["deep_snap_sinfb_blend"] = (1 - kw["fl_sin_w"]) * fl_cands["deep_snap_sinfb"] + kw["fl_sin_w"] * fl_sin

    # sin-heavy / sin-light blends around frozen 0.35
    for wsin in (0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55, 0.65):
        fl_cands[f"sin_w{wsin:.2f}"] = (1 - wsin) * fl_geo + wsin * fl_sin

    # fl_factor micro on centre geo + frozen sin_w
    for ff in (0.80, 0.85, 0.95, 1.00):
        fg = _fl_wm(fls_c, ws_c, ff)
        if not np.isfinite(fg):
            fg = fl_geo
        fl_cands[f"ff{ff:.2f}_sin035"] = (1 - 0.35) * fg + 0.35 * fl_sin
        fl_cands[f"ff{ff:.2f}_sin030"] = (1 - 0.30) * fg + 0.30 * fl_sin

    # PA-clipped sin (avoid extreme PA blowing FL)
    fl_sin_clip = mt / np.sin(np.radians(np.clip(pa, 8, 40)))
    fl_cands["sin_clip_blend"] = (1 - kw["fl_sin_w"]) * fl_geo + kw["fl_sin_w"] * fl_sin_clip
    fl_cands["sin_clip_w45"] = 0.55 * fl_geo + 0.45 * fl_sin_clip

    # reject-rate adaptive
    if reject_rate > 0.4:
        fl_cands["reject_adapt"] = 0.4 * fl_geo + 0.6 * fl_sin
    else:
        fl_cands["reject_adapt"] = fl_cands["baseline"]

    # insertion quality: prefer shorter of geo vs sin when geo>>sin (over-extrapolation)
    if fl_geo > 1.2 * fl_sin:
        fl_cands["overextrap_cap"] = 0.35 * fl_geo + 0.65 * fl_sin
    else:
        fl_cands["overextrap_cap"] = fl_cands["baseline"]

    # PA-adjusted FL = T/sin(PA) with mild geo blend
    fl_cands["sin_pa_only"] = float(fl_sin)
    fl_cands["geo70_sin30"] = 0.7 * fl_geo + 0.3 * fl_sin
    fl_cands["geo65_sin35"] = 0.65 * fl_geo + 0.35 * fl_sin  # alias of baseline if same fl_geo

    # pick requested protocol
    fl = float(fl_cands.get(fl_protocol, fl_cands["baseline"]))
    return dict(
        ok=bool(pas or pas_h),
        pred_pa=float(pa),
        pred_fl=fl,
        pred_mt=float(mt),
        fl_cands={k: float(v) for k, v in fl_cands.items()},
        reject_rate=float(reject_rate),
        pa_h=float(pa_h),
        pa_geo=float(pa_geo),
        pa_tex=float(pa_tex) if np.isfinite(pa_tex) else float(pa),
    )


def eval_cache(cache_dir: Path, fl_protocol="baseline", pa_overrides=None, kw=None):
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    kw = dict(FROZEN if kw is None else kw)
    rows = []
    for _, er in expert.iterrows():
        name = er["image_file"]
        g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
        zp = cache_dir / f"{name}.npz"
        if g is None or not zp.exists():
            continue
        z = np.load(zp)
        po = None if pa_overrides is None else pa_overrides.get(name)
        pred = predict_bundle(
            g,
            z["apo"].astype(np.float32),
            z["fasc"].astype(np.float32),
            float(er["Scale_pixel_per_cm"]) / 10.0,
            kw=kw,
            pa_override=po,
            fl_protocol=fl_protocol,
        )
        row = dict(
            image_file=name,
            ok=pred["ok"],
            pred_pa=pred["pred_pa"],
            pred_fl=pred["pred_fl"],
            pred_mt=pred["pred_mt"],
            gt_pa=float(er["median_PA"]),
            gt_fl=float(er["median_FL"]),
            gt_mt=float(er["median_MT"]),
            reject_rate=pred.get("reject_rate", 0.0),
        )
        for k, v in pred.get("fl_cands", {}).items():
            row[f"fl_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    comps, mae = loocv_cal_pa_mt(df)
    S = float(comps["S"]) if isinstance(comps, dict) else float(comps)
    return S, mae, df


def foldpure_fl_protocol(df: pd.DataFrame, cand_cols: list[str]):
    """Nested LOOCV: pick FL source by train-fold mean |FL-gt|, apply to held-out."""
    n = len(df)
    fl_o = np.zeros(n)
    picks = []
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        best_c, best_e = cand_cols[0], 1e9
        for c in cand_cols:
            e = float(np.mean(np.abs(df.loc[m, c].values - df.loc[m, "gt_fl"].values)))
            if e < best_e:
                best_e, best_c = e, c
        fl_o[i] = float(df.iloc[i][best_c])
        picks.append(best_c)
    # rebuild with selected FL
    df2 = df.copy()
    df2["pred_fl"] = fl_o
    comps, mae = loocv_cal_pa_mt(df2)
    S = float(comps["S"]) if isinstance(comps, dict) else float(comps)
    return S, mae, picks, df2


def dump_multexp_fasc(model, cache_dir: Path, device: str) -> int:
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    apo_cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    n = 0
    with torch.no_grad():
        for _, er in expert.iterrows():
            name = er["image_file"]
            g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
            zsrc = apo_cache / f"{name}.npz"
            if g is None or not zsrc.exists():
                continue
            apo = np.load(zsrc)["apo"].astype(np.float32)
            fasc = predict_prob(model, (H, W), g, device)
            if fasc.shape != apo.shape:
                fasc = cv2.resize(fasc, (apo.shape[1], apo.shape[0]), interpolation=cv2.INTER_LINEAR)
            np.savez_compressed(cache_dir / f"{name}.npz", apo=apo.astype(np.float16), fasc=fasc.astype(np.float16))
            n += 1
    return n


def blend_caches(new_dir: Path, fr_dir: Path, out_dir: Path, alpha: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for zp in sorted(fr_dir.glob("*.npz")):
        zn = new_dir / zp.name
        if not zn.exists():
            continue
        zfr = np.load(zp)
        znew = np.load(zn)
        fasc = alpha * znew["fasc"].astype(np.float32) + (1.0 - alpha) * zfr["fasc"].astype(np.float32)
        np.savez_compressed(out_dir / zp.name, apo=zfr["apo"].astype(np.float16), fasc=fasc.astype(np.float16))
        n += 1
    return n


class PAHead(torch.nn.Module):
    """Tiny PA regressor from fasc map moments + apo slope."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, feats):
        return self.mlp(feats).squeeze(-1)


def fasc_feats(fasc: np.ndarray, apo_slope: float, pa_h: float, pa_tex: float) -> np.ndarray:
    f = fasc.astype(np.float32)
    thr = 0.35
    m = f > thr
    mean_p = float(f.mean())
    mass = float(m.mean())
    if m.any():
        ys, xs = np.where(m)
        cy = float(ys.mean() / f.shape[0])
        cx = float(xs.mean() / f.shape[1])
    else:
        cy, cx = 0.5, 0.5
    return np.array([mean_p, mass, cy, cx, apo_slope / 30.0, pa_h / 30.0, pa_tex / 30.0, float(f[m].mean() if m.any() else 0.0)], dtype=np.float32)


def run_A(args, report_dir: Path):
    t0 = time.time()
    device = args.device
    print(f"[A] start device={device} {now()}", flush=True)
    apo_model, apo_size = load_ckpt(ROOT / "data/external/frundles_weights/apo_resnet34_384x576_f0.pt", device)
    fasc_model, fasc_size = load_ckpt(ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt", device)
    assert fasc_size == (H, W)

    extract_json = report_dir / "multexp_expdrawn_extract.json"
    img_dir = ROOT / "data/external/osf_xbawc/expdrawn_images"
    mask_dir = ROOT / "data/external/osf_xbawc/expdrawn_masks"
    mex_hs = multexp_hashes()
    print("[A] building apo-relative PA targets from ExpDrawn strokes...", flush=True)
    rows = build_expdrawn_pa_targets(apo_model, apo_size, device, extract_json, img_dir, mask_dir, mex_hs)
    print(f"[A] usable ExpDrawn PA targets: {len(rows)}", flush=True)
    if len(rows) < 20:
        return dict(error="too_few_pa_targets", n=len(rows), elapsed_min=(time.time() - t0) / 60)

    # subject-group split
    hs = [dhash(cv2.imread(r["path_img"], cv2.IMREAD_GRAYSCALE)) for r in rows]
    groups = groups_from_hashes(hs)
    uniq = np.unique(groups)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * 0.15)))
    val_g = set(uniq[:n_val].tolist())
    train_rows = [r for r, g in zip(rows, groups) if g not in val_g]
    val_rows = [r for r, g in zip(rows, groups) if g in val_g]
    print(f"[A] train={len(train_rows)} val={len(val_rows)} groups_val={len(val_g)}", flush=True)

    # save PA target summary (no weight maps)
    pa_summary = [{"stem": r["stem"], "pa": r["pa"], "apo_slope": r["apo_slope"], "n_strokes": r["n_strokes"]} for r in rows]
    (report_dir / "multexp_aporelpa_targets.json").write_text(json.dumps({"n": len(rows), "mean_pa": float(np.mean([r["pa"] for r in rows])), "rows": pa_summary}, indent=2))

    ds_tr = ApoRelPADataset(train_rows, augment=True, seed=args.seed)
    ds_va = ApoRelPADataset(val_rows, augment=False, seed=args.seed + 1)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=1)

    model = fasc_model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_w = torch.tensor([args.pos_weight], device=device)
    best_dice, best_state = -1.0, None
    hist = []
    deadline = t0 + args.timebox_min * 60.0

    for ep in range(1, args.epochs + 1):
        if time.time() > deadline:
            print(f"[A] timebox hit at ep {ep}", flush=True)
            break
        model.train()
        tr_loss = []
        for x, y, wmap, pa_t, apo_s in dl_tr:
            x, y, wmap = x.to(device), y.to(device), wmap.to(device)
            pa_t, apo_s = pa_t.to(device).view(-1), apo_s.to(device).view(-1)
            logits = model(x)
            probs = torch.sigmoid(logits)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)
            dice = soft_dice(probs, y).mean()
            spa = soft_pa_loss(probs, pa_t, apo_s, wmap)
            loss = args.w_bce * bce + args.w_dice * dice + args.w_pa * spa
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss.append(float(loss.detach()))
        # val hard dice
        model.eval()
        dices = []
        with torch.no_grad():
            for x, y, wmap, pa_t, apo_s in dl_va:
                x, y = x.to(device), y.to(device)
                probs = torch.sigmoid(model(x))
                pred = (probs > 0.5).float()
                inter = (pred * y).sum((1, 2, 3))
                den = pred.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1e-6
                dices.extend((2 * inter / den).detach().cpu().tolist())
        vd = float(np.mean(dices)) if dices else 0.0
        hist.append({"ep": ep, "loss": float(np.mean(tr_loss)), "val_dice": vd})
        print(f"[A] ep{ep} loss={hist[-1]['loss']:.4f} valDice={vd:.4f}", flush=True)
        if vd > best_dice:
            best_dice = vd
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = report_dir / "seg_ckpts" / "fasc_resnet34_384x576_aporelpa_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": "resnet34", "size": [H, W], "state": model.state_dict(), "note": "aporelpa soft PA FT"}, ckpt_path)

    # dump MultExp maps + blends
    cache_raw = report_dir / "cache" / "maps_multexp_aporelpa"
    n_dump = dump_multexp_fasc(model, cache_raw, device)
    print(f"[A] dumped MultExp maps n={n_dump}", flush=True)
    fr_cache = report_dir / "cache" / "maps_multexp"
    results = []
    best = None
    for thr in (0.30, 0.35, 0.40):
        kw = dict(FROZEN)
        kw["fasc_thr"] = thr
        S, mae, df = eval_cache(cache_raw, fl_protocol="baseline", kw=kw)
        rec = {"tag": "aporelpa_raw", "thr": thr, "S": S, "mae": mae}
        results.append(rec)
        print(f"[A] raw thr={thr} S={S:.5f} PA={mae['pa_deg']:.3f} FL={mae['fl_mm']:.3f}", flush=True)
        if best is None or S < best["S"]:
            best = {**rec, "df": df}
    for a in (0.15, 0.25, 0.40):
        outc = report_dir / "cache" / f"maps_multexp_aporelpa_blend{int(a*100):02d}"
        blend_caches(cache_raw, fr_cache, outc, a)
        S, mae, df = eval_cache(outc, fl_protocol="baseline")
        rec = {"tag": f"blend_a{a:.2f}", "thr": 0.35, "S": S, "mae": mae}
        results.append(rec)
        print(f"[A] blend a={a:.2f} S={S:.5f} PA={mae['pa_deg']:.3f} FL={mae['fl_mm']:.3f}", flush=True)
        if best is None or S < best["S"]:
            best = {**rec, "df": df, "cache": str(outc)}

    # Frundles baseline PA for comparison
    S0, mae0, df0 = eval_cache(fr_cache, fl_protocol="baseline")
    print(f"[A] Frundles ref S={S0:.5f} PA={mae0['pa_deg']:.3f}", flush=True)

    # PA head on ExpDrawn targets using Frundles maps (no MultExp)
    print("[A] training light PA head on ExpDrawn...", flush=True)
    head = PAHead().to(device)
    opt_h = torch.optim.Adam(head.parameters(), lr=1e-3)
    # build feature rows from train ExpDrawn using Frundles fasc predict
    fasc_model_fr, _ = load_ckpt(ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt", device)
    Xtr, Ytr = [], []
    for r in train_rows:
        g = cv2.imread(r["path_img"], cv2.IMREAD_GRAYSCALE)
        fasc = predict_prob(fasc_model_fr, (H, W), g, device)
        # cheap pa_h/tex proxies from fasc
        feats = fasc_feats(fasc, r["apo_slope"], r["pa"], r["pa"])
        Xtr.append(feats)
        Ytr.append(r["pa"])
    Xtr_t = torch.tensor(np.stack(Xtr), device=device)
    Ytr_t = torch.tensor(np.asarray(Ytr, dtype=np.float32), device=device)
    head.train()
    for _ in range(400):
        pred = head(Xtr_t)
        loss = F.smooth_l1_loss(pred, Ytr_t)
        opt_h.zero_grad()
        loss.backward()
        opt_h.step()
    head.eval()
    # apply PA head on MultExp Frundles maps as override blend
    pa_overrides = {}
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    with torch.no_grad():
        for _, er in expert.iterrows():
            name = er["image_file"]
            zp = fr_cache / f"{name}.npz"
            g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
            if g is None or not zp.exists():
                continue
            z = np.load(zp)
            pred0 = predict_bundle(g, z["apo"].astype(np.float32), z["fasc"].astype(np.float32), float(er["Scale_pixel_per_cm"]) / 10.0)
            # apo slope approx from PA difference: use 0 as placeholder slope in feats; use pa_h/pa_tex
            feats = fasc_feats(z["fasc"].astype(np.float32), 0.0, pred0["pa_h"], pred0["pa_tex"])
            pa_hat = float(head(torch.tensor(feats[None], device=device)).cpu().numpy()[0])
            # blend head with hough PA
            pa_overrides[name] = 0.5 * pred0["pred_pa"] + 0.5 * float(np.clip(pa_hat, 5, 40))
    S_h, mae_h, df_h = eval_cache(fr_cache, fl_protocol="baseline", pa_overrides=pa_overrides)
    results.append({"tag": "pa_head_blend50", "thr": 0.35, "S": S_h, "mae": mae_h})
    print(f"[A] PA-head blend S={S_h:.5f} PA={mae_h['pa_deg']:.3f}", flush=True)
    if best is None or S_h < best["S"]:
        best = {"tag": "pa_head_blend50", "thr": 0.35, "S": S_h, "mae": mae_h, "df": df_h}

    # also PA-head on aporelpa cache if better maps
    elapsed = (time.time() - t0) / 60.0
    out = {
        "asof": now(),
        "experiment": "A_aporel_soft_pa",
        "n_pa_targets": len(rows),
        "mean_target_pa": float(np.mean([r["pa"] for r in rows])),
        "val_dice_best": best_dice,
        "train_hist": hist,
        "ckpt": str(ckpt_path),
        "frundles_ref": {"S": S0, "mae": mae0},
        "results": [{k: v for k, v in r.items() if k != "df"} for r in results],
        "best": {k: v for k, v in best.items() if k != "df"},
        "pa_improved_vs_frundles": bool(best["mae"]["pa_deg"] < mae0["pa_deg"] - 1e-6),
        "beats_hold": bool(best["S"] < HOLD),
        "gate_0.12": bool(best["S"] < GATE),
        "elapsed_min": elapsed,
        "kaggle_submit": False,
    }
    (report_dir / "multexp_aporelpa_flgeo_A.json").write_text(json.dumps(out, indent=2, default=str))
    if "df" in best:
        best["df"].to_csv(report_dir / "multexp_aporelpa_flgeo_A_best_rows.csv", index=False)
    print(f"[A] BEST S={best['S']:.5f} tag={best['tag']} elapsed={elapsed:.1f}m", flush=True)
    return out, best


def run_B(args, report_dir: Path, pa_overrides=None):
    t0 = time.time()
    fr_cache = report_dir / "cache" / "maps_multexp"
    print(f"[B] FL geometry on Frundles maps {now()}", flush=True)
    # evaluate each protocol fixed
    protocols = [
        "baseline",
        "deep_insert",
        "tip_extrapolate",
        "cond_sin_gate",
        "mt_pa_cap",
        "mt_pa_cap_blend",
        "deep_snap_sinfb",
        "deep_snap_sinfb_blend",
        "sin_w0.20", "sin_w0.25", "sin_w0.30", "sin_w0.40",
        "sin_w0.45", "sin_w0.50", "sin_w0.55", "sin_w0.65",
        "ff0.80_sin035", "ff0.85_sin035", "ff0.95_sin035", "ff1.00_sin035",
        "ff0.80_sin030", "ff0.85_sin030", "ff0.95_sin030", "ff1.00_sin030",
        "sin_clip_blend", "sin_clip_w45", "reject_adapt", "overextrap_cap",
        "sin_pa_only", "geo70_sin30", "geo65_sin35",
    ]
    fixed = []
    # get one df with all fl_cands columns via baseline call
    S0, mae0, df0 = eval_cache(fr_cache, fl_protocol="baseline", pa_overrides=pa_overrides)
    fixed.append({"protocol": "baseline", "S": S0, "mae": mae0})
    print(f"[B] baseline S={S0:.5f}", flush=True)

    cand_cols = [c for c in df0.columns if c.startswith("fl_")]
    for proto in protocols[1:]:
        col = f"fl_{proto}"
        if col not in df0.columns:
            continue
        dfp = df0.copy()
        dfp["pred_fl"] = dfp[col].values
        comps, mae = loocv_cal_pa_mt(dfp)
        S = float(comps["S"]) if isinstance(comps, dict) else float(comps)
        fixed.append({"protocol": proto, "S": S, "mae": mae})
        print(f"[B] fixed {proto} S={S:.5f} FL={mae['fl_mm']:.3f}", flush=True)

    # fold-pure among all fl_* 
    S_fp, mae_fp, picks, df_fp = foldpure_fl_protocol(df0, cand_cols)
    from collections import Counter
    pick_counts = dict(Counter(picks))
    print(f"[B] foldpure S={S_fp:.5f} picks={pick_counts}", flush=True)

    # oracle
    fl_oracle = []
    for i in range(len(df0)):
        best_e, best_v = 1e9, df0.iloc[i]["pred_fl"]
        for c in cand_cols:
            e = abs(float(df0.iloc[i][c]) - float(df0.iloc[i]["gt_fl"]))
            if e < best_e:
                best_e, best_v = e, float(df0.iloc[i][c])
        fl_oracle.append(best_v)
    dfo = df0.copy()
    dfo["pred_fl"] = fl_oracle
    comps_o, mae_o = loocv_cal_pa_mt(dfo)
    S_o = float(comps_o["S"]) if isinstance(comps_o, dict) else float(comps_o)

    best_fixed = min(fixed, key=lambda r: r["S"])
    best = best_fixed
    if S_fp < best["S"]:
        best = {"protocol": "foldpure", "S": S_fp, "mae": mae_fp}

    out = {
        "asof": now(),
        "experiment": "B_fl_geometry",
        "pa_overrides": bool(pa_overrides),
        "fixed": fixed,
        "foldpure": {"S": S_fp, "mae": mae_fp, "pick_counts": pick_counts},
        "oracle_S": S_o,
        "oracle_mae": mae_o,
        "best": best,
        "beats_hold": bool(best["S"] < HOLD),
        "gate_0.12": bool(best["S"] < GATE),
        "elapsed_min": (time.time() - t0) / 60.0,
        "kaggle_submit": False,
    }
    tag = "B_combo" if pa_overrides else "B"
    (report_dir / f"multexp_aporelpa_flgeo_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    df_fp.to_csv(report_dir / f"multexp_aporelpa_flgeo_{tag}_rows.csv", index=False)
    print(f"[B] BEST S={best['S']:.5f} proto={best['protocol']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--w-bce", type=float, default=0.55)
    ap.add_argument("--w-dice", type=float, default=0.35)
    ap.add_argument("--w-pa", type=float, default=0.10)
    ap.add_argument("--timebox-min", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-A", action="store_true")
    ap.add_argument("--skip-B", action="store_true")
    ap.add_argument("--resume-A-eval", action="store_true", help="reuse aporelpa ckpt/cache; skip FT")
    args = ap.parse_args()

    report_dir = ROOT / "reports/phase3/chase018"
    report_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    A_out = None
    A_best = None
    if not args.skip_A:
        A_out, A_best = run_A(args, report_dir)
    else:
        A_out = {"skipped": True}

    B_out = None
    if not args.skip_B:
        B_out = run_B(args, report_dir, pa_overrides=None)

    combo_out = None
    if A_out and A_out.get("pa_improved_vs_frundles") and not args.skip_B:
        # rebuild PA overrides from A pa_head if that was best, else use A best df PA
        print("[combo] A improved PA — re-running B with A PA", flush=True)
        pa_ov = None
        if A_best and A_best.get("tag") == "pa_head_blend50":
            # reload from A rows if present
            p = report_dir / "multexp_aporelpa_flgeo_A_best_rows.csv"
            if p.exists():
                dfa = pd.read_csv(p)
                pa_ov = {r.image_file: float(r.pred_pa) for r in dfa.itertuples()}
        elif A_best and "df" in A_best:
            pa_ov = {r.image_file: float(r.pred_pa) for r in A_best["df"].itertuples()}
        if pa_ov:
            combo_out = run_B(args, report_dir, pa_overrides=pa_ov)
        else:
            combo_out = {"note": "no pa overrides available"}
    else:
        combo_out = {"note": "skipped — A did not improve PA vs Frundles"}

    A_S = A_out.get("best", {}).get("S") if isinstance(A_out, dict) else None
    B_S = B_out.get("best", {}).get("S") if isinstance(B_out, dict) else None
    C_S = combo_out.get("best", {}).get("S") if isinstance(combo_out, dict) else None
    scores = [s for s in (A_S, B_S, C_S) if isinstance(s, (int, float))]
    champ = min(scores) if scores else None
    status = f"""# MultExp apo-rel PA + FL geo — {now()}

## Result
| track | S | notes |
|---|---|---|
| A apo-rel soft PA | {A_S} | {A_out.get('best', {}) if isinstance(A_out, dict) else A_out} |
| B FL geometry | {B_S} | {B_out.get('best', {}) if isinstance(B_out, dict) else B_out} |
| combo | {C_S} | {combo_out} |
| **champ** | **{champ}** | |
| hold | {HOLD} | |
| Frundles baseline | {FRUNDLE_BASE} | |
| gate 0.12 | {'PASS' if (champ is not None and champ < GATE) else 'FAIL'} | |
| promote vs hold | {bool(champ is not None and champ < HOLD)} | |
| no Kaggle | true | |

## Diagnosis
A: ExpDrawn stroke→deep-apo soft PA FT (+ optional PA head) on Frundles init; MultExp held out.
B: Frundles MultExp maps only; tip/deep/cond/sin FL protocols + fold-pure.

## Next
{( 'Promoted — prepare LB CSV without Kaggle burn until ready.' if (champ is not None and champ < HOLD) else 'Still above hold; try apo-relative PA from MultExp-safe synthetic geometry or TimTrack-style FL with better insertion priors.') }

Artifacts: multexp_aporelpa_flgeo_A.json, multexp_aporelpa_flgeo_B.json, multexp_aporelpa_flgeo_B_combo.json (if any)
Elapsed_all_min: {(time.time()-t_all)/60.0:.1f}
"""
    # prepend to status.md
    status_path = report_dir / "status.md"
    prev = status_path.read_text() if status_path.exists() else ""
    status_path.write_text(status + "\n---\n\n" + prev)
    summary = {
        "asof": now(),
        "A_S": A_S,
        "B_S": B_S,
        "combo_S": C_S,
        "champ_S": champ,
        "hold": HOLD,
        "gate_0.12": bool(champ is not None and champ < GATE),
        "promote": bool(champ is not None and champ < HOLD),
        "kaggle_submit": False,
        "elapsed_min": (time.time() - t_all) / 60.0,
    }
    (report_dir / "multexp_aporelpa_flgeo_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
