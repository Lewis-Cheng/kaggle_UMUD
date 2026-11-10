"""Synthetic B-mode muscle images with exact PA / FL / MT under host-faithful definitions.

Used only to check that the attention + RL pipeline can learn and that its
contracts hold. A synthetic score says nothing about the public leaderboard.

The image contains the true superficial and deep aponeuroses, parallel
fascicles of varying contrast, and two distractors seen on real data: a
central band between the aponeuroses and short near-horizontal speckle
streaks. The annotated fascicle mask holds only the three most visible
fascicles, mimicking the host's three-fascicle procedure. The ``p_apo`` and
``p_fasc`` maps imitate a generic segmenter: they respond to the distractors
and to unannotated fascicles, and break lines into fragments.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from umud.geometry.attended_readout import fold as fold_deg
from umud.geometry.host_faithful import line_curve_intersection

N_ANNOTATED = 3


@dataclass
class SyntheticSample:
    image: np.ndarray  # uint8 HxW
    p_apo: np.ndarray  # float32 HxW, segmenter-like
    p_fasc: np.ndarray  # float32 HxW, segmenter-like
    sup_mask: np.ndarray  # uint8 HxW, true superficial band
    deep_mask: np.ndarray  # uint8 HxW, true deep band
    fasc_mask: np.ndarray  # uint8 HxW, annotated fascicles only
    lines: np.ndarray  # (k, 5) annotated visible segments: x1, y1, x2, y2, angle_deg (pixel axes)
    sup_poly: np.ndarray  # centre polynomial, pixel units
    deep_poly: np.ndarray
    sup_th: float  # band thickness, pixels
    deep_th: float
    mm_per_px: float
    labels: dict[str, float]


def _band(h: int, w: int, poly: np.ndarray, th: float, x_lo: int = 0, x_hi: int | None = None) -> np.ndarray:
    x_hi = w if x_hi is None else x_hi
    yy = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)
    yc = np.polyval(poly, xs)[None, :]
    band = (np.abs(yy - yc) <= th / 2.0).astype(np.uint8)
    band[:, :x_lo] = 0
    band[:, x_hi:] = 0
    return band


def _visible_segment(x0: float, y0: float, ang: float, inside) -> tuple[float, float, float, float] | None:
    u = np.arange(-600.0, 600.0, 0.5)
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    xs, ys = x0 + u * c, y0 + u * s
    keep = inside(xs, ys)
    if keep.sum() < 2:
        return None
    idx = np.nonzero(keep)[0]
    a, b = idx.min(), idx.max()
    return float(xs[a]), float(ys[a]), float(xs[b]), float(ys[b])


def make_sample(
    rng: np.random.Generator,
    h: int = 96,
    w: int = 144,
    mm_per_px: float = 0.25,
    p_central: float = 0.2,
    n_streaks: tuple[int, int] = (2, 6),
) -> SyntheticSample:
    xc = w / 2.0
    ys0 = rng.uniform(0.10, 0.18) * h
    bs = rng.uniform(-0.05, 0.05)
    sup_poly = np.array([bs, ys0 - bs * xc])
    yd0 = rng.uniform(0.62, 0.82) * h
    bd = rng.uniform(-0.10, 0.10)
    cd = rng.uniform(-3e-4, 3e-4)
    # y = cd (x - xc)^2 + bd (x - xc) + yd0, expanded to numpy polyval order.
    deep_poly = np.array([cd, bd - 2 * cd * xc, cd * xc * xc - bd * xc + yd0])
    sup_th = float(rng.uniform(3.0, 5.0))
    deep_th = float(rng.uniform(3.0, 5.0))

    def sup_inner(x):
        return np.polyval(sup_poly, x) + sup_th / 2.0

    def deep_inner(x):
        return np.polyval(deep_poly, x) - deep_th / 2.0

    def inside(xs, ys):
        return (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1) & (ys > sup_inner(xs)) & (ys < deep_inner(xs))

    pa_true = rng.uniform(10.0, 28.0)
    # lean +1: fascicles rise to the right of their deep insertion; -1: to the left.
    lean = 1.0 if rng.random() < 0.5 else -1.0
    run = (yd0 - ys0) / np.tan(np.radians(pa_true))
    lo, hi = (-run, float(w)) if lean > 0 else (0.0, w + run)
    n_fasc = int(rng.integers(6, 11))
    crossings = np.linspace(lo, hi, n_fasc + 2)[1:-1] + rng.uniform(-4.0, 4.0, n_fasc)
    fascicles = []
    for xk in crossings:
        yk = float(np.polyval(deep_poly, xk))
        beta = float(np.degrees(np.arctan(np.polyval(np.polyder(deep_poly), xk))))
        pa_k = pa_true + rng.uniform(-1.5, 1.5)
        ang = float(fold_deg(beta - lean * pa_k))
        seg = _visible_segment(float(xk), yk, ang, inside)
        if seg is None:
            continue
        x1, y1, x2, y2 = seg
        length = float(np.hypot(x2 - x1, y2 - y1))
        fascicles.append(dict(x0=float(xk), y0=yk, ang=ang, seg=seg, length=length, contrast=float(rng.uniform(0.25, 1.0))))
    if len(fascicles) < N_ANNOTATED:
        return make_sample(rng, h, w, mm_per_px, p_central, n_streaks)

    visible = [f for f in fascicles if f["length"] >= 0.25 * w]
    if len(visible) < N_ANNOTATED:
        visible = sorted(fascicles, key=lambda f: -f["length"])[:N_ANNOTATED]
    annotated = sorted(visible, key=lambda f: -f["contrast"] * min(1.0, f["length"] / (0.5 * w)))[:N_ANNOTATED]
    for f in annotated:
        f["contrast"] = max(f["contrast"], 0.75)

    img = np.full((h, w), 0.22, np.float32)
    sup_mask = _band(h, w, sup_poly, sup_th)
    deep_mask = _band(h, w, deep_poly, deep_th)
    img[sup_mask > 0] = 0.85
    img[deep_mask > 0] = 0.85
    for f in fascicles:
        canvas = np.zeros((h, w), np.uint8)
        x1, y1, x2, y2 = f["seg"]
        cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, 1, cv2.LINE_AA)
        img = np.maximum(img, 0.22 + 0.5 * f["contrast"] * canvas.astype(np.float32) / 255.0)

    central = None
    if rng.random() < p_central:
        frac = rng.uniform(0.40, 0.60)
        cy = np.polyval(sup_poly, xc) + frac * (np.polyval(deep_poly, xc) - np.polyval(sup_poly, xc))
        central = np.array([rng.uniform(-0.04, 0.04), 0.0])
        central[1] = cy - central[0] * xc
        x_lo = int(rng.uniform(0.0, 0.2) * w)
        x_hi = int(rng.uniform(0.75, 1.0) * w)
        cmask = _band(h, w, central, 2.5, x_lo, x_hi)
        img[cmask > 0] = np.maximum(img[cmask > 0], 0.75)
    else:
        cmask = np.zeros((h, w), np.uint8)

    streaks = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(n_streaks[0], n_streaks[1] + 1))):
        sx = rng.uniform(0.1 * w, 0.9 * w)
        frac = rng.uniform(0.15, 0.85)
        sy = np.polyval(sup_poly, sx) + frac * (np.polyval(deep_poly, sx) - np.polyval(sup_poly, sx))
        sang = np.radians(rng.uniform(-5.0, 5.0))
        half = rng.uniform(0.06, 0.14) * w
        p1 = (int(round(sx - half * np.cos(sang))), int(round(sy - half * np.sin(sang))))
        p2 = (int(round(sx + half * np.cos(sang))), int(round(sy + half * np.sin(sang))))
        cv2.line(streaks, p1, p2, 255, 1, cv2.LINE_AA)
    img = np.maximum(img, 0.22 + 0.45 * streaks.astype(np.float32) / 255.0)

    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    speckle = rng.rayleigh(0.55, size=img.shape).astype(np.float32)
    img = np.clip(img * speckle * 1.1 + rng.normal(0, 0.03, img.shape), 0, 1)
    image = (img * 255).astype(np.uint8)

    fasc_mask = np.zeros((h, w), np.uint8)
    lines = []
    for f in annotated:
        x1, y1, x2, y2 = f["seg"]
        cv2.line(fasc_mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 1, 2)
        lines.append((x1, y1, x2, y2, f["ang"]))

    # Segmenter-like maps.
    p_apo = np.maximum(sup_mask, deep_mask).astype(np.float32)
    if rng.random() < 0.5:
        a = int(rng.uniform(0, 0.4) * w)
        p_apo[:, a : a + int(0.15 * w)] *= rng.uniform(0.2, 0.6)
    p_apo = np.maximum(p_apo, cmask.astype(np.float32) * rng.uniform(0.7, 1.0))
    p_apo = cv2.GaussianBlur(p_apo, (5, 5), 1.0)
    p_apo = np.clip(p_apo + rng.normal(0, 0.04, p_apo.shape), 0, 1).astype(np.float32)

    p_fasc = np.zeros((h, w), np.float32)
    for f in fascicles:
        x1, y1, x2, y2 = f["seg"]
        n_pieces = int(rng.integers(2, 5))
        ts = np.sort(rng.uniform(0, 1, size=2 * n_pieces))
        for i in range(n_pieces):
            a, b = ts[2 * i], ts[2 * i + 1]
            if b - a < 0.08:
                continue
            q1 = (int(round(x1 + a * (x2 - x1))), int(round(y1 + a * (y2 - y1))))
            q2 = (int(round(x1 + b * (x2 - x1))), int(round(y1 + b * (y2 - y1))))
            canvas = np.zeros((h, w), np.uint8)
            cv2.line(canvas, q1, q2, 255, 2)
            p_fasc = np.maximum(p_fasc, (0.4 + 0.6 * f["contrast"]) * canvas.astype(np.float32) / 255.0)
    p_fasc = np.maximum(p_fasc, cv2.dilate(streaks, np.ones((2, 2), np.uint8)).astype(np.float32) / 255.0 * 0.9)
    p_fasc = cv2.GaussianBlur(p_fasc, (3, 3), 0.8)
    p_fasc = np.clip(p_fasc + rng.normal(0, 0.03, p_fasc.shape), 0, 1).astype(np.float32)

    labels = exact_labels(annotated, sup_poly, deep_poly, sup_th, deep_th, w, mm_per_px)
    return SyntheticSample(
        image=image,
        p_apo=p_apo,
        p_fasc=p_fasc,
        sup_mask=sup_mask,
        deep_mask=deep_mask,
        fasc_mask=fasc_mask,
        lines=np.asarray(lines, np.float64).reshape(-1, 5),
        sup_poly=sup_poly,
        deep_poly=deep_poly,
        sup_th=sup_th,
        deep_th=deep_th,
        mm_per_px=mm_per_px,
        labels=labels,
    )


def exact_labels(
    annotated: list[dict],
    sup_poly: np.ndarray,
    deep_poly: np.ndarray,
    sup_th: float,
    deep_th: float,
    w: int,
    mm_per_px: float,
) -> dict[str, float]:
    pas, fls = [], []
    for f in annotated:
        x1, y1, x2, y2 = f["seg"]
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        beta = float(np.degrees(np.arctan(np.polyval(np.polyder(deep_poly), np.clip(cx, 0, w - 1)))))
        pas.append(abs(float(fold_deg(f["ang"] - beta))))
        p1 = line_curve_intersection(cx, cy, f["ang"], sup_poly)
        p2 = line_curve_intersection(cx, cy, f["ang"], deep_poly)
        if p1 is not None and p2 is not None:
            fls.append(float(np.linalg.norm(p1 - p2)) * mm_per_px)
    xs = np.array([0.2, 0.5, 0.8]) * (w - 1)
    mt_px = np.polyval(deep_poly, xs) - np.polyval(sup_poly, xs) - (sup_th + deep_th) / 2.0
    return {
        "pa_deg": float(np.median(pas)),
        "fl_mm": float(np.median(fls)) if fls else float("nan"),
        "mt_mm": float(np.mean(mt_px)) * mm_per_px,
    }


def make_dataset(n: int, seed: int = 0, **kw) -> list[SyntheticSample]:
    rng = np.random.default_rng(seed)
    return [make_sample(rng, **kw) for _ in range(n)]
