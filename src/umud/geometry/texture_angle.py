"""Pennation angle from raw image texture (Mawra LB~0.306 recipe).

Independent of fascicle segmentation: isotropic resample → high-pass →
discrete Radon (projection variance) over candidate angles inside the deep
muscle belly between inner apo edges. PA is relative to deep-apo centreline
slope.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from umud.geometry.host_faithful import slope_deg


def texture_angle(
    gray: np.ndarray,
    sx: float,
    sy: float,
    apo: list[dict[str, Any]],
    *,
    r: float = 4.0,
    depth_lo: float = 0.45,
    depth_hi: float = 0.92,
    margin_mm: float = 1.0,
    angles: np.ndarray | None = None,
    min_abs: float = 4.0,
) -> tuple[float, float]:
    """Return (fascicle angle image-coords deg, PA vs deep apo deg) or (nan, nan).

    ``apo`` is a length-2 list of rich bands from ``aponeuroses_rich``:
    each has keys lo, hi, fit['c'|'top'|'bot'].
    """
    if angles is None:
        angles = np.arange(-50, 50.5, 0.5)
    if apo is None or len(apo) < 2:
        return float("nan"), float("nan")
    sup, deep = apo[0], apo[1]
    H0, W0 = gray.shape
    iso = cv2.resize(
        gray,
        (max(int(W0 * r / sx), 8), max(int(H0 * r / sy), 8)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    hp = iso - cv2.GaussianBlur(iso, (0, 0), 2.0 * r)
    h, w = hp.shape
    X = (np.arange(w) + 0.5) / r
    lo, hi = max(sup["lo"], deep["lo"]), min(sup["hi"], deep["hi"])
    ytop = np.polyval(sup["fit"]["bot"], X) + margin_mm
    ybot = np.polyval(deep["fit"]["top"], X) - margin_mm
    y0 = ytop + depth_lo * (ybot - ytop)
    y1 = ytop + depth_hi * (ybot - ytop)
    Y = (np.arange(h)[:, None] + 0.5) / r
    roi = (
        (Y >= y0[None])
        & (Y <= y1[None])
        & (X[None] >= lo)
        & (X[None] <= hi)
        & (ybot > ytop)[None]
    )
    if int(roi.sum()) < 400:
        return float("nan"), float("nan")
    yy, xx = np.nonzero(roi)
    v = hp[yy, xx]
    v = (v - v.mean()) / (v.std() + 1e-6)
    best, best_a = -1.0, float("nan")
    for a in angles:
        if abs(float(a)) < min_abs:
            continue
        t = np.radians(float(a))
        b = np.round(-xx * np.sin(t) + yy * np.cos(t)).astype(np.int64)
        b -= b.min()
        n = np.bincount(b)
        s = np.bincount(b, weights=v)
        ok = n >= 15
        if int(ok.sum()) < 5:
            continue
        mu = s[ok] / n[ok]
        score = float(np.average(mu**2, weights=n[ok]))
        if score > best:
            best, best_a = score, float(a)
    if not np.isfinite(best_a):
        return float("nan"), float("nan")
    xc = float(np.clip((lo + hi) / 2, deep["lo"], deep["hi"]))
    pa = abs(((best_a - float(slope_deg(deep["fit"]["c"], xc))) + 90) % 180 - 90)
    return float(best_a), float(pa)
