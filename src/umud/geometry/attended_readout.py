"""Host-faithful PA / FL / MT from attended fascicle lines and two band maps.

Same definitions as ``host_faithful.measure``: PA against the local deep
aponeurosis slope, FL as the straight line extrapolated to both centre curves
(never ``MT / sin(PA)``), MT between inner borders. Differences from
``measure``: the superficial and deep bands come from separate maps, and the
lines come from the attention map or from the RL agent instead of every
thresholded fragment of ``p_fasc``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from umud.geometry.host_faithful import _column_profile, line_curve_intersection, slope_deg, weighted_median


def fold(a: np.ndarray | float) -> np.ndarray | float:
    # Float % can round up to the divisor (e.g. -1e-14 % 180 == 180.0).
    f = (np.asarray(a, dtype=np.float64) + 90.0) % 180.0 - 90.0
    return np.where(f >= 90.0, f - 180.0, f)


def pa_of(a: np.ndarray | float) -> np.ndarray | float:
    return np.abs(fold(a))


@dataclass
class PixelLine:
    """A line through ``(x, y)`` at ``t_deg`` in pixel axes (y down)."""

    x: float
    y: float
    t_deg: float
    weight: float = 1.0


def band_fit(p: np.ndarray, mx: float, my: float, thr: float = 0.5, min_rel_width: float = 0.03) -> dict[str, Any] | None:
    """Fit one band map. All components belong to the band; the model chose it."""
    h, w = p.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((p > thr).astype(np.uint8), connectivity=8)
    keep = np.zeros((h, w), bool)
    for i in range(1, n):
        if st[i, 2] >= min_rel_width * w:
            keep |= lab == i
    if keep.sum() < 3:
        return None
    cols = np.unique(np.nonzero(keep)[1])
    if len(cols) < 3:
        return None
    cy, top, bot = _column_profile(keep, p, cols)
    X = cols.astype(float) * mx
    band = dict(X=X, c=cy * my, top=top * my, bot=bot * my)
    deg = 2 if X.max() - X.min() > 15 else 1
    band["fit"] = {k: np.polyfit(X, band[k], deg) for k in ("c", "top", "bot")}
    band["lo"], band["hi"] = float(X.min()), float(X.max())
    return band


def inner_border_mt(sup: dict[str, Any], deep: dict[str, Any]) -> float:
    th = lambda b: float(np.median(b["bot"] - b["top"]))  # noqa: E731
    lo, hi = max(sup["lo"], deep["lo"]), min(sup["hi"], deep["hi"])
    if hi - lo < 3:
        lo, hi = min(sup["lo"], deep["lo"]), max(sup["hi"], deep["hi"])
    xs = lo + np.array([0.2, 0.5, 0.8]) * (hi - lo)
    d = np.polyval(deep["fit"]["c"], xs) - np.polyval(sup["fit"]["c"], xs) - (th(sup) + th(deep)) / 2.0
    return float(np.mean(d))


def mm_angle(t_deg: float, mx: float, my: float) -> float:
    """Direction angle in millimetre axes of a pixel-axis direction."""
    t = np.radians(t_deg)
    return float(np.degrees(np.arctan2(np.sin(t) * my, np.cos(t) * mx)))


def lines_from_attention(
    attn: np.ndarray,
    orient: np.ndarray | None = None,
    thr: float = 0.5,
    min_len_px: float = 6.0,
) -> list[PixelLine]:
    """One line per attention component: centroid, PCA direction, mass weight."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((attn > thr).astype(np.uint8), connectivity=8)
    out: list[PixelLine] = []
    for i in range(1, n):
        if st[i, 4] < 6:
            continue
        ys, xs = np.nonzero(lab == i)
        P = np.stack([xs, ys], 1).astype(np.float64)
        c = P.mean(0)
        _, s, vt = np.linalg.svd(P - c, full_matrices=False)
        proj = (P - c) @ vt[0]
        length = float(proj.max() - proj.min())
        if length < min_len_px or s[0] < 3 * (s[1] + 1e-6):
            continue
        t = float(fold(np.degrees(np.arctan2(vt[0][1], vt[0][0]))))
        if orient is not None:
            o = orient[:, ys, xs].mean(1)
            if np.hypot(o[0], o[1]) > 0.3:
                t_o = float(fold(0.5 * np.degrees(np.arctan2(o[1], o[0]))))
                # Keep PCA unless the field disagrees strongly with a short blob.
                if length < 2 * min_len_px and abs(float(fold(t_o - t))) > 10:
                    t = t_o
        out.append(PixelLine(float(c[0]), float(c[1]), t, length * float(attn[ys, xs].mean())))
    return out


def readout(
    sup_map: np.ndarray,
    deep_map: np.ndarray,
    lines: Sequence[PixelLine],
    mx: float,
    my: float,
    thr: float = 0.5,
) -> dict[str, Any]:
    """PA/FL/MT in degrees and mm. ``ok`` False leaves NaN so callers can abstain."""
    out: dict[str, Any] = dict(pa_deg=float("nan"), fl_mm=float("nan"), mt_mm=float("nan"), n_lines=0, ok=False)
    sup = band_fit(sup_map, mx, my, thr)
    deep = band_fit(deep_map, mx, my, thr)
    if sup is None or deep is None:
        return out
    xc = 0.5 * (max(sup["lo"], deep["lo"]) + min(sup["hi"], deep["hi"]))
    if np.polyval(deep["fit"]["c"], xc) <= np.polyval(sup["fit"]["c"], xc):
        return out
    out["mt_mm"] = inner_border_mt(sup, deep)
    out["sup"], out["deep"] = sup, deep
    pas, fls, ws = [], [], []
    for ln in lines:
        cx, cy = ln.x * mx, ln.y * my
        ang = mm_angle(ln.t_deg, mx, my)
        ys_sup = float(np.polyval(sup["fit"]["c"], cx))
        ys_deep = float(np.polyval(deep["fit"]["c"], cx))
        if not (ys_sup - 2 < cy < ys_deep + 2):
            continue
        pa = float(pa_of(ang - float(slope_deg(deep["fit"]["c"], np.clip(cx, deep["lo"], deep["hi"])))))
        if not (2 < pa < 60):
            continue
        p1 = line_curve_intersection(cx, cy, ang, sup["fit"]["c"])
        p2 = line_curve_intersection(cx, cy, ang, deep["fit"]["c"])
        pas.append(pa)
        ws.append(max(ln.weight, 1e-6))
        fls.append(float(np.linalg.norm(p1 - p2)) if p1 is not None and p2 is not None else float("nan"))
    out["n_lines"] = len(pas)
    if pas:
        pas_a, fls_a, ws_a = map(np.asarray, (pas, fls, ws))
        out["pa_deg"] = weighted_median(pas_a, ws_a)
        good = np.isfinite(fls_a) & (fls_a > 10) & (fls_a < 300)
        if good.any():
            out["fl_mm"] = weighted_median(fls_a[good], ws_a[good])
        out["ok"] = bool(np.isfinite(out["fl_mm"]))
    return out
