"""TimTrack-style fascicle length from vesselness ridges + orientation tracking.

Pipeline (fold-pure / transferable; no LB feedback):
  1. Optional apo inner-border refine (morphology + distance-transform edges).
  2. Frangi-like vesselness on grayscale (or fasc map) inside muscle belly.
  3. Structure-tensor orientation field on vesselness / high-pass gray.
  4. Seed streamlines mid-belly; integrate along local orientation until apo
     inner borders; report (a) path centreline length and (b) extrapolated
     straight chord between endpoints (× fl_factor for MultExp GT protocol).
  5. Aggregate track FLs with weighted median (weight = mean vesselness).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from umud.geometry.host_faithful import (
    apo_curves,
    aponeuroses_rich,
    line_curve_intersection,
    weighted_median,
)


def refine_apo_inner(
    p_apo: np.ndarray,
    mx: float,
    my: float,
    thr: float = 0.6,
    *,
    open_k: int = 3,
    close_k: int = 5,
) -> list[dict[str, Any]] | None:
    """Rich apo with morphology-cleaned masks + DT-refined top/bot inner edges.

    Superficial inner = bot edge; deep inner = top edge (muscle-facing).
    """
    h, w = p_apo.shape
    mask0 = (p_apo > thr).astype(np.uint8)
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask0 = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask0 = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, k)
    # Use cleaned mask as probability proxy for profile
    prob = np.where(mask0 > 0, np.maximum(p_apo, thr), 0.0).astype(np.float32)
    rich = aponeuroses_rich(prob, mx, my, thr=thr * 0.5)
    if rich is None:
        rich = aponeuroses_rich(p_apo, mx, my, thr=thr)
    if rich is None:
        return None
    # DT refine: for each band component, snap top/bot to distance-transform
    # outer silhouette of the cleaned mask intersecting the band's y-range.
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask0, connectivity=8)
    if n < 2:
        return rich
    # Build per-column top/bot of entire apo mask; assign to superficial/deep by depth
    ys, xs = np.nonzero(mask0 > 0)
    if len(xs) == 0:
        return rich
    # Recompute column profiles on cleaned mask only (stricter edges)
    for bi, band in enumerate(rich):
        # approximate which CC: pick CC whose mean y closest to band centre
        bcy = float(np.mean(band["c"]))
        best_i, best_d = None, 1e9
        for i in range(1, n):
            if st[i, 4] < 20:
                continue
            cy = st[i, 1] + 0.5 * st[i, 3]
            d = abs(cy * my - bcy)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None:
            continue
        m = lab == best_i
        cols = np.unique(np.nonzero(m)[1])
        cy, top, bot = [], [], []
        for c in cols:
            yc = np.nonzero(m[:, c])[0]
            if len(yc) == 0:
                continue
            # DT-ish: erode one pixel vertically by taking interior
            # top = first on, bot = last on (already morph-cleaned)
            cy.append(float(yc.mean()))
            top.append(float(yc.min()) - 0.5)
            bot.append(float(yc.max()) + 0.5)
        if len(cols) < 5:
            continue
        X = cols.astype(float) * mx
        cy_a = np.array(cy) * my
        top_a = np.array(top) * my
        bot_a = np.array(bot) * my
        deg = 2 if X.max() - X.min() > 15 else 1
        band["X"], band["c"], band["top"], band["bot"] = X, cy_a, top_a, bot_a
        band["fit"] = {k: np.polyfit(X, getattr(band, k) if False else band[k], deg) for k in ("c", "top", "bot")}
        # fix: use band dict values
        band["fit"] = {
            "c": np.polyfit(X, cy_a, deg),
            "top": np.polyfit(X, top_a, deg),
            "bot": np.polyfit(X, bot_a, deg),
        }
        band["lo"], band["hi"] = float(X.min()), float(X.max())
    return rich


def _frangi_vesselness(gray: np.ndarray, sigmas: tuple[float, ...] = (1.0, 1.5, 2.5, 4.0)) -> np.ndarray:
    """Multi-scale Frangi vesselness (bright ridges). Returns float32 [0,1]-ish."""
    img = gray.astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    out = np.zeros_like(img)
    beta, c = 0.5, 0.5
    for s in sigmas:
        # Hessian via second Gaussian derivatives
        k = max(3, int(2 * round(3 * s) + 1))
        g = cv2.GaussianBlur(img, (k, k), s)
        # finite differences
        Ixx = cv2.Sobel(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.CV_32F, 1, 0, ksize=3)
        Iyy = cv2.Sobel(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3), cv2.CV_32F, 0, 1, ksize=3)
        Ixy = cv2.Sobel(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.CV_32F, 0, 1, ksize=3)
        # scale-normalize
        sn = s * s
        Ixx, Iyy, Ixy = Ixx * sn, Iyy * sn, Ixy * sn
        tr = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
        l1 = 0.5 * (tr + disc)  # larger | usually |
        l2 = 0.5 * (tr - disc)
        # order |λ1| <= |λ2| for vessel (λ2 large negative for bright ridge... we take abs)
        # Swap so |l1| <= |l2|
        swap = np.abs(l1) > np.abs(l2)
        l1s = np.where(swap, l2, l1)
        l2s = np.where(swap, l1, l2)
        rb = np.abs(l1s) / (np.abs(l2s) + 1e-8)  # blobness
        s2 = l1s * l1s + l2s * l2s
        # bright ridge: λ2 < 0 (in image coords dark/bright depends); use |λ2|
        v = np.exp(-(rb * rb) / (2 * beta * beta)) * (1.0 - np.exp(-s2 / (2 * c * c)))
        v = np.where(l2s <= 0, v, 0.0)  # prefer dark? US fascicles often hypoechoic streaks
        # Also try bright: take max of both conventions
        v_bright = np.exp(-(rb * rb) / (2 * beta * beta)) * (1.0 - np.exp(-s2 / (2 * c * c)))
        v_bright = np.where(l2s >= 0, v_bright, 0.0)
        v = np.maximum(v, v_bright)
        out = np.maximum(out, v)
    m = float(out.max())
    if m > 1e-8:
        out /= m
    return out.astype(np.float32)


def _orientation_field(
    field: np.ndarray, sigma_grad: float = 1.0, sigma_int: float = 2.5
) -> tuple[np.ndarray, np.ndarray]:
    """Return (angle_deg in [-90,90), coherence) maps from structure tensor on `field`."""
    sg = max(sigma_grad, 0.5)
    k = max(3, int(2 * round(3 * sg) + 1))
    Ix = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=k)
    Iy = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=k)
    si = max(sigma_int, 0.5)
    Jxx = cv2.GaussianBlur(Ix * Ix, (0, 0), si)
    Jyy = cv2.GaussianBlur(Iy * Iy, (0, 0), si)
    Jxy = cv2.GaussianBlur(Ix * Iy, (0, 0), si)
    tr = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    coher = (l1 - l2) / (l1 + l2 + 1e-8)
    grad_ang = 0.5 * np.degrees(np.arctan2(2.0 * Jxy, Jxx - Jyy + 1e-12))
    fasc_ang = ((grad_ang + 90.0 + 90) % 180) - 90  # ridge = perp to gradient
    return fasc_ang.astype(np.float32), coher.astype(np.float32)


def _belly_roi_px(
    rich: list[dict[str, Any]],
    h: int,
    w: int,
    mx: float,
    my: float,
    margin_mm: float = 0.8,
) -> np.ndarray:
    """Boolean ROI between superficial bot and deep top in map pixels."""
    sup, deep = rich[0], rich[1]
    xs = (np.arange(w) + 0.5) * mx
    ytop = np.polyval(sup["fit"]["bot"], xs) + margin_mm
    ybot = np.polyval(deep["fit"]["top"], xs) - margin_mm
    lo = max(sup["lo"], deep["lo"])
    hi = min(sup["hi"], deep["hi"])
    Y = (np.arange(h)[:, None] + 0.5) * my
    return (
        (Y >= ytop[None])
        & (Y <= ybot[None])
        & (xs[None] >= lo)
        & (xs[None] <= hi)
        & ((ybot - ytop) > 2.0)[None]
    )


def _hit_apo(
    x_mm: float,
    y_mm: float,
    rich: list[dict[str, Any]],
    tol_mm: float = 0.6,
) -> str | None:
    """Return 'sup' / 'deep' if near inner border, else None."""
    sup, deep = rich[0], rich[1]
    if not (min(sup["lo"], deep["lo"]) - 2 <= x_mm <= max(sup["hi"], deep["hi"]) + 2):
        return None
    y_sup = float(np.polyval(sup["fit"]["bot"], np.clip(x_mm, sup["lo"], sup["hi"])))
    y_deep = float(np.polyval(deep["fit"]["top"], np.clip(x_mm, deep["lo"], deep["hi"])))
    if abs(y_mm - y_sup) <= tol_mm or y_mm < y_sup:
        return "sup"
    if abs(y_mm - y_deep) <= tol_mm or y_mm > y_deep:
        return "deep"
    return None


def _track_one(
    x0_mm: float,
    y0_mm: float,
    ang0_deg: float,
    ang_map: np.ndarray,
    coh_map: np.ndarray,
    ves_map: np.ndarray,
    rich: list[dict[str, Any]],
    mx: float,
    my: float,
    *,
    step_mm: float = 0.35,
    max_steps: int = 400,
    ang_smooth: float = 0.35,
    coh_min: float = 0.05,
) -> dict[str, Any] | None:
    """Bidirectional streamline; stop at apo inners. Returns path + lengths."""
    h, w = ang_map.shape

    def sample(x_mm, y_mm):
        c = int(np.clip(x_mm / mx, 0, w - 1))
        r = int(np.clip(y_mm / my, 0, h - 1))
        return float(ang_map[r, c]), float(coh_map[r, c]), float(ves_map[r, c])

    def integrate(sign: float):
        pts = [(x0_mm, y0_mm)]
        x, y = x0_mm, y0_mm
        ang = ang0_deg
        wsum, wves = 0.0, 0.0
        for _ in range(max_steps):
            a, coh, ves = sample(x, y)
            if coh >= coh_min and np.isfinite(a):
                # unwrap toward current ang
                da = ((a - ang + 90) % 180) - 90
                ang = ang + ang_smooth * da
            rad = np.radians(ang)
            # direction: ensure we generally go toward apo depending on sign of dy preference
            dx = step_mm * np.cos(rad) * sign
            dy = step_mm * np.sin(rad) * sign
            x2, y2 = x + dx, y + dy
            # stay in image
            if not (0 <= x2 / mx < w and 0 <= y2 / my < h):
                break
            hit = _hit_apo(x2, y2, rich)
            pts.append((x2, y2))
            wsum += 1.0
            wves += ves
            x, y = x2, y2
            if hit is not None and len(pts) > 3:
                return pts, hit, wves / max(wsum, 1)
        # if exited without hit, still return
        hit = _hit_apo(x, y, rich, tol_mm=1.5)
        return pts, hit, wves / max(wsum, 1)

    # Choose sign so first step moves "up-slope" vs "down" — try both and keep better
    best = None
    for s1, s2 in ((+1, -1), (-1, +1)):
        p1, h1, v1 = integrate(s1)
        p2, h2, v2 = integrate(s2)
        hits = {h1, h2}
        if "sup" in hits and "deep" in hits:
            # merge path: reverse p1 without seed dup + p2
            path = list(reversed(p1[1:])) + p2
            if len(path) < 4:
                continue
            arr = np.asarray(path)
            # centreline path length
            d = np.sqrt(((arr[1:] - arr[:-1]) ** 2).sum(1)).sum()
            # endpoints
            # identify which end is sup/deep by y
            if arr[0, 1] < arr[-1, 1]:
                p_sup, p_deep = arr[0], arr[-1]
            else:
                p_sup, p_deep = arr[-1], arr[0]
            chord = float(np.linalg.norm(p_deep - p_sup))
            score = d * (0.5 * (v1 + v2) + 0.1)
            cand = dict(
                path=arr,
                path_len=float(d),
                chord=chord,
                p_sup=p_sup,
                p_deep=p_deep,
                weight=float(0.5 * (v1 + v2) * d),
                score=score,
            )
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best


def timtrack_fascicle_lengths(
    gray: np.ndarray,
    p_apo: np.ndarray,
    p_fasc: np.ndarray | None,
    mx: float,
    my: float,
    *,
    apo_thr: float = 0.6,
    refine_apo: bool = True,
    n_seeds: int = 12,
    fl_factor: float = 0.9,
    use_fasc_for_vessel: bool = True,
    ves_sigmas: tuple[float, ...] = (1.0, 1.5, 2.5, 4.0),
) -> dict[str, Any]:
    """Compute TimTrack-style FL summaries for one image.

    Returns dict with:
      fl_path: weighted-median path centreline length (× fl_factor)
      fl_chord: weighted-median extrapolated straight chord (× fl_factor)
      fl_chord_raw / fl_path_raw without factor
      n_tracks, ok, rich
    """
    out: dict[str, Any] = dict(
        ok=False,
        fl_path=70.0,
        fl_chord=70.0,
        fl_path_raw=70.0,
        fl_chord_raw=70.0,
        n_tracks=0,
        rich=None,
    )
    if refine_apo:
        rich = refine_apo_inner(p_apo, mx, my, thr=apo_thr)
    else:
        rich = aponeuroses_rich(p_apo, mx, my, thr=apo_thr)
    if rich is None:
        return out
    out["rich"] = rich
    h, w = p_apo.shape
    # Resize gray to map size if needed
    if gray.shape != (h, w):
        g = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        g = gray.astype(np.float32)

    ves_g = _frangi_vesselness(g, sigmas=ves_sigmas)
    if use_fasc_for_vessel and p_fasc is not None:
        # blend fasc probability as ridge prior
        ves = np.maximum(ves_g, 0.65 * p_fasc.astype(np.float32))
    else:
        ves = ves_g
    # orientation from high-pass gray + vesselness blend
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    field = 0.5 * (hp / (np.abs(hp).max() + 1e-6)) + 0.5 * ves
    ang, coh = _orientation_field(field)
    roi = _belly_roi_px(rich, h, w, mx, my)
    if int(roi.sum()) < 200:
        return out

    # Global weighted-median orientation in ROI (TimTrack-style)
    a_roi = ang[roi]
    c_roi = coh[roi]
    v_roi = ves[roi]
    wts = np.clip(c_roi, 0, None) * (0.3 + v_roi)
    if wts.sum() < 1e-6:
        wts = np.ones_like(a_roi)
    # circular-ish median via projection
    ang_med = weighted_median(a_roi, wts)

    # Seeds: grid along mid-belly
    sup, deep = rich[0], rich[1]
    lo = max(sup["lo"], deep["lo"])
    hi = min(sup["hi"], deep["hi"])
    if hi - lo < 3:
        lo, hi = min(sup["lo"], deep["lo"]), max(sup["hi"], deep["hi"])
    seeds_x = np.linspace(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo), n_seeds)
    tracks = []
    for tdepth in (0.35, 0.5, 0.65):
        for sx in seeds_x:
            y_sup = float(np.polyval(sup["fit"]["bot"], sx))
            y_deep = float(np.polyval(deep["fit"]["top"], sx))
            sy = y_sup + tdepth * (y_deep - y_sup)
            # local angle
            c = int(np.clip(sx / mx, 0, w - 1))
            r = int(np.clip(sy / my, 0, h - 1))
            a_loc = float(ang[r, c]) if coh[r, c] > 0.08 else ang_med
            # blend local with median
            da = ((a_loc - ang_med + 90) % 180) - 90
            a0 = ang_med + 0.5 * da
            tr = _track_one(sx, sy, a0, ang, coh, ves, rich, mx, my)
            if tr is None:
                continue
            if not (8.0 < tr["chord"] < 280 and 8.0 < tr["path_len"] < 350):
                continue
            tracks.append(tr)

    if not tracks:
        # fallback: single median-orientation chord between inners
        fls = []
        for t in (0.3, 0.5, 0.7):
            cx = lo + t * (hi - lo)
            cy = 0.5 * (
                float(np.polyval(sup["fit"]["bot"], cx))
                + float(np.polyval(deep["fit"]["top"], cx))
            )
            curves = apo_curves(p_apo, mx, my, thr=apo_thr)
            if len(curves) < 2:
                continue
            # use inner polys from rich
            p1 = line_curve_intersection(cx, cy, ang_med, sup["fit"]["bot"])
            p2 = line_curve_intersection(cx, cy, ang_med, deep["fit"]["top"])
            if p1 is not None and p2 is not None:
                L = float(np.linalg.norm(p1 - p2))
                if 10 < L < 300:
                    fls.append(L)
        if fls:
            chord = float(np.median(fls))
            out.update(
                ok=True,
                fl_chord_raw=chord,
                fl_path_raw=chord,
                fl_chord=chord * fl_factor,
                fl_path=chord * fl_factor,
                n_tracks=0,
                ang_med=float(ang_med),
            )
        return out

    chords = np.array([t["chord"] for t in tracks])
    paths = np.array([t["path_len"] for t in tracks])
    ww = np.array([max(t["weight"], 1e-3) for t in tracks])
    fl_c = weighted_median(chords, ww)
    fl_p = weighted_median(paths, ww)
    out.update(
        ok=True,
        fl_chord_raw=float(fl_c),
        fl_path_raw=float(fl_p),
        fl_chord=float(fl_c) * fl_factor,
        fl_path=float(fl_p) * fl_factor,
        n_tracks=len(tracks),
        ang_med=float(ang_med),
    )
    return out
