"""Host-faithful PA / FL / MT from apo+fasc probability maps (Frundles geometry).

Definitions that moved public LB 0.547 → 0.336 in the reference notebook:
  * MT between **inner borders** of the aponeuroses (centreline distance minus
    half of each band thickness).
  * PA relative to the **local slope of the deep aponeurosis** (not image-x).
  * Optional PA depth-fit toward the deep apo (fascicles curve).
  * FL = straight fascicle line **extrapolated** to both apo curves; ×0.9 factor
    is applied by the caller as MultExpData / LB calibration.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

FALLBACK = dict(pa_deg=15.0, fl_mm=70.0, mt_mm=20.0)


def apo_curves(
    p_apo: np.ndarray,
    mx: float,
    my: float,
    thr: float = 0.5,
    min_rel_extent: float = 0.35,
    merge_mm: float = 4.0,
    min_gap_mm: float = 6.0,
    deg: int = 2,
) -> list[tuple[np.ndarray, float, float, float]]:
    """Return up to two apo bands as (poly, x_min, x_max, thickness_mm)."""
    h, w = p_apo.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((p_apo > thr).astype(np.uint8), connectivity=8)
    comps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(1, n):
        if st[i, 2] < 0.03 * w:
            continue
        ys, xs = np.nonzero(lab == i)
        cols = np.unique(xs)
        cy = np.array([np.average(ys[xs == c], weights=p_apo[ys[xs == c], c]) for c in cols])
        th = np.array([(xs == c).sum() for c in cols]) * my
        comps.append((cols.astype(float) * mx, cy * my, th))
    if not comps:
        return []
    xc = float(np.median(np.concatenate([c[0] for c in comps])))
    depth = [
        float(np.polyval(np.polyfit(X, Y, 1), xc)) if len(X) > 1 else float(Y.mean())
        for X, Y, _ in comps
    ]
    bands: list[list] = []
    for k in np.argsort(depth):
        if bands and depth[k] - bands[-1][0] < merge_mm:
            b = bands[-1]
            for j in (1, 2, 3):
                b[j] = np.concatenate([b[j], comps[k][j - 1]])
        else:
            bands.append([depth[k], *comps[k]])
    ext = np.array([b[1].max() - b[1].min() for b in bands])
    bands = [b for b, e in zip(bands, ext) if e >= min_rel_extent * float(ext.max())]
    out: list[tuple[np.ndarray, float, float, float]] = []
    last_d: float | None = None
    for d, X, Y, T in bands:
        if last_d is not None and d - last_d < min_gap_mm:
            continue
        dd = deg if (float(X.max()) - float(X.min())) > 15 else 1
        poly = np.polyfit(X, Y, dd)
        out.append((poly, float(X.min()), float(X.max()), float(np.median(T))))
        last_d = float(d)
        if len(out) == 2:
            break
    return out


def _column_profile(mask_comp: np.ndarray, prob: np.ndarray, cols: np.ndarray):
    """Per column: weighted centre, top edge, bottom edge (map pixels)."""
    ys, xs = np.nonzero(mask_comp)
    cy, top, bot = [], [], []
    for c in cols:
        yc = ys[xs == c]
        cy.append(np.average(yc, weights=prob[yc, c]))
        top.append(float(yc.min()) - 0.5)
        bot.append(float(yc.max()) + 0.5)
    return np.array(cy), np.array(top), np.array(bot)


def aponeuroses_rich(
    p_apo: np.ndarray,
    mx: float,
    my: float,
    thr: float = 0.5,
    merge_mm: float = 4.0,
    min_gap_mm: float = 6.0,
    min_rel_extent: float = 0.35,
) -> list[dict[str, Any]] | None:
    """Superficial+deep apo with centre/top/bot polynomial fits (mm). Mawra-compatible."""
    h, w = p_apo.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((p_apo > thr).astype(np.uint8), connectivity=8)
    comps: list[dict[str, np.ndarray]] = []
    for i in range(1, n):
        if st[i, 2] < 0.03 * w:
            continue
        m = lab == i
        cols = np.unique(np.nonzero(m)[1])
        cy, top, bot = _column_profile(m, p_apo, cols)
        comps.append(dict(X=cols.astype(float) * mx, c=cy * my, top=top * my, bot=bot * my))
    if not comps:
        return None
    xc = float(np.median(np.concatenate([c["X"] for c in comps])))
    depth = [
        float(np.polyval(np.polyfit(c["X"], c["c"], 1), xc)) if len(c["X"]) > 1 else float(c["c"].mean())
        for c in comps
    ]
    bands: list[dict] = []
    for k in np.argsort(depth):
        if bands and depth[k] - bands[-1]["d"] < merge_mm:
            b = bands[-1]
            for key in ("X", "c", "top", "bot"):
                b[key] = np.concatenate([b[key], comps[k][key]])
        else:
            bands.append(dict(d=depth[k], **comps[k]))
    ext = np.array([b["X"].max() - b["X"].min() for b in bands])
    bands = [b for b, e in zip(bands, ext) if e >= min_rel_extent * float(ext.max())]
    out: list[dict[str, Any]] = []
    for b in bands:
        if out and b["d"] - out[-1]["d"] < min_gap_mm:
            continue
        deg = 2 if b["X"].max() - b["X"].min() > 15 else 1
        b["fit"] = {k: np.polyfit(b["X"], b[k], deg) for k in ("c", "top", "bot")}
        b["lo"], b["hi"] = float(b["X"].min()), float(b["X"].max())
        out.append(b)
        if len(out) == 2:
            break
    return out if len(out) == 2 else None



def slope_deg(poly: np.ndarray, X: float | np.ndarray) -> float | np.ndarray:
    return np.degrees(np.arctan(np.polyval(np.polyder(poly), X)))


def centre_distance(sup, deep, n: int = 50) -> float:
    lo, hi = max(sup[1], deep[1]), min(sup[2], deep[2])
    if hi - lo < 3:
        lo, hi = min(sup[1], deep[1]), max(sup[2], deep[2])
    pad = 0.1 * (hi - lo)
    Xd = np.linspace(lo + pad, hi - pad, n)
    Yd = np.polyval(deep[0], Xd)
    Xs = np.linspace(lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo), 400)
    Ys = np.polyval(sup[0], Xs)
    return float(np.sqrt((Xd[:, None] - Xs[None]) ** 2 + (Yd[:, None] - Ys[None]) ** 2).min(1).mean())


def fascicle_segments(
    p_fasc: np.ndarray, mx: float, my: float, thr: float = 0.5, min_len_mm: float = 4.0
) -> list[tuple[float, float, float, float]]:
    n, lab, st, _ = cv2.connectedComponentsWithStats((p_fasc > thr).astype(np.uint8), connectivity=8)
    segs: list[tuple[float, float, float, float]] = []
    for i in range(1, n):
        if st[i, 4] < 6:
            continue
        ys, xs = np.nonzero(lab == i)
        P = np.stack([xs * mx, ys * my], 1)
        c = P.mean(0)
        _, s, vt = np.linalg.svd(P - c, full_matrices=False)
        proj = (P - c) @ vt[0]
        L = float(proj.max() - proj.min())
        if L < min_len_mm or s[0] < 3 * (s[1] + 1e-6):
            continue
        ang = (np.degrees(np.arctan2(vt[0][1], vt[0][0])) + 90) % 180 - 90
        segs.append((float(c[0]), float(c[1]), float(ang), L))
    return segs


def line_curve_intersection(cx: float, cy: float, ang: float, poly: np.ndarray) -> np.ndarray | None:
    k = np.tan(np.radians(ang))
    q = np.array(poly, dtype=float).copy()
    q[-1] -= cy - k * cx
    q[-2] -= k
    r = np.roots(q)
    r = r[np.isreal(r)].real
    if len(r) == 0:
        return None
    X = float(r[np.argmin(np.abs(r - cx))])
    return np.array([X, cy + k * (X - cx)])


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, cw[-1] / 2)])


def measure(
    p_apo: np.ndarray,
    p_fasc: np.ndarray,
    mx: float,
    my: float,
) -> dict[str, Any]:
    """Return PA/FL/MT (+ diagnostics) for one image's probability maps."""
    out: dict[str, Any] = dict(FALLBACK, n_fasc=0, ok=False, segs=np.zeros((0, 3)), apo=None)
    rich = aponeuroses_rich(p_apo, mx, my)
    out["apo"] = rich
    curves = apo_curves(p_apo, mx, my)
    if len(curves) < 2:
        return out
    # Prefer rich thickness for MT when available (centre_minus_half)
    if rich is not None:
        sup_r, deep_r = rich[0], rich[1]
        th = lambda b: float(np.median(b["bot"] - b["top"]))
        lo, hi = max(sup_r["lo"], deep_r["lo"]), min(sup_r["hi"], deep_r["hi"])
        if hi - lo < 3:
            lo, hi = min(sup_r["lo"], deep_r["lo"]), max(sup_r["hi"], deep_r["hi"])
        Xs = lo + np.array([0.2, 0.5, 0.8]) * (hi - lo)
        d = np.polyval(deep_r["fit"]["c"], Xs) - np.polyval(sup_r["fit"]["c"], Xs) - (th(sup_r) + th(deep_r)) / 2
        out["mt_mm"] = float(np.mean(d))
        # keep using simple polys for PA/FL intersections below
    sup, deep = curves[0], curves[1]
    if rich is None:
        out["mt_mm"] = centre_distance(sup, deep) - (sup[3] + deep[3]) / 2.0
    pas: list[float] = []
    fls: list[float] = []
    ws: list[float] = []
    dep: list[float] = []
    for cx, cy, ang, L in fascicle_segments(p_fasc, mx, my):
        ys_sup = float(np.polyval(sup[0], cx))
        ys_deep = float(np.polyval(deep[0], cx))
        if not (ys_sup - 2 < cy < ys_deep + 2):
            continue
        pa = abs(((ang - float(slope_deg(deep[0], np.clip(cx, deep[1], deep[2])))) + 90) % 180 - 90)
        if not (2 < pa < 60):
            continue
        p1 = line_curve_intersection(cx, cy, ang, sup[0])
        p2 = line_curve_intersection(cx, cy, ang, deep[0])
        pas.append(pa)
        ws.append(L)
        dep.append((cy - ys_sup) / (ys_deep - ys_sup) if ys_deep != ys_sup else 0.5)
        fls.append(float(np.linalg.norm(p1 - p2)) if p1 is not None and p2 is not None else float("nan"))
    out["n_fasc"] = len(pas)
    out["segs"] = np.array([dep, pas, ws]).T.reshape(-1, 3) if pas else np.zeros((0, 3))
    if pas:
        pas_a, fls_a, ws_a = map(np.asarray, (pas, fls, ws))
        out["pa_deg"] = weighted_median(pas_a, ws_a)
        good = np.isfinite(fls_a) & (fls_a > 10) & (fls_a < 300)
        if good.any():
            out["fl_mm"] = weighted_median(fls_a[good], ws_a[good])
        out["ok"] = True
    return out


def pa_depth_fit(segs: np.ndarray, at: float = 0.9, min_n: int = 5) -> float | None:
    """Fit PA vs relative depth; evaluate near deep apo (at≈0.9)."""
    if len(segs) < min_n or np.ptp(segs[:, 0]) < 0.2:
        return None
    return float(np.polyval(np.polyfit(segs[:, 0], segs[:, 1], 1, w=np.sqrt(segs[:, 2])), at))


def groups_from_hashes(hs: list[np.ndarray], max_dist: int = 12) -> np.ndarray:
    """Union-find near-duplicate frames via dHash Hamming distance."""
    parent = list(range(len(hs)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    arr = np.array(hs, dtype=np.uint8)
    for i in range(len(arr)):
        for j in np.nonzero((arr[i + 1 :] != arr[i]).sum(1) <= max_dist)[0] + i + 1:
            parent[find(int(j))] = find(i)
    return np.array([find(i) for i in range(len(hs))])


def dhash(g: np.ndarray, n: int = 16) -> np.ndarray:
    s = cv2.resize(g, (n + 1, n), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (s[:, 1:] > s[:, :-1]).ravel()
