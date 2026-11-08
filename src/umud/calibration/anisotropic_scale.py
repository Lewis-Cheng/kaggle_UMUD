"""Per-image anisotropic ultrasound scale (sx, sy in px/mm).

Ported from the public Frundles notebook
``ans-is-simple-physical-scale-u-net-geometry`` (Apache-2.0 / CC notebook terms),
which reported public LB 0.336 with this scale + dual U-Net + host-faithful geometry.

Devices after non-uniform resize to ~1200×800 need **separate** sx and sy.
Siemens aspect is fixed from the native 1152×864 → 1200×800 squash; Telemed
reads both rulers; Lumify uses depth OCR + probe-icon aspect.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

SIEMENS_ASPECT = (1200 / 1152) / (800 / 864)  # sx/sy after 1152x864 -> 1200x800

_ocr = None


@dataclass
class AnisoScaleRow:
    image_id: str
    layout: str
    w: int
    h: int
    sx: float  # px per mm along x
    sy: float  # px per mm along y
    aspect: float
    depth_mm: float | None = None
    tick_mm: float | None = None
    ny: int = 0
    nx: int = 0
    span: float | None = None
    ocr: str = ""
    depth_src: str = "ocr"
    provenance: str = "frundles_aniso_v1"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ocr_text(img: np.ndarray, scale: int = 3) -> str:
    """OCR a BGR/gray ROI with RapidOCR (lazy singleton)."""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr = RapidOCR()
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res, _ = _ocr(big)
    return " ".join(r[1] for r in (res or []))


def parse_depth_mm(text: str) -> float:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(cm|mm)", text, flags=re.I)
    if not m:
        return float("nan")
    v = float(m.group(1).replace(",", "."))
    return v * 10.0 if m.group(2).lower() == "cm" else v


def tick_positions(profile: np.ndarray, min_gap: int = 12, max_width: int = 6) -> np.ndarray:
    """Centres of thin bright peaks in a 1-D profile (ruler ticks)."""
    p = profile.astype(np.float32)
    on = p > max(float(p.max()) * 0.5, 40.0)
    pos: list[float] = []
    i = 0
    while i < len(p):
        if on[i]:
            j = i
            while j < len(p) and on[j]:
                j += 1
            if j - i <= max_width:
                pos.append((i + j - 1) / 2.0)
            i = j
        else:
            i += 1
    keep: list[float] = []
    for q in pos:
        if not keep or q - keep[-1] >= min_gap:
            keep.append(q)
    return np.asarray(keep, dtype=np.float64)


def regular_train(
    pos: np.ndarray, tol: float = 0.06, max_skip: int = 3
) -> tuple[float, int, float, float]:
    """Longest tick lattice tolerating missed ticks -> (spacing, n_intervals, first, last)."""
    if len(pos) < 3:
        return float("nan"), 0, float("nan"), float("nan")
    best: tuple[float, int, float, float] = (float("nan"), 0, float("nan"), float("nan"))
    key = (0, 0)
    for i in range(len(pos) - 1):
        s = float(pos[i + 1] - pos[i])
        for k in range(1, max_skip + 1):
            sk = s / k
            steps, last, hits = k, float(pos[i + 1]), 2
            for q in pos[i + 2 :]:
                m = int(round((q - last) / sk))
                if 1 <= m <= max_skip and abs(q - last - m * sk) < tol * sk * m:
                    steps += m
                    last = float(q)
                    hits += 1
                elif q - last > max_skip * sk * 1.5:
                    break
            if hits >= 3 and hits >= 0.6 * (steps + 1) and (steps, hits) > key:
                best = ((last - float(pos[i])) / steps, steps, float(pos[i]), last)
                key = (steps, hits)
    return best


def best_ruler(
    gray: np.ndarray, axis: str, band: tuple[int, int], span: tuple[int, int]
) -> tuple[float, int, float, float]:
    """Scan rows/cols in `band` for the longest regular tick train along `span`."""
    best: tuple[float, int, float, float] = (float("nan"), 0, float("nan"), float("nan"))
    lo, hi = span
    for k in range(*band):
        prof = gray[lo:hi, k] if axis == "y" else gray[k, lo:hi]
        t = regular_train(tick_positions(prof))
        if t[1] > best[1]:
            best = (t[0], t[1], t[2] + lo, t[3] + lo)
    return best


def classify_layout(path: Path, gray: np.ndarray) -> str:
    h, w = gray.shape[:2]
    if path.suffix.lower() == ".png":
        return "lumify"
    if (w, h) in ((1088, 644), (1200, 800)):
        side = float(np.median(gray[100 : h - 20, 90:140]))
        return "telemed" if 35 < side < 80 else "siemens"
    return "crop"


def lumify_icon_ratio(gray: np.ndarray) -> float:
    colmean = gray[150:600].astype(float).mean(0)
    x0 = int(np.argmax(colmean[200:700] > 25)) + 200
    roi = gray[60:250, max(x0 - 90, 0) : x0 - 3]
    n, lab, st, _ = cv2.connectedComponentsWithStats((roi > 20).astype(np.uint8))
    if n < 2:
        return float("nan")
    i = 1 + int(np.argmax(st[1:, 4]))
    cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return float("nan")
    _, (a, b), ang = cv2.fitEllipse(cs[0])
    wa, ha = (a, b) if (ang < 45 or ang > 135) else (b, a)
    return float(wa / ha) if ha > 0 else float("nan")


def measure_scale(path: Path | str) -> dict[str, Any]:
    """Measure raw scale fields for one image (before cross-image fill/snap)."""
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lay = classify_layout(path, gray)
    r: dict[str, Any] = dict(
        image_id=path.name,
        layout=lay,
        w=w,
        h=h,
        sx=float("nan"),
        sy=float("nan"),
        aspect=float("nan"),
        depth_mm=float("nan"),
        tick_mm=float("nan"),
        ny=0,
        nx=0,
        span=float("nan"),
        ocr="",
        depth_src="ocr",
        provenance="frundles_aniso_v1",
    )
    if lay == "siemens":
        r["ocr"] = ocr_text(bgr[90 : h - 60, 960:1100])
        s, n, a, b = best_ruler(gray, "y", (w - 80, w), (40, h - 40))
        r.update(depth_mm=parse_depth_mm(r["ocr"]), ny=n, span=b - a, aspect=SIEMENS_ASPECT)
    elif lay == "telemed":
        r["ocr"] = ocr_text(bgr[40:140, w - 110 : w])
        r["depth_mm"] = parse_depth_mm(r["ocr"])
        band = (0, 20) if w == 1088 else (40, 110)
        s, n, a, b = best_ruler(gray, "y", band, (0, h))
        r.update(sy=s / 10.0, ny=n, tick_mm=10.0)
        s, n, a, b = best_ruler(gray, "x", (h - 12, h), (130, w - 130))
        r.update(sx=s / 10.0, nx=n)
        r["aspect"] = r["sx"] / r["sy"] if r["sy"] and r["sy"] == r["sy"] else float("nan")
    elif lay == "crop":
        s, n, a, b = best_ruler(gray, "x", (h - 8, h), (0, w))
        r.update(sx=s / 10.0, sy=s / 10.0, nx=n, tick_mm=10.0, aspect=1.0)
    elif lay == "lumify":
        r["ocr"] = ocr_text(bgr[60:600, w - 120 : w])
        s, n, a, b = best_ruler(gray, "y", (0, 30), (90, h - 40))
        r.update(
            depth_mm=parse_depth_mm(r["ocr"]),
            ny=n,
            span=b - a,
            aspect=lumify_icon_ratio(gray),
        )
    return r


def fill_depth_rulers(df: pd.DataFrame) -> pd.DataFrame:
    """siemens/lumify: sy = ruler span / depth; missing OCR from identical-ruler LUT."""
    for lay in ("siemens", "lumify"):
        m = df.layout == lay
        def key(g: pd.DataFrame) -> list[tuple[Any, Any]]:
            return list(zip(g.ny, (g.span / 4).round()))

        ok = m & df.depth_mm.notna()
        lut: dict[tuple[Any, Any], list[float]] = {}
        for k, D in zip(key(df[ok]), df.depth_mm[ok]):
            lut.setdefault(k, []).append(float(D))
        for i, k in zip(df.index[m & df.depth_mm.isna()], key(df[m & df.depth_mm.isna()])):
            vals = lut.get(k, [])
            if vals and len(set(vals)) == 1:
                df.loc[i, ["depth_mm", "depth_src"]] = vals[0], "lut"
        df.loc[m, "sy"] = df.span[m] / df.depth_mm[m]
        df.loc[m, "sx"] = df.sy[m] * df.aspect[m]
        df.loc[m, "tick_mm"] = df.depth_mm[m] / df.ny[m]
    return df


def fill_crops(df: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    """Crops of one capture share a scale; replace values >tol off their group median."""
    crop = df.layout == "crop"
    grp = (df.h / 10).round()
    med = df[crop].groupby(grp[crop]).sx.median()
    for i in df.index[crop]:
        m = med[grp[i]]
        if not abs(df.sx[i] / m - 1) < tol:
            df.loc[i, ["sx", "sy"]] = m
    return df


def snap_odd_ticks(
    df: pd.DataFrame, ok_ticks: Sequence[float] = (2.0, 2.5, 5.0, 10.0)
) -> pd.DataFrame:
    """Partly detected rulers → non-standard tick size; use (layout, depth) median."""
    std = df.tick_mm.round(2).isin(ok_ticks)
    med = df[std].groupby(["layout", "depth_mm"])[["sx", "sy"]].median()
    for i in df.index[~std & df.depth_mm.notna()]:
        k = (df.layout[i], df.depth_mm[i])
        if k in med.index:
            df.loc[i, ["sx", "sy"]] = med.loc[k].values
    return df


def snap_device_aspect(df: pd.DataFrame, tol: float = 0.03) -> pd.DataFrame:
    """Telemed frames of one resolution share one pixel aspect."""
    tm = df.layout == "telemed"
    med = df[tm].groupby(["w", "h"]).aspect.median()
    for i in df.index[tm]:
        m = med[(df.w[i], df.h[i])]
        if not abs(df.aspect[i] / m - 1) < tol:
            df.loc[i, "sx"] = df.sy[i] * m
            df.loc[i, "aspect"] = m
    return df


def estimate_anisotropic_scales(
    paths: Iterable[Path | str],
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Run full Frundles scale pipeline on a list of image paths → DataFrame."""
    rows = []
    paths = list(paths)
    for i, p in enumerate(paths):
        if show_progress and (i % 25 == 0 or i + 1 == len(paths)):
            print(f"scale {i+1}/{len(paths)} {Path(p).name}", flush=True)
        rows.append(measure_scale(p))
    df = pd.DataFrame(rows)
    df = snap_device_aspect(snap_odd_ticks(fill_crops(fill_depth_rulers(df))))
    return df


def mm_per_prob_pixel(sx_px_per_mm: float, sy_px_per_mm: float, W0: int, H0: int, w: int, h: int) -> tuple[float, float]:
    """Map original-image sx/sy (px/mm) to mm-per-probability-map-pixel (mx, my)."""
    mx = (W0 / w) / sx_px_per_mm
    my = (H0 / h) / sy_px_per_mm
    return float(mx), float(my)
