"""Rival-style mask proxies (PCA fascicle / peak-gap apo). Tier A only."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def load_mask(path: Path | str) -> np.ndarray:
    """Load mask as boolean HxW (foreground = nonzero)."""
    path = Path(path)
    try:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise OSError("cv2.imread returned None")
    except Exception:
        from PIL import Image

        arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.asarray(arr) > 0


def load_gray(path: Path | str) -> np.ndarray:
    """Load image as float32 grayscale HxW in [0, 1]."""
    path = Path(path)
    try:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise OSError("cv2.imread returned None")
        return arr.astype(np.float32) / 255.0
    except Exception:
        from PIL import Image

        im = Image.open(path).convert("L")
        return np.asarray(im, dtype=np.float32) / 255.0


def proxy_fascicle_pca(mask: np.ndarray) -> dict[str, float] | None:
    """Rival-like fascicle proxy: PCA extent along principal axis (pixels).

    Returns pixel length and angle (deg) or None if empty.
    Label: PROXY — not expert FL / PA.
    """
    ys, xs = np.nonzero(mask)
    if ys.size < 5:
        return None
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0]
    proj = centered @ axis
    length_px = float(proj.max() - proj.min())
    angle_deg = float(math.degrees(math.atan2(axis[1], axis[0])))
    pa_proxy = abs(angle_deg)
    pa_proxy = min(pa_proxy, 180.0 - pa_proxy)
    return {
        "fl_length_px": length_px,
        "pa_deg_proxy": pa_proxy,
        "axis_x": float(axis[0]),
        "axis_y": float(axis[1]),
        "eigval0": float(eigvals[0]),
        "n_fg": int(ys.size),
    }


def proxy_apo_peak_gap(mask: np.ndarray) -> dict[str, float | str] | None:
    """Rival-like MT proxy: gap between two peaks of the row-sum profile (px).

    Falls back to foreground row min/max span when peaks are unclear.
    Label: PROXY — not expert MT. Organizer thickness is vertical.
    """
    if mask.size == 0 or not mask.any():
        return None
    profile = mask.sum(axis=1).astype(np.float64)
    if float(profile.max()) <= 0:
        return None
    k = 5
    kernel = np.ones(k, dtype=np.float64) / k
    smooth = np.convolve(profile, kernel, mode="same")
    peaks: list[int] = []
    for i in range(1, len(smooth) - 1):
        if (
            smooth[i] >= smooth[i - 1]
            and smooth[i] > smooth[i + 1]
            and smooth[i] > 0.2 * smooth.max()
        ):
            peaks.append(i)
    gap_px: float | None = None
    method = "row_minmax"
    if len(peaks) >= 2:
        peak_scores = sorted(((smooth[i], i) for i in peaks), reverse=True)
        i1, i2 = sorted([peak_scores[0][1], peak_scores[1][1]])
        gap_px = float(i2 - i1)
        method = "two_peak_gap"
    ys, _xs = np.nonzero(mask)
    span = float(ys.max() - ys.min()) if ys.size else 0.0
    if gap_px is None or gap_px <= 0:
        gap_px = span
        method = "row_minmax"
    return {"mt_gap_px": float(gap_px), "mt_method": method, "row_span_px": span}


def apply_isotropic_mm(
    *,
    fl_length_px: float | None,
    pa_deg: float | None,
    mt_gap_px: float | None,
    mm_per_px: float,
) -> dict[str, float | None]:
    """Convert pixel proxies to mm/deg at an isotropic scale guess."""
    if mm_per_px <= 0:
        raise ValueError("mm_per_px must be positive")
    return {
        "pa_deg": float(pa_deg) if pa_deg is not None else None,
        "fl_mm": float(fl_length_px) * mm_per_px if fl_length_px is not None else None,
        "mt_mm": float(mt_gap_px) * mm_per_px if mt_gap_px is not None else None,
    }


# Hard fallbacks used when masks fail (approx rival notebook medians).
FALLBACK_PRIORS = {
    "pa_deg": 13.0,
    "fl_mm": 14.6,
    "mt_mm": 21.5,
}
