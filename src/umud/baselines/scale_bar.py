"""Vertical scale-bar / tick detection for ultrasound frames.

Organizer (Paul Ritsche): vertical scale bars give pix/cm or mm ratio (image
depth). Aspect ratios preserved; some train images (public DL_Track mix) have
NO scale bar — leave provenance unknown for those.

This module never treats TIFF print-dpi as ultrasound spacing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ScaleBarDetection:
    found: bool
    side: str | None = None  # "left" | "right"
    tick_spacing_px: float | None = None
    n_ticks: int = 0
    mm_per_px: float | None = None
    provenance: str = "unknown"
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _margin_strip(gray: np.ndarray, side: str, width_frac: float = 0.08) -> np.ndarray:
    h, w = gray.shape[:2]
    width = max(4, int(round(w * width_frac)))
    if side == "right":
        return gray[:, -width:]
    return gray[:, :width]


def _detect_ticks_in_strip(strip: np.ndarray) -> dict[str, Any] | None:
    """Detect roughly periodic bright horizontal ticks in a vertical margin."""
    if strip.size == 0:
        return None
    # Column-mean → emphasize ticks that span the margin width.
    # Row energy: max across columns (bright ticks).
    row_energy = strip.max(axis=1).astype(np.float64)
    # Normalize
    re = row_energy - np.median(row_energy)
    if re.max() <= 0:
        return None
    # Threshold relative to robust scale
    mad = np.median(np.abs(re - np.median(re))) + 1e-6
    thr = max(3.0 * mad, 0.15 * (re.max() if re.max() > 0 else 1.0))
    peaks = []
    for i in range(1, len(re) - 1):
        if re[i] >= re[i - 1] and re[i] > re[i + 1] and re[i] > thr:
            peaks.append(i)
    if len(peaks) < 3:
        return None
    gaps = np.diff(np.asarray(peaks, dtype=np.float64))
    # Keep gaps near the median (reject outliers)
    med = float(np.median(gaps))
    if med < 2.0:
        return None
    keep = gaps[(gaps > 0.6 * med) & (gaps < 1.4 * med)]
    if keep.size < 2:
        return None
    spacing = float(np.median(keep))
    return {
        "n_ticks": int(len(peaks)),
        "tick_spacing_px": spacing,
        "gap_std": float(keep.std()),
        "peaks_sample": peaks[:20],
    }


def detect_vertical_scale_bar(
    gray: np.ndarray,
    *,
    mm_per_tick: float | None = None,
) -> ScaleBarDetection:
    """Detect vertical tick marks on left/right margins.

    If ``mm_per_tick`` is provided (from organizer notes / depth legend),
    returns mm_per_px = mm_per_tick / tick_spacing_px with provenance
    ``scale_bar_ticks+mm_per_tick``. Otherwise provenance stays
    ``scale_bar_ticks_detected_physical_unknown`` when ticks are found, or
    ``unknown`` when absent.
    """
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    # Ensure [0,1] or [0,255] both OK — relative thresholds used.
    candidates: list[tuple[str, dict[str, Any]]] = []
    for side in ("right", "left"):
        strip = _margin_strip(gray, side)
        det = _detect_ticks_in_strip(strip)
        if det is not None:
            candidates.append((side, det))
    if not candidates:
        return ScaleBarDetection(
            found=False,
            provenance="unknown",
            notes=["no_periodic_vertical_ticks_in_margins"],
        )
    # Prefer more ticks, then smaller gap_std
    side, det = max(
        candidates, key=lambda sd: (sd[1]["n_ticks"], -sd[1]["gap_std"])
    )
    spacing = float(det["tick_spacing_px"])
    if mm_per_tick is not None and mm_per_tick > 0 and spacing > 0:
        return ScaleBarDetection(
            found=True,
            side=side,
            tick_spacing_px=spacing,
            n_ticks=int(det["n_ticks"]),
            mm_per_px=float(mm_per_tick) / spacing,
            provenance="scale_bar_ticks+mm_per_tick",
            notes=[f"mm_per_tick={mm_per_tick}"],
            diagnostics=det,
        )
    return ScaleBarDetection(
        found=True,
        side=side,
        tick_spacing_px=spacing,
        n_ticks=int(det["n_ticks"]),
        mm_per_px=None,
        provenance="scale_bar_ticks_detected_physical_unknown",
        notes=[
            "ticks_found_but_physical_tick_pitch_unknown",
            "do_not_use_tiff_dpi",
        ],
        diagnostics=det,
    )
