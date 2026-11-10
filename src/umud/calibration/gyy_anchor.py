"""One vertical OCR-depth anchor, and the refusal when it does not identify G.

A depth label over a vertical ruler span is one equation for ``g_yy``.
It does not bound a length that has a horizontal component. An unbounded
length interval is a refusal, not a millimetre to submit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite_positive(value: float | None) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def assess_vertical_anchor(
    *,
    image_id: str,
    y0: float | None,
    y1: float | None,
    depth_mm: float | None,
    ocr: str,
    layout: str,
    provenance: str,
) -> dict[str, Any]:
    """Assess one vertical displacement against a depth label.

    ``y0`` and ``y1`` are ruler endpoints in pixels. ``depth_mm`` is the OCR
    length claimed for that same span. The report is ``BLOCKED`` unless the
    endpoints, the rank of ``H``, and the feasible length interval all hold.
    A vertical-only equation leaves horizontal length unbounded, so the full
    metric stays ``BLOCKED`` even when ``g_yy`` itself is a finite number.
    """
    reasons: list[str] = []
    y0_f = float(y0) if y0 is not None else float("nan")
    y1_f = float(y1) if y1 is not None else float("nan")
    depth_f = float(depth_mm) if depth_mm is not None else float("nan")
    endpoints_ok = (
        math.isfinite(y0_f)
        and math.isfinite(y1_f)
        and y1_f > y0_f
        and _finite_positive(depth_f)
        and bool(ocr.strip())
    )
    if not endpoints_ok:
        reasons.append("endpoints")

    dy = y1_f - y0_f if endpoints_ok else float("nan")
    if endpoints_ok:
        design = np.array([[0.0, 0.0, dy * dy]], dtype=np.float64)
        rank = int(np.linalg.matrix_rank(design, tol=1e-8))
        singular = np.linalg.svd(design, compute_uv=False)
    else:
        design = np.zeros((1, 3), dtype=np.float64)
        rank = 0
        singular = np.array([0.0], dtype=np.float64)
    if rank != 1:
        reasons.append("rank")

    partial_gyy = None
    vertical_interval = None
    if endpoints_ok and rank == 1:
        partial_gyy = float((depth_f * depth_f) / (dy * dy))
        vertical_interval = [depth_f, depth_f]

    # One vertical row does not constrain g_xx. A unit horizontal step is unbounded.
    horizontal_unbounded = True
    if horizontal_unbounded:
        reasons.append("feasible_interval")

    return {
        "status": "BLOCKED",
        "image_id": image_id,
        "layout": layout,
        "provenance": provenance,
        "ocr": ocr,
        "units": "mm",
        "anchor": {
            "y0_px": None if not math.isfinite(y0_f) else y0_f,
            "y1_px": None if not math.isfinite(y1_f) else y1_f,
            "displacement_px": [0.0, None if not math.isfinite(dy) else dy],
            "length_mm": None if not _finite_positive(depth_f) else depth_f,
        },
        "H": design.tolist(),
        "H_rank": rank,
        "H_columns": ["g_xx", "g_xy", "g_yy"],
        "singular_values": [float(v) for v in singular],
        "conditioning": "rank_deficient" if rank < 3 else "full_column_rank",
        "partial_gyy": partial_gyy,
        "vertical_length_interval_mm": vertical_interval,
        "horizontal_length_unbounded": True,
        "G_feasible_bounded": False,
        "reasons": reasons,
        "fills_millimetres": False,
    }
