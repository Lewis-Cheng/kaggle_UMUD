"""Mask → structure dict for GeometryHead (better than rival PCA / peak-gap).

Organizer note (Paul Ritsche): muscle thickness is measured **vertical**, not
perpendicular to the aponeurosis. Pennation uses physical fascicle vs deep-apo
tangents. Fascicle length uses intersection / near-parallel reject via
``GeometryHead``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from umud.baselines.mask_proxy import proxy_apo_peak_gap, proxy_fascicle_pca
from umud.models.geometry_head import GeometryHead, GeometryHeadResult


def _fit_line_pca(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (mean_xy, tangent_xy) from 2D points, or None if too few."""
    if xs.size < 5:
        return None
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    # Ensure roughly left→right tangent
    if axis[0] < 0:
        axis = -axis
    return mean, axis


def _split_apo_bands(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split apo mask into superficial (upper) and deep (lower) bands by row.

    Uses two-peak row profile when available; else splits at median foreground row.
    """
    if mask.size == 0 or not mask.any():
        return None
    profile = mask.sum(axis=1).astype(np.float64)
    k = 5
    smooth = np.convolve(profile, np.ones(k) / k, mode="same")
    peaks: list[int] = []
    for i in range(1, len(smooth) - 1):
        if (
            smooth[i] >= smooth[i - 1]
            and smooth[i] > smooth[i + 1]
            and smooth[i] > 0.15 * smooth.max()
        ):
            peaks.append(i)
    if len(peaks) >= 2:
        peak_scores = sorted(((smooth[i], i) for i in peaks), reverse=True)
        r1, r2 = sorted([peak_scores[0][1], peak_scores[1][1]])
        mid = 0.5 * (r1 + r2)
    else:
        ys, _ = np.nonzero(mask)
        mid = float(np.median(ys))
        r1, r2 = int(ys.min()), int(ys.max())
    rows = np.arange(mask.shape[0])[:, None]
    superf = mask & (rows <= mid)
    deep = mask & (rows > mid)
    if not superf.any() or not deep.any():
        return None
    return superf, deep


def vertical_thickness_px(apo_mask: np.ndarray) -> dict[str, float | str] | None:
    """Vertical MT in pixels (organizer: vertical, not perp-to-apo)."""
    bands = _split_apo_bands(apo_mask)
    if bands is None:
        # Fall back to rival peak-gap / row span.
        return proxy_apo_peak_gap(apo_mask)
    superf, deep = bands
    ys_s, _ = np.nonzero(superf)
    ys_d, _ = np.nonzero(deep)
    # Mean row of each band → vertical gap
    gap = float(ys_d.mean() - ys_s.mean())
    if gap <= 0:
        gap = float(ys_d.min() - ys_s.max())
    if gap <= 0:
        return proxy_apo_peak_gap(apo_mask)
    return {
        "mt_gap_px": gap,
        "mt_method": "vertical_band_mean_gap",
        "row_span_px": float(max(ys_d.max(), ys_s.max()) - min(ys_d.min(), ys_s.min())),
    }


def structure_from_masks(
    *,
    fasc_mask: np.ndarray | None,
    apo_mask: np.ndarray | None,
    sx: float,
    sy: float,
) -> dict[str, Any] | None:
    """Build a GeometryHead structure dict from available masks.

    - Both masks: fascicle tangent + deep apo tangent + vertical MT.
    - Fasc only: fascicle tangent vs image-horizontal apo; FL from PCA chord;
      MT omitted (caller fills prior).
    - Apo only: deep/superficial curves for MT; PA/FL omitted.
    """
    structure: dict[str, Any] = {}

    fasc_info = proxy_fascicle_pca(fasc_mask) if fasc_mask is not None and fasc_mask.any() else None
    mt_info = vertical_thickness_px(apo_mask) if apo_mask is not None and apo_mask.any() else None

    if fasc_info is not None:
        structure["fascicle_tangent_px"] = (
            float(fasc_info["axis_x"]),
            float(fasc_info["axis_y"]),
        )
        # Direct chord also available for soft fallbacks
        structure["fascicle_length_px"] = (
            float(fasc_info["fl_length_px"]) * float(fasc_info["axis_x"]),
            float(fasc_info["fl_length_px"]) * float(fasc_info["axis_y"]),
        )

    if apo_mask is not None and apo_mask.any():
        bands = _split_apo_bands(apo_mask)
        if bands is not None:
            superf, deep = bands
            ys_s, xs_s = np.nonzero(superf)
            ys_d, xs_d = np.nonzero(deep)
            fit_s = _fit_line_pca(xs_s, ys_s)
            fit_d = _fit_line_pca(xs_d, ys_d)
            if fit_d is not None:
                _mean_d, tang_d = fit_d
                structure["deep_apo_tangent_px"] = (float(tang_d[0]), float(tang_d[1]))
                structure["deep_curve"] = np.array(
                    [
                        _mean_d - 20 * tang_d,
                        _mean_d + 20 * tang_d,
                    ]
                )
            if fit_s is not None:
                _mean_s, tang_s = fit_s
                structure["superficial_curve"] = np.array(
                    [
                        _mean_s - 20 * tang_s,
                        _mean_s + 20 * tang_s,
                    ]
                )
        if mt_info is not None:
            # Organizer: vertical thickness → thickness_vector_px = (0, gap)
            gap = float(mt_info["mt_gap_px"])
            structure["thickness_vector_px"] = (0.0, gap)
            structure["separation_mm"] = float(gap) * float(sy)

    # Fasc-only: assume deep apo ≈ horizontal in image coords (weak prior).
    if "fascicle_tangent_px" in structure and "deep_apo_tangent_px" not in structure:
        structure["deep_apo_tangent_px"] = (1.0, 0.0)

    # Need enough keys for GeometryHead; otherwise return None.
    if "fascicle_tangent_px" not in structure or "deep_apo_tangent_px" not in structure:
        return None
    if (
        "separation_mm" not in structure
        and "thickness_vector_px" not in structure
        and not (
            "superficial_curve" in structure and "deep_curve" in structure
        )
    ):
        # Fasc-only: still allow FL via fascicle_length_px + dummy MT from sy*1?
        # Prefer leaving MT to caller prior — inject tiny separation so PA/FL work
        # only when we intentionally want fasc-only PA+FL via length_px path.
        if "fascicle_length_px" in structure:
            # Provide a placeholder separation so GeometryHead can compute PA;
            # FL will use fascicle_length_px path if we pass it — GeometryHead
            # prefers fascicle_length_px when present.
            structure["separation_mm"] = max(float(sy), 1e-3)
            structure["_mt_is_placeholder"] = True
        else:
            return None

    # Silence unused sx for isotropic callers (anisotropic path uses both).
    _ = sx
    return structure


def measure_masks(
    *,
    fasc_mask: np.ndarray | None,
    apo_mask: np.ndarray | None,
    sx: float,
    sy: float,
    min_perp_component: float = 1e-3,
    priors: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Measure PA/FL/MT from masks; fill missing targets from priors.

    Returns dict with pa_deg/fl_mm/mt_mm, ok flag, failure_reason, method tags.
    Never silently drops — failures fall back to priors and are flagged.
    """
    from umud.baselines.mask_proxy import FALLBACK_PRIORS

    priors = dict(FALLBACK_PRIORS if priors is None else priors)
    head = GeometryHead(min_perp_component=min_perp_component, fail_closed=True)
    structure = structure_from_masks(
        fasc_mask=fasc_mask, apo_mask=apo_mask, sx=sx, sy=sy
    )

    out: dict[str, Any] = {
        "pa_deg": float(priors["pa_deg"]),
        "fl_mm": float(priors["fl_mm"]),
        "mt_mm": float(priors["mt_mm"]),
        "ok": False,
        "failure_reason": None,
        "used_prior_targets": [],
        "method": "geometry_head_masks",
        "scale_sx": sx,
        "scale_sy": sy,
    }

    # Always try vertical MT from apo even if full structure fails.
    if apo_mask is not None and apo_mask.any():
        mt_info = vertical_thickness_px(apo_mask)
        if mt_info is not None:
            out["mt_mm"] = float(mt_info["mt_gap_px"]) * float(sy)
            out["mt_method"] = mt_info.get("mt_method")

    if structure is None:
        out["failure_reason"] = "insufficient_masks_for_structure"
        out["used_prior_targets"] = [
            t
            for t in ("pa_deg", "fl_mm", "mt_mm")
            if t == "pa_deg"
            or t == "fl_mm"
            or (t == "mt_mm" and "mt_method" not in out)
        ]
        if "mt_method" in out:
            out["used_prior_targets"] = ["pa_deg", "fl_mm"]
        return out

    result: GeometryHeadResult = head.measure(structure, sx=sx, sy=sy)
    if result.ok:
        out["pa_deg"] = float(result.pa_deg)
        out["fl_mm"] = float(result.fl_mm)
        if structure.get("_mt_is_placeholder"):
            out["used_prior_targets"] = ["mt_mm"]
            out["mt_mm"] = float(priors["mt_mm"])
            out["ok"] = True
            out["failure_reason"] = "mt_placeholder_fasc_only"
        else:
            # Prefer vertical MT already computed when present.
            if "mt_method" not in out:
                out["mt_mm"] = float(result.mt_mm)
            out["ok"] = True
            out["failure_reason"] = None
        out["diagnostics"] = result.diagnostics
        return out

    out["failure_reason"] = result.failure_reason
    # Keep vertical MT if we have it; PA/FL stay priors.
    used = ["pa_deg", "fl_mm"]
    if "mt_method" not in out:
        used.append("mt_mm")
    out["used_prior_targets"] = used
    return out
