"""Calibrated segmentation + classical geometry baseline.

Verification order: reproduce this simple baseline and report measurement
error / failure rate / subgroup results / runtime BEFORE adding attention.

Pipeline (plan-aligned interface)
---------------------------------
  1. segment(image)           → structure masks / curves (or raise NotImplemented)
  2. classical_geometry(...)  → pixel-space tangents / intersections
  3. physical measurements    → pa_deg, fl_mm, mt_mm via umud.geometry

Heavy learned segmentation is intentionally NotImplemented until the data
audit + pinned scorer + duplicate-aware split gates pass. Signatures and
docstrings match the win plan so the attention candidate can share the same
geometry / failure-accounting path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from umud.geometry.physical import (
    IntersectionRejected,
    angle_between_physical_tangents_deg,
    near_parallel_intersection_length,
    physical_length,
)
from umud.models.geometry_head import GeometryHead, GeometryHeadResult


@dataclass
class SegGeometryBaselineConfig:
    """Baseline knobs (no trained weights in scaffold)."""

    min_perp_component: float = 1e-3
    fail_closed: bool = True
    # Optional path to a future checkpoint; unused until seg is implemented.
    seg_checkpoint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "SegGeometryBaselineConfig":
        if not raw:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


class SegGeometryBaseline:
    """segment → classical geometry → PA/FL/MT.

    Methods that need real masks / a trained seg net raise NotImplementedError
    with an actionable message. Classical geometry helpers are wired and usable
    from synthetic structure dicts / unit tests.
    """

    def __init__(
        self, config: SegGeometryBaselineConfig | Mapping[str, Any] | None = None
    ) -> None:
        if isinstance(config, SegGeometryBaselineConfig):
            self.config = config
        else:
            self.config = SegGeometryBaselineConfig.from_dict(config)
        self.geometry_head = GeometryHead(
            min_perp_component=self.config.min_perp_component,
            fail_closed=self.config.fail_closed,
        )

    def segment(self, image: Any) -> dict[str, Any]:
        """Return structure masks/curves for one image.

        Expected keys (plan): superficial_mask, deep_mask, fascicle_mask or
        superficial_curve, deep_curve, fascicle_curve, optionally orientation.

        Raises
        ------
        NotImplementedError
            Until a calibrated seg model is loaded after data-audit gates.
        """
        raise NotImplementedError(
            "TODO: calibrated segmentation after data audit + pinned scorer + "
            "duplicate-aware split. Do not train attention before this baseline "
            f"exists. config.seg_checkpoint={self.config.seg_checkpoint!r}"
        )

    def classical_geometry(
        self,
        structure: Mapping[str, Any],
        *,
        sx: float,
        sy: float,
    ) -> GeometryHeadResult:
        """Map structure outputs + anisotropic spacing to PA/FL/MT.

        Delegates to GeometryHead (fail-closed by default). Suitable for
        synthetic fixtures and for future seg outputs that already expose
        curves / tangents.
        """
        return self.geometry_head.measure(structure, sx=sx, sy=sy)

    def predict(self, image: Any, spacing_sx: float, spacing_sy: float) -> dict[str, float]:
        """End-to-end: segment → classical geometry → pa_deg / fl_mm / mt_mm.

        Raises NotImplementedError from ``segment`` until that stage is filled.
        On geometry failure with fail_closed=True, GeometryHead raises; callers
        must count failures rather than silently drop images.
        """
        structure = self.segment(image)
        result = self.classical_geometry(
            structure, sx=float(spacing_sx), sy=float(spacing_sy)
        )
        if not result.ok:
            raise RuntimeError(
                f"geometry failed: {result.failure_reason} "
                "(failure_accounting: include_failures_in_validation)"
            )
        return {
            "pa_deg": float(result.pa_deg),
            "fl_mm": float(result.fl_mm),
            "mt_mm": float(result.mt_mm),
        }

    # --- thin helpers exposed for tests / classical-only fixtures -------------

    @staticmethod
    def pennation_from_tangents(
        fascicle_px: tuple[float, float] | np.ndarray,
        deep_apo_px: tuple[float, float] | np.ndarray,
        sx: float,
        sy: float,
    ) -> float:
        """Acute pennation (deg) between physical fascicle and deep-apo tangents."""
        return angle_between_physical_tangents_deg(fascicle_px, deep_apo_px, sx, sy)

    @staticmethod
    def fascicle_length_parallel_apo(
        separation_mm: float,
        fascicle_unit_phys: tuple[float, float] | np.ndarray,
        apo_normal_unit_phys: tuple[float, float] | np.ndarray,
        *,
        min_perp_component: float = 1e-3,
    ) -> float:
        """Intersection length for parallel apo lines; rejects near-parallel."""
        return near_parallel_intersection_length(
            separation_mm,
            fascicle_unit_phys,
            apo_normal_unit_phys,
            min_perp_component=min_perp_component,
        )

    @staticmethod
    def muscle_thickness_mm(
        superficial_to_deep_px: tuple[float, float] | np.ndarray,
        sx: float,
        sy: float,
    ) -> float:
        """MT as physical length of a superficial→deep displacement."""
        return physical_length(superficial_to_deep_px, sx, sy)


# Re-export for callers that catch geometry rejects via the baseline module.
__all__ = [
    "SegGeometryBaseline",
    "SegGeometryBaselineConfig",
    "IntersectionRejected",
]
