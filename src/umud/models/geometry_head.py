"""Map structure outputs + anisotropic spacing to PA / FL / MT.

Physical geometry lives outside the attention / seg modules. This head is the
shared fail-closed bridge used by SegGeometryBaseline and (later) the
anatomy-query candidate.

Fail-closed policy (configs/protocol.yaml failure_accounting):
  - near-parallel fascicle/apo → reject (IntersectionRejected / failure flag)
  - missing required structure keys → reject
  - zero-length tangents → reject
  - do NOT silently drop failed images when reporting validation performance
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
    physical_vector,
)


@dataclass
class GeometryHeadResult:
    pa_deg: float | None
    fl_mm: float | None
    mt_mm: float | None
    ok: bool
    failure_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _as_vec2(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 2:
        raise ValueError(f"{name} must be length-2, got shape {np.shape(value)}")
    return arr


def _tangent_from_curve(curve_px: Any) -> np.ndarray:
    """Finite-difference tangent from a polyline [N,2] (pixel coords)."""
    pts = np.asarray(curve_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        raise ValueError("curve needs at least 2 points")
    return pts[-1] - pts[0]


def _mean_point(curve_px: Any) -> np.ndarray:
    pts = np.asarray(curve_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 1:
        raise ValueError("empty curve")
    return pts.mean(axis=0)


class GeometryHead:
    """structure dict + (sx, sy) → pa_deg, fl_mm, mt_mm."""

    def __init__(
        self,
        *,
        min_perp_component: float = 1e-3,
        fail_closed: bool = True,
    ) -> None:
        self.min_perp_component = float(min_perp_component)
        self.fail_closed = bool(fail_closed)

    def measure(
        self,
        structure: Mapping[str, Any],
        *,
        sx: float,
        sy: float,
    ) -> GeometryHeadResult:
        """Compute PA/FL/MT from structure outputs.

        Accepted structure keys (any one path per quantity):
          PA: fascicle_tangent_px + deep_apo_tangent_px
              OR fascicle_curve + deep_curve
          MT: separation_mm
              OR superficial_curve + deep_curve (mean-point displacement)
              OR thickness_vector_px
          FL: fascicle_unit_phys + apo_normal_unit_phys + separation_mm
              OR fascicle_tangent_px (+ deep normal derived) + separation/curves
              OR fascicle_length_px (direct pixel chord; converted via spacing)
        """
        try:
            pa, fl, mt, diag = self._measure_strict(structure, sx=sx, sy=sy)
            return GeometryHeadResult(
                pa_deg=pa, fl_mm=fl, mt_mm=mt, ok=True, diagnostics=diag
            )
        except (IntersectionRejected, ValueError, KeyError) as exc:
            if self.fail_closed:
                return GeometryHeadResult(
                    pa_deg=None,
                    fl_mm=None,
                    mt_mm=None,
                    ok=False,
                    failure_reason=str(exc),
                )
            # Soft policy still surfaces the failure; callers must not treat
            # NaNs as successful measurements without counting failures.
            return GeometryHeadResult(
                pa_deg=float("nan"),
                fl_mm=float("nan"),
                mt_mm=float("nan"),
                ok=False,
                failure_reason=str(exc),
            )

    def _measure_strict(
        self,
        structure: Mapping[str, Any],
        *,
        sx: float,
        sy: float,
    ) -> tuple[float, float, float, dict[str, Any]]:
        diag: dict[str, Any] = {}

        # --- tangents ---------------------------------------------------------
        if "fascicle_tangent_px" in structure and "deep_apo_tangent_px" in structure:
            fasc_t = _as_vec2("fascicle_tangent_px", structure["fascicle_tangent_px"])
            deep_t = _as_vec2("deep_apo_tangent_px", structure["deep_apo_tangent_px"])
        elif "fascicle_curve" in structure and "deep_curve" in structure:
            fasc_t = _tangent_from_curve(structure["fascicle_curve"])
            deep_t = _tangent_from_curve(structure["deep_curve"])
        else:
            raise KeyError(
                "need fascicle_tangent_px+deep_apo_tangent_px or fascicle_curve+deep_curve"
            )

        pa = angle_between_physical_tangents_deg(fasc_t, deep_t, sx, sy)
        diag["fascicle_tangent_px"] = fasc_t.tolist()
        diag["deep_apo_tangent_px"] = deep_t.tolist()

        # --- thickness --------------------------------------------------------
        if "separation_mm" in structure:
            mt = float(structure["separation_mm"])
            if mt <= 0:
                raise ValueError("separation_mm must be positive")
        elif "thickness_vector_px" in structure:
            mt = physical_length(structure["thickness_vector_px"], sx, sy)
        elif "superficial_curve" in structure and "deep_curve" in structure:
            disp = _mean_point(structure["deep_curve"]) - _mean_point(
                structure["superficial_curve"]
            )
            mt = physical_length(disp, sx, sy)
        else:
            raise KeyError(
                "need separation_mm, thickness_vector_px, or superficial_curve+deep_curve"
            )
        diag["mt_mm"] = mt

        # --- fascicle length --------------------------------------------------
        if "fascicle_length_px" in structure:
            fl = physical_length(structure["fascicle_length_px"], sx, sy)
        elif (
            "fascicle_unit_phys" in structure
            and "apo_normal_unit_phys" in structure
            and "separation_mm" in structure
        ):
            fl = near_parallel_intersection_length(
                float(structure["separation_mm"]),
                structure["fascicle_unit_phys"],
                structure["apo_normal_unit_phys"],
                min_perp_component=self.min_perp_component,
            )
        else:
            # Derive physical fascicle unit + apo normal from tangents + MT.
            fasc_phys = physical_vector(fasc_t, sx, sy)
            fn = np.linalg.norm(fasc_phys)
            if fn == 0.0:
                raise ValueError("zero-length fascicle tangent")
            fasc_unit = fasc_phys / fn
            deep_phys = physical_vector(deep_t, sx, sy)
            # Apo normal: rotate deep tangent by 90° in physical plane.
            apo_normal = np.array([-deep_phys[1], deep_phys[0]], dtype=np.float64)
            nn = np.linalg.norm(apo_normal)
            if nn == 0.0:
                raise ValueError("zero-length deep apo tangent")
            apo_normal = apo_normal / nn
            fl = near_parallel_intersection_length(
                mt,
                fasc_unit,
                apo_normal,
                min_perp_component=self.min_perp_component,
            )
            diag["derived_fascicle_unit_phys"] = fasc_unit.tolist()
            diag["derived_apo_normal_unit_phys"] = apo_normal.tolist()

        return float(pa), float(fl), float(mt), diag
