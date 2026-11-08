"""Anisotropic physical length / angle helpers for ultrasound geometry.

For pixel displacement v and spacing matrix D = diag(sx, sy):
  physical vector = D v
  physical length = ||D v||

Pennation is the acute angle between the physical fascicle tangent and the
physical deep-aponeurosis tangent (clinical reference is not image horizontal).

If preprocessing maps original coords → model coords with affine linear part J,
physical displacement recovers via D J^{-1} v'. Equivalently the model-coordinate
metric is G' = J^{-T} D^T D J^{-1}.

Orientation note: a doubled-angle director (cos 2θ, sin 2θ) removes 180° sign
ambiguity, but after anisotropic augmentation the underlying tangent must be
transformed first and the director recomputed; do not rotate the director alone.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


class IntersectionRejected(ValueError):
    """Raised when fascicle/aponeurosis intersection is ill-conditioned."""


def spacing_matrix(sx: float, sy: float) -> np.ndarray:
    """Return D = diag(sx, sy) in mm/pixel."""
    if sx <= 0 or sy <= 0:
        raise ValueError("spacings must be positive")
    return np.diag([float(sx), float(sy)]).astype(np.float64)


def physical_vector(v_px: ArrayLike, sx: float, sy: float) -> np.ndarray:
    """Map pixel displacement to physical (mm) displacement via D v."""
    v = np.asarray(v_px, dtype=np.float64).reshape(2)
    return spacing_matrix(sx, sy) @ v


def physical_length(v_px: ArrayLike, sx: float, sy: float) -> float:
    """Physical length ||D v|| in millimetres."""
    return float(np.linalg.norm(physical_vector(v_px, sx, sy)))


def physical_angle_to_horizontal_deg(v_px: ArrayLike, sx: float, sy: float) -> float:
    """Angle of physical vector D v to the physical horizontal, in degrees.

    Fixture: pixel tangent (1, 1) with spacings (1, 2) → ≈ 63.4349488°.
    """
    vx, vy = physical_vector(v_px, sx, sy)
    return float(math.degrees(math.atan2(vy, vx)))


def angle_between_physical_tangents_deg(
    fascicle_px: ArrayLike,
    apo_px: ArrayLike,
    sx: float,
    sy: float,
) -> float:
    """Acute pennation angle between physical fascicle and apo tangents (degrees)."""
    vf = physical_vector(fascicle_px, sx, sy)
    va = physical_vector(apo_px, sx, sy)
    nf = np.linalg.norm(vf)
    na = np.linalg.norm(va)
    if nf == 0.0 or na == 0.0:
        raise ValueError("zero-length tangent")
    cos_t = float(np.clip(abs(np.dot(vf, va)) / (nf * na), 0.0, 1.0))
    acute = math.degrees(math.acos(cos_t))
    return float(min(acute, 180.0 - acute))


def inverse_affine_metric(sx: float, sy: float, J: np.ndarray) -> np.ndarray:
    """Model-coordinate metric G' = J^{-T} D^T D J^{-1}.

    J is the 2x2 linear part of original→model affine. Prefer mapping
    predictions back through the full inverse transform in production.
    """
    J = np.asarray(J, dtype=np.float64).reshape(2, 2)
    D = spacing_matrix(sx, sy)
    J_inv = np.linalg.inv(J)
    return J_inv.T @ (D.T @ D) @ J_inv


def director_from_theta(theta_rad: float) -> np.ndarray:
    """Doubled-angle director (cos 2θ, sin 2θ)."""
    return np.array(
        [math.cos(2.0 * theta_rad), math.sin(2.0 * theta_rad)], dtype=np.float64
    )


def theta_from_director(director: ArrayLike) -> float:
    """Recover θ ∈ (-π/2, π/2] from director (cos 2θ, sin 2θ)."""
    c2, s2 = np.asarray(director, dtype=np.float64).reshape(2)
    return 0.5 * math.atan2(s2, c2)


def recompute_director_after_anisotropic_aug(
    tangent_px: ArrayLike,
    sx: float,
    sy: float,
    linear_aug: np.ndarray,
) -> np.ndarray:
    """Transform the pixel tangent by linear_aug, then recompute the director.

    After anisotropic augmentation the original angle is not preserved; always
    transform the underlying tangent first.
    """
    A = np.asarray(linear_aug, dtype=np.float64).reshape(2, 2)
    v = A @ np.asarray(tangent_px, dtype=np.float64).reshape(2)
    # Physical orientation in the augmented frame (spacings still apply to px).
    phys = physical_vector(v, sx, sy)
    theta = math.atan2(phys[1], phys[0])
    return director_from_theta(theta)


def near_parallel_intersection_length(
    separation_mm: float,
    fascicle_unit_phys: ArrayLike,
    apo_normal_unit_phys: ArrayLike,
    *,
    min_perp_component: float = 1e-3,
) -> float:
    """Intersection length d / ε for parallel apo lines separated by d.

    ε is the fascicle unit direction's component along the apo normal.
    When |ε| < min_perp_component the intersection is rejected (ill-conditioned).

    For two parallel lines separated by d, a unit fascicle direction with
    perpendicular component ε gives intersection length d/ε. As ε→0 the length
    becomes arbitrarily long; at zero there is no intersection.
    """
    if separation_mm <= 0:
        raise ValueError("separation_mm must be positive")
    f = np.asarray(fascicle_unit_phys, dtype=np.float64).reshape(2)
    n = np.asarray(apo_normal_unit_phys, dtype=np.float64).reshape(2)
    fn = np.linalg.norm(f)
    nn = np.linalg.norm(n)
    if fn == 0.0 or nn == 0.0:
        raise ValueError("zero vector")
    f = f / fn
    n = n / nn
    eps = float(np.dot(f, n))
    if abs(eps) < min_perp_component:
        raise IntersectionRejected(
            f"near-parallel fascicle/apo (|ε|={abs(eps):.3e} < {min_perp_component})"
        )
    return float(separation_mm / abs(eps))
