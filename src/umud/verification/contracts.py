"""Named aborting properties for the UMUD geometry search.

Each function raises ``ContractAbort`` (an ``AssertionError``) when the
property fails. Callers must not catch it and continue as if the claim held.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from umud.geometry.physical import inverse_affine_metric


class ContractAbort(AssertionError):
    """Property failed. Raised explicitly so ``python -O`` cannot drop it."""


def pullback_length_sq(u: np.ndarray, sx: float, sy: float, J: np.ndarray) -> float:
    """Length squared of ``u`` after ``u' = J u`` with metric ``G' = J^{-T} G J^{-1}``.

    ``J`` is the linear part of an original-pixel → model-pixel map.
    ``sx`` and ``sy`` are millimetres per original pixel.
    """
    G_diag_x = float(sx) ** 2
    G_diag_y = float(sy) ** 2
    u = np.asarray(u, dtype=np.float64).reshape(2)
    J = np.asarray(J, dtype=np.float64).reshape(2, 2)
    Gp = inverse_affine_metric(sx, sy, J)
    up = J @ u
    pulled = float(up @ Gp @ up)
    if not np.isfinite(pulled):
        raise ContractAbort("pullback_length_sq produced a non-finite length")
    # Touch the diagonal so a zero spacing cannot silently pass a vacuous call.
    if G_diag_x <= 0.0 or G_diag_y <= 0.0:
        raise ContractAbort("pullback_length_sq requires positive spacing")
    return pulled


def physical_cosine(u: np.ndarray, v: np.ndarray, sx: float, sy: float) -> float:
    """Cosine of the angle between ``u`` and ``v`` under ``G = diag(sx^2, sy^2)``."""
    u = np.asarray(u, dtype=np.float64).reshape(2)
    v = np.asarray(v, dtype=np.float64).reshape(2)
    Gx = float(sx) ** 2
    Gy = float(sy) ** 2
    dot = Gx * u[0] * v[0] + Gy * u[1] * v[1]
    nu = np.sqrt(Gx * u[0] ** 2 + Gy * u[1] ** 2)
    nv = np.sqrt(Gx * v[0] ** 2 + Gy * v[1] ** 2)
    if nu == 0.0 or nv == 0.0:
        raise ContractAbort("physical_cosine received a zero-length vector")
    cos_t = float(dot / (nu * nv))
    if not np.isfinite(cos_t):
        raise ContractAbort("physical_cosine produced a non-finite cosine")
    return cos_t


def assert_groups_disjoint(train_groups: Iterable[object], test_groups: Iterable[object]) -> None:
    """Train and evaluation group ids must not share a member."""
    train = set(train_groups)
    test = set(test_groups)
    overlap = train & test
    if overlap:
        shown = sorted(overlap, key=str)[:8]
        raise ContractAbort(f"group_disjoint overlap: {shown}")


def assert_scale_not_relabeled(source_provenance: str, stored_label: str) -> None:
    """A scale that was not measured must not be stored as ``measured``.

    ``family_prior`` and ``unknown`` may still be used as a registered
    estimated fallback. The forbidden step is changing the label.
    """
    source = str(source_provenance)
    stored = str(stored_label)
    if stored == "measured" and source != "measured":
        raise ContractAbort(
            f"metric_anchor: source {source!r} stored as measured"
        )


def assert_promotable(evidence: Mapping[str, object]) -> None:
    """Refuse promotion when the evidence is only a synthetic harness pass.

    Required keys must all be true. ``production_readiness`` cannot be PASS
    while ``real_images_ran`` is false.
    """
    required = (
        "scorer_parity",
        "coordinate_contract",
        "real_images_ran",
        "texture_rl_replay_matched",
        "deployment_parity",
    )
    missing = [key for key in required if not evidence.get(key)]
    if missing:
        raise ContractAbort(f"promotion_evidence missing or false: {missing}")
    readiness = evidence.get("production_readiness")
    if readiness == "PASS" and not evidence.get("real_images_ran"):
        raise ContractAbort("promotion_evidence: PASS without real_images_ran")
