"""Evaluation tier separation.

Tier A — component diagnostics: mask overlap, annotated-fragment orientation
error, curve localization, transform invariants, mask-derived pseudo-measurements.

Tier B — real measurement accuracy: calibrated PA/FL/MT against independently
annotated expert measurements using the pinned scorer, including all failures.

Comparing g(M_hat) with g(M) evaluates predicted vs annotated masks under the
chosen geometry; it does NOT establish agreement with independent expert
PA/FL/MT. Keep the tiers explicitly separated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from umud.metric.score import official_score, score_components


@dataclass
class TierAReport:
    """Component diagnostics only — never promote models from Tier A alone."""

    diagnostics: dict[str, Any] = field(default_factory=dict)
    note: str = "tier_a_component_diagnostics_not_leaderboard"


@dataclass
class TierBReport:
    """Independent PA/FL/MT with pinned scorer; failures included."""

    components: dict[str, float] = field(default_factory=dict)
    official_S: float | None = None
    failure_count: int = 0
    n_images: int = 0
    note: str = "tier_b_independent_measurements_pinned_scorer"


def run_tier_a(diagnostics: Mapping[str, Any] | None = None) -> TierAReport:
    """Package component diagnostics. Stub-friendly; no invented scores."""
    return TierAReport(diagnostics=dict(diagnostics or {}))


def run_tier_b(
    errors: Mapping[str, Any] | None = None,
    *,
    failure_count: int = 0,
    n_images: int = 0,
) -> TierBReport:
    """Score independent measurement errors with the pinned official formula.

    Pass absolute-error vectors keyed by pa_deg / fl_mm / mt_mm. If errors is
    None, returns an empty report (scaffold / no labels yet).
    """
    if errors is None:
        return TierBReport(failure_count=failure_count, n_images=n_images)
    comps = score_components(errors)
    return TierBReport(
        components=comps,
        official_S=comps["S"],
        failure_count=failure_count,
        n_images=n_images,
    )


# Silence unused import warning if official_score is only re-exported conceptually.
_ = official_score
