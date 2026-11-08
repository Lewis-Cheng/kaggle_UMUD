"""Combined objective placeholders.

If valid scalar labels exist, normalized L1 (by official tolerances) is the
natural starting measurement loss — raw-unit L1 gives targets different
effective weights than the leaderboard.

Do not apply a global connected-skeleton target to disconnected annotation
fragments merely because real fascicles are continuous.
"""

from __future__ import annotations

from dataclasses import dataclass

# Documented constants only — not tuned weights. Fill after Tier-B evidence.
LOSS_WEIGHT_PLACEHOLDERS = {
    "seg": None,  # TODO: mask / conservative sparse-label loss
    "curve": None,  # TODO: apo / fascicle curve localization
    "orientation": None,  # TODO: trusted annotated-fragment orientation
    "skeleton": None,  # TODO: optional topology; ablation only
    "measurement": None,  # TODO: tolerance-normalized L1 on PA/FL/MT when valid
}


@dataclass(frozen=True)
class CombinedObjectiveSpec:
    """Named loss terms for the combined objective (weights still TBD)."""

    include_seg: bool = True
    include_curve: bool = True
    include_orientation: bool = True
    include_skeleton: bool = False  # optional ablation
    include_measurement: bool = False  # only with independent / flagged scalars

    def active_terms(self) -> list[str]:
        terms = []
        if self.include_seg:
            terms.append("seg")
        if self.include_curve:
            terms.append("curve")
        if self.include_orientation:
            terms.append("orientation")
        if self.include_skeleton:
            terms.append("skeleton")
        if self.include_measurement:
            terms.append("measurement")
        return terms
