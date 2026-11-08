"""Provenance tags for physical scale estimates."""

from __future__ import annotations

from enum import Enum


class ScaleProvenance(str, Enum):
    """How px/mm was obtained.

    measured
        Tick positions read from this image; physical pitch ``q`` still comes
        from AmbrosM device-UI semantics (e.g. major ticks = 10 mm).
    family_prior
        Fixed AmbrosM per-family formula (shape / HUD constants), optionally
        gated by a pixel fingerprint check. Not a per-image tick measure.
    unknown
        No AmbrosM family match or parser failed. Never invent TIFF dpi.
    """

    MEASURED = "measured"
    FAMILY_PRIOR = "family_prior"
    UNKNOWN = "unknown"
