"""Immutable manifest schema for UMUD file audit.

Missing-label-is-not-background
-------------------------------
Train a shared encoder with task-specific losses if useful, but supervise only
labels genuinely available for that image. Aponeurosis-only images must not
teach that fascicles are absent, and vice versa.

Sparse fascicle annotations introduce a second issue: even within an annotated
image, unmarked pixels may contain real fascicles. Do not treat every unmarked
pixel as a true negative without inspecting primary annotation semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

MANIFEST_FIELDS = (
    "image_id",
    "image_path",
    "mask_path",
    "task",  # "aponeurosis" | "fascicle" | "paired" | "unlabeled"
    "file_hash_image",
    "file_hash_mask",
    "content_group_id",
    "subject_id",
    "sequence_id",
    "native_height",
    "native_width",
    "transform_json",
    "calibration_provenance",  # metadata | ruler | unknown | ...
    "spacing_sx_mm_per_px",
    "spacing_sy_mm_per_px",
    "annotation_available_apo",
    "annotation_available_fasc",
    "scalar_labels_available",
)

_DEFAULTS: dict[str, Any] = {
    "mask_path": None,
    "file_hash_mask": None,
    "subject_id": None,
    "sequence_id": None,
    "native_height": None,
    "native_width": None,
    "transform_json": None,
    "spacing_sx_mm_per_px": None,
    "spacing_sy_mm_per_px": None,
    "annotation_available_apo": False,
    "annotation_available_fasc": False,
    "scalar_labels_available": False,
    "calibration_provenance": "unknown",
}


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    image_path: str
    mask_path: str | None
    task: str
    file_hash_image: str
    file_hash_mask: str | None
    content_group_id: str
    subject_id: str | None
    sequence_id: str | None
    native_height: int | None
    native_width: int | None
    transform_json: str | None
    calibration_provenance: str
    spacing_sx_mm_per_px: float | None
    spacing_sy_mm_per_px: float | None
    annotation_available_apo: bool
    annotation_available_fasc: bool
    scalar_labels_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_manifest_row(values: Mapping[str, Any]) -> ManifestRow:
    """Build a ManifestRow from a mapping; unknown keys are ignored."""
    allowed = {f.name for f in fields(ManifestRow)}
    payload: dict[str, Any] = {}
    for key in allowed:
        if key in values and values[key] is not None:
            payload[key] = values[key]
        elif key in _DEFAULTS:
            payload[key] = _DEFAULTS[key]
        else:
            payload[key] = values.get(key)

    required = (
        "image_id",
        "image_path",
        "task",
        "file_hash_image",
        "content_group_id",
    )
    for key in required:
        if payload.get(key) in (None, ""):
            raise ValueError(f"manifest missing required field: {key}")
    return ManifestRow(**payload)
