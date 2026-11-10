"""Executable calibration spec for the texture_rl public winner.

PA and MT are affine. FL is the identity. A manifest whose declared function
or applied flag disagrees with this spec aborts. FL affine is never turned on
to make a mismatched flag succeed.
"""

from __future__ import annotations

from typing import Mapping

EXECUTABLE_CALIBRATION: dict[str, dict[str, str | bool]] = {
    "pa_deg": {"function": "affine", "applied": True},
    "fl_mm": {"function": "identity", "applied": False},
    "mt_mm": {"function": "affine", "applied": True},
}


class CalibrationSpecError(RuntimeError):
    """Manifest function or applied flag disagrees with the executable spec."""


def require_executable_calibration(
    manifest: Mapping[str, object],
) -> dict[str, tuple[float, float] | None]:
    """Return coefficients the infer path may execute.

    Affine targets return ``(a, b)``. The identity target returns ``None``
    so the caller keeps the raw value. A mismatch raises
    ``CalibrationSpecError`` and applies nothing.
    """
    out: dict[str, tuple[float, float] | None] = {}
    for key, spec in EXECUTABLE_CALIBRATION.items():
        block = manifest.get(key)
        if not isinstance(block, Mapping):
            raise CalibrationSpecError(f"missing calibration block {key}")
        function = block.get("function")
        applied = block.get("applied")
        if function != spec["function"] or applied is not spec["applied"]:
            raise CalibrationSpecError(
                f"{key}: manifest function={function!r} applied={applied!r} "
                f"disagrees with executable {spec['function']} applied={spec['applied']}"
            )
        if spec["function"] == "identity":
            out[key] = None
            continue
        if spec["function"] == "affine":
            try:
                out[key] = (float(block["a"]), float(block["b"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise CalibrationSpecError(f"{key}: affine block needs numeric a and b") from exc
            continue
        raise CalibrationSpecError(f"unknown executable function for {key}")
    return out
