"""AmbrosM / xiaoleilian per-family ultrasound scale parsers.

Ported from public Kaggle notebooks (Apache 2.0):
  - ambrosm/umud-quick-and-dirty ``collect_metadata``
  - xiaoleilian/umud-hardcore-eda ``ambrosm_test_scale`` + ``family_of``

Soft-fails instead of ``assert False``. Never uses TIFF print-dpi as spacing.
Absolute scale still needs AmbrosM's device-UI ``q`` (mm per tick or fixed
cm span); that ``q`` is tagged family_prior even when tick *spacing* is measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from umud.calibration.provenance import ScaleProvenance

# AmbrosM right-tick FOV table: n_ticks → (l, t, r); b = last_tick
_RIGHT_TICK_FOV: dict[int, tuple[int, int, int]] = {
    7: (142, 91, 1058),
    8: (163, 91, 1037),
    9: (211, 91, 989),
    10: (249, 91, 951),
    11: (282, 91, 918),
    12: (308, 91, 892),
    13: (331, 91, 869),
    14: (349, 91, 851),
    15: (142, 91, 1058),
}


@dataclass
class ScaleEstimate:
    family: str
    provenance: str
    px_per_cm: float | None = None
    px_per_mm: float | None = None
    mm_per_px: float | None = None
    tick_spacing_px: float | None = None
    assumed_mm_per_tick: float | None = None
    q_provenance: str = ScaleProvenance.UNKNOWN.value
    fov_l: int | None = None
    fov_t: int | None = None
    fov_r: int | None = None
    fov_b: int | None = None
    fov_w_mm: float | None = None
    fov_h_mm: float | None = None
    scale_note: str = ""
    height: int = 0
    width: int = 0
    ndim: int = 0
    suffix: str = ""
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def load_image_array(path: Path | str) -> np.ndarray:
    """Load image as numpy array preserving RGB vs grayscale (AmbrosM style)."""
    from PIL import Image

    path = Path(path)
    with Image.open(path) as img:
        img.load()
        return np.asarray(img)


def classify_family(
    *,
    height: int,
    width: int,
    ndim: int,
    suffix: str,
    array: np.ndarray | None = None,
) -> str:
    """HUD / shape family label (xiaoleilian ``family_of`` + train hflip)."""
    suf = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    if suf == ".png":
        return "png_lumify"
    if (height, width) == (800, 1200):
        if array is not None and array.ndim == 3 and height >= 90 and width >= 1160:
            rt = float(array[87, 1147:1157].mean())
            lt = float(array[42, 67:74, 0].mean()) if array.shape[2] >= 1 else 0.0
            # AmbrosM: train may be horizontally flipped → right ticks land near x=49
            rt_m = float(array[87, 43:53].mean())
            lt_m = float(array[42, 1126:1133, 0].mean()) if array.shape[2] >= 1 else 0.0
            if rt > 100:
                return "rgb_800x1200_right_ticks"
            if lt > 100:
                return "rgb_800x1200_left_ticks"
            if rt_m > 100:
                return "rgb_800x1200_right_ticks_hflip"
            if lt_m > 100:
                return "rgb_800x1200_left_ticks_hflip"
            # Soft: bright tick column at mirrored right-tick x without HUD fingerprint
            col49 = array[:, 49].mean(axis=-1) if array.ndim == 3 else array[:, 49]
            if float(np.max(col49)) > 120:
                return "rgb_800x1200_right_ticks_hflip"
        return "rgb_800x1200_other"
    if (height, width) == (644, 1088):
        return "rgb_644x1088"
    if height in (512, 513):
        return f"h{int(height)}_bottom_ticks"
    if height == 853 and ndim == 2:
        return "gray_853_bottom_ticks"
    mode = "RGB" if ndim == 3 else "L"
    return f"other_{int(height)}x{int(width)}_{mode}"


def _finalize(
    *,
    family: str,
    provenance: ScaleProvenance,
    px_per_cm: float,
    tick_spacing_px: float | None,
    assumed_mm_per_tick: float | None,
    q_provenance: ScaleProvenance,
    fov: tuple[int, int, int, int] | None,
    scale_note: str,
    height: int,
    width: int,
    ndim: int,
    suffix: str,
    notes: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> ScaleEstimate:
    notes = list(notes or [])
    notes.append("do_not_use_tiff_dpi")
    px_per_mm = px_per_cm / 10.0 if px_per_cm and px_per_cm > 0 else None
    mm_per_px = (1.0 / px_per_mm) if px_per_mm and px_per_mm > 0 else None
    fov_l = fov_t = fov_r = fov_b = None
    fov_w_mm = fov_h_mm = None
    if fov is not None:
        fov_l, fov_t, fov_r, fov_b = (int(x) for x in fov)
        if px_per_mm and px_per_mm > 0 and fov_r > fov_l and fov_b > fov_t:
            fov_w_mm = (fov_r - fov_l) / px_per_mm
            fov_h_mm = (fov_b - fov_t) / px_per_mm
    return ScaleEstimate(
        family=family,
        provenance=provenance.value,
        px_per_cm=float(px_per_cm) if px_per_cm and px_per_cm > 0 else None,
        px_per_mm=px_per_mm,
        mm_per_px=mm_per_px,
        tick_spacing_px=tick_spacing_px,
        assumed_mm_per_tick=assumed_mm_per_tick,
        q_provenance=q_provenance.value,
        fov_l=fov_l,
        fov_t=fov_t,
        fov_r=fov_r,
        fov_b=fov_b,
        fov_w_mm=fov_w_mm,
        fov_h_mm=fov_h_mm,
        scale_note=scale_note,
        height=height,
        width=width,
        ndim=ndim,
        suffix=suffix,
        notes=notes,
        diagnostics=dict(diagnostics or {}),
    )


def _unknown(
    *,
    family: str,
    height: int,
    width: int,
    ndim: int,
    suffix: str,
    scale_note: str,
    notes: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> ScaleEstimate:
    return ScaleEstimate(
        family=family,
        provenance=ScaleProvenance.UNKNOWN.value,
        q_provenance=ScaleProvenance.UNKNOWN.value,
        scale_note=scale_note,
        height=height,
        width=width,
        ndim=ndim,
        suffix=suffix,
        notes=list(notes or []) + ["do_not_use_tiff_dpi"],
        diagnostics=dict(diagnostics or {}),
    )


def _col_mean(a: np.ndarray, col: int) -> np.ndarray:
    if a.ndim == 3:
        return a[:, col].mean(axis=-1).astype(np.float64)
    return a[:, col].astype(np.float64)


def _parse_png_lumify(a: np.ndarray) -> ScaleEstimate | None:
    """Lumify PNG: left-column major/minor ticks; q encodes 10 mm interval."""
    try:
        col6 = _col_mean(a, 6)
        col9 = _col_mean(a, 9)
        first_tick = int(np.argmax(col6 > 50))
        second_minor = 150 + int(np.argmax(col6[150:] > 50))
        second_major = 150 + int(np.argmax(col9[150:] > 50))
        last_tick = len(a) - 1 - int(np.argmax(col6[::-1] > 50))
        if (second_major - first_tick) < 3 * (second_minor - first_tick):
            # major ticks 10 mm, minor 5 mm → px_per_cm = major gap
            px_per_cm = float(second_major - first_tick)
            tick_spacing = float(second_major - first_tick)
            q = 10.0
            note = "png_major_10mm"
        else:
            # major 50 mm, minor 10 mm → px_per_cm = minor gap
            px_per_cm = float(second_minor - first_tick)
            tick_spacing = float(second_minor - first_tick)
            q = 10.0
            note = "png_minor_10mm"
        if px_per_cm <= 0:
            return None
        hw = a.shape[1] // 2
        if a.ndim == 3:
            s = a[:, hw:].sum(axis=(0, 2))
        else:
            s = a[:, hw:].sum(axis=0)
        w2 = int(np.argmin(s))
        fov = (hw - w2, first_tick, hw + w2, last_tick)
        notes = []
        if px_per_cm > 0:
            width_cm = (fov[2] - fov[0]) / px_per_cm
            if not (2.8 < width_cm < 2.9):
                notes.append(f"fov_width_cm_outside_2.85_band={width_cm:.3f}")
        return _finalize(
            family="png_lumify",
            provenance=ScaleProvenance.MEASURED,
            px_per_cm=px_per_cm,
            tick_spacing_px=tick_spacing,
            assumed_mm_per_tick=q,
            q_provenance=ScaleProvenance.FAMILY_PRIOR,
            fov=fov,
            scale_note=note,
            height=a.shape[0],
            width=a.shape[1],
            ndim=a.ndim,
            suffix=".png",
            notes=notes,
            diagnostics={
                "first_tick": first_tick,
                "second_minor": second_minor,
                "second_major": second_major,
                "last_tick": last_tick,
            },
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail AmbrosM asserts
        return _unknown(
            family="png_lumify",
            height=a.shape[0],
            width=a.shape[1],
            ndim=a.ndim,
            suffix=".png",
            scale_note=f"png_err:{exc}",
        )



def _parse_right_ticks_at_column(
    a: np.ndarray,
    col: int,
    *,
    family: str,
    scale_note_prefix: str,
    hflip: bool = False,
) -> ScaleEstimate | None:
    """AmbrosM right-border tick math on an arbitrary vertical column.

    Used for native right ticks (col≈1150) and train hflip (col≈49).
    ``q`` remains AmbrosM UI prior (5 mm if n<=14, 2 mm if n==15).
    """
    h, w = a.shape[:2]
    if a.ndim != 3:
        return None
    col = int(col)
    if col < 0 or col >= w:
        return None
    colv = a[:, col].mean(axis=-1).astype(np.float64)
    if float(colv.max()) < 50:
        return None
    first_tick = int(np.argmax(colv > 50))
    if first_tick + 25 >= h:
        return None
    second_major = first_tick + 20 + int(np.argmax(colv[first_tick + 20 :] > 50))
    last_tick = len(a) - 1 - int(np.argmax(colv[::-1] > 50))
    gap = max(second_major - first_tick, 1)
    n_ticks_f = (last_tick - first_tick) / gap
    n = int(round(n_ticks_f))
    if 7 <= n <= 14:
        px_per_cm = (last_tick - first_tick) / n * 2.0
        q = 5.0
        note = f"{scale_note_prefix} n={n}"
    elif n == 15:
        px_per_cm = (last_tick - first_tick) / 3.0
        q = 2.0
        note = f"{scale_note_prefix} n=15"
    else:
        return None
    fov = None
    if n in _RIGHT_TICK_FOV:
        l, t, r = _RIGHT_TICK_FOV[n]
        # AmbrosM FOV table is horizontally centered (l+r==width); hflip keeps it.
        if hflip:
            l, r = w - r, w - l
        fov = (l, t, r, last_tick)
    notes = ["hflip_train"] if hflip else []
    if fov is not None and px_per_cm > 0:
        width_cm = (fov[2] - fov[0]) / px_per_cm
        if not (5.7 < width_cm < 5.8):
            notes.append(f"fov_width_cm_outside_5.75_band={width_cm:.3f}")
    return _finalize(
        family=family,
        provenance=ScaleProvenance.MEASURED,
        px_per_cm=float(px_per_cm),
        tick_spacing_px=float(gap),
        assumed_mm_per_tick=q,
        q_provenance=ScaleProvenance.FAMILY_PRIOR,
        fov=fov,
        scale_note=note,
        height=h,
        width=w,
        ndim=3,
        suffix=".tif",
        notes=notes,
        diagnostics={
            "first_tick": first_tick,
            "second_major": second_major,
            "last_tick": last_tick,
            "n_ticks": n,
            "tick_col": col,
            "hflip": hflip,
        },
    )


def _parse_800x1200(a: np.ndarray) -> ScaleEstimate:
    h, w = a.shape[:2]
    if a.ndim != 3:
        return _unknown(
            family="rgb_800x1200_other",
            height=h,
            width=w,
            ndim=a.ndim,
            suffix=".tif",
            scale_note="800x1200_not_rgb",
        )
    # Right ticks (native)
    right_hit = (a[87, 1147:1157] == 175).all() or float(a[87, 1147:1157].mean()) > 150
    if right_hit:
        est = _parse_right_ticks_at_column(
            a, 1150, family="rgb_800x1200_right_ticks", scale_note_prefix="800x1200_right"
        )
        if est is not None:
            return est
        return _unknown(
            family="rgb_800x1200_right_ticks",
            height=h,
            width=w,
            ndim=3,
            suffix=".tif",
            scale_note="800x1200_right_bad_n",
        )
    # Left ticks — AmbrosM fixed 5 cm span
    if (a[42, 67:74, 0] > 115).all():
        px_per_cm = (783 - 42) / 5.0
        return _finalize(
            family="rgb_800x1200_left_ticks",
            provenance=ScaleProvenance.FAMILY_PRIOR,
            px_per_cm=float(px_per_cm),
            tick_spacing_px=(783 - 42) / 5.0,
            assumed_mm_per_tick=10.0,
            q_provenance=ScaleProvenance.FAMILY_PRIOR,
            fov=(171, 42, 1029, 798),
            scale_note="800x1200_left",
            height=h,
            width=w,
            ndim=3,
            suffix=".tif",
            notes=["fixed_span_5cm_over_px_42_to_783"],
            diagnostics={"y0": 42, "y1": 783, "span_cm": 5.0},
        )
    # Train hflip: right ticks mirrored to ~x=49 (AmbrosM comment; not in original asserts)
    for col in (49, 50, 48, 51, 47, 52):
        est = _parse_right_ticks_at_column(
            a,
            col,
            family="rgb_800x1200_right_ticks_hflip",
            scale_note_prefix="800x1200_right_hflip",
            hflip=True,
        )
        if est is not None:
            return est
    # Train hflip: left-tick HUD mirrored to x≈1126..1133; y-span unchanged
    if (a[42, 1126:1133, 0] > 115).all() or float(a[42, 1126:1133, 0].mean()) > 100:
        px_per_cm = (783 - 42) / 5.0
        # FOV table is centered; mirror is identical for (171,1029) on W=1200
        return _finalize(
            family="rgb_800x1200_left_ticks_hflip",
            provenance=ScaleProvenance.FAMILY_PRIOR,
            px_per_cm=float(px_per_cm),
            tick_spacing_px=(783 - 42) / 5.0,
            assumed_mm_per_tick=10.0,
            q_provenance=ScaleProvenance.FAMILY_PRIOR,
            fov=(171, 42, 1029, 798),
            scale_note="800x1200_left_hflip",
            height=h,
            width=w,
            ndim=3,
            suffix=".tif",
            notes=["hflip_train", "fixed_span_5cm_over_px_42_to_783"],
            diagnostics={"y0": 42, "y1": 783, "span_cm": 5.0, "hflip": True},
        )
    return _unknown(
        family="rgb_800x1200_other",
        height=h,
        width=w,
        ndim=3,
        suffix=".tif",
        scale_note="800x1200_unparsed",
        notes=["no_ambrosm_or_hflip_match"],
    )

def _parse_644x1088(a: np.ndarray) -> ScaleEstimate:
    h, w = a.shape[:2]
    px_per_cm = 630.5 / 5.0
    return _finalize(
        family="rgb_644x1088",
        provenance=ScaleProvenance.FAMILY_PRIOR,
        px_per_cm=float(px_per_cm),
        tick_spacing_px=630.5 / 5.0,
        assumed_mm_per_tick=10.0,
        q_provenance=ScaleProvenance.FAMILY_PRIOR,
        fov=(140, 0, 947, 643),
        scale_note="644x1088",
        height=h,
        width=w,
        ndim=a.ndim,
        suffix=".tif",
        notes=["fixed_span_5cm_over_630.5px"],
    )


def _parse_h512_bottom(a: np.ndarray) -> ScaleEstimate:
    h, w = a.shape[:2]
    family = f"h{h}_bottom_ticks"
    if a.ndim != 3:
        return _unknown(
            family=family,
            height=h,
            width=w,
            ndim=a.ndim,
            suffix=".tif",
            scale_note=f"h{h}_not_rgb",
        )
    # AmbrosM fingerprint gates (soft)
    ok = False
    try:
        if a[-5, 49, 0] == 168 and a[-5, 438, 0] == 168:
            ok = True
        elif a[-5, 52, 0] == 168 and a[-5, 441, 0] == 168:
            ok = True
        elif a[-5, 53, 0] == 168 and a[-5, 442, 0] == 168:
            ok = True
    except Exception:
        ok = False
    px_per_cm = (442 - 53) / 5.0
    notes = []
    if not ok:
        notes.append("bottom_tick_fingerprint_miss_applied_family_prior_anyway")
    return _finalize(
        family=family,
        provenance=ScaleProvenance.FAMILY_PRIOR,
        px_per_cm=float(px_per_cm),
        tick_spacing_px=(442 - 53) / 5.0,
        assumed_mm_per_tick=10.0,
        q_provenance=ScaleProvenance.FAMILY_PRIOR,
        fov=(0, 0, w, h - 10),
        scale_note=f"h{h}_bottom",
        height=h,
        width=w,
        ndim=3,
        suffix=".tif",
        notes=notes + ["fixed_span_5cm_over_px_53_to_442"],
        diagnostics={"fingerprint_ok": ok},
    )


def _parse_gray_853(a: np.ndarray) -> ScaleEstimate:
    h, w = a.shape[:2]
    if a.ndim != 2:
        return _unknown(
            family="gray_853_bottom_ticks",
            height=h,
            width=w,
            ndim=a.ndim,
            suffix=".tif",
            scale_note="h853_not_gray",
        )
    px_per_cm = None
    note = "h853"
    diag: dict[str, Any] = {}
    try:
        if a[-5, 100] == 170 and a[-5, 934] == 170:
            px_per_cm = (934 - 100) / 5.0
            note = "h853_100_934"
            diag = {"x0": 100, "x1": 934}
        elif a[-5, 44] == 170 and a[-5, 879] == 170:
            px_per_cm = (879 - 44) / 5.0
            note = "h853_44_879"
            diag = {"x0": 44, "x1": 879}
    except Exception as exc:  # noqa: BLE001
        return _unknown(
            family="gray_853_bottom_ticks",
            height=h,
            width=w,
            ndim=2,
            suffix=".tif",
            scale_note=f"h853_err:{exc}",
        )
    if px_per_cm is None:
        return _unknown(
            family="gray_853_bottom_ticks",
            height=h,
            width=w,
            ndim=2,
            suffix=".tif",
            scale_note="h853_fingerprint_miss",
        )
    return _finalize(
        family="gray_853_bottom_ticks",
        provenance=ScaleProvenance.FAMILY_PRIOR,
        px_per_cm=float(px_per_cm),
        tick_spacing_px=float(px_per_cm),
        assumed_mm_per_tick=10.0,
        q_provenance=ScaleProvenance.FAMILY_PRIOR,
        fov=(0, 0, w, h),
        scale_note=note,
        height=h,
        width=w,
        ndim=2,
        suffix=".tif",
        notes=["fixed_span_5cm_bottom_markers"],
        diagnostics=diag,
    )


def estimate_scale(path: Path | str, array: np.ndarray | None = None) -> ScaleEstimate:
    """Estimate px/mm for one ultrasound frame via AmbrosM family parsers."""
    path = Path(path)
    suffix = path.suffix.lower()
    a = array if array is not None else load_image_array(path)
    if a.ndim not in (2, 3):
        return _unknown(
            family="unknown",
            height=0,
            width=0,
            ndim=int(a.ndim),
            suffix=suffix,
            scale_note=f"bad_ndim={a.ndim}",
        )
    h, w = int(a.shape[0]), int(a.shape[1])
    ndim = int(a.ndim)
    family = classify_family(height=h, width=w, ndim=ndim, suffix=suffix, array=a)

    # Route by AmbrosM branches (filetype png first, then shapes)
    if suffix == ".png" or family == "png_lumify":
        est = _parse_png_lumify(a)
        if est is not None:
            est.suffix = suffix
            return est
        return _unknown(
            family="png_lumify",
            height=h,
            width=w,
            ndim=ndim,
            suffix=suffix,
            scale_note="png_parse_failed",
        )

    if (h, w) == (800, 1200):
        est = _parse_800x1200(a)
        est.suffix = suffix
        return est

    if (h, w) == (644, 1088):
        est = _parse_644x1088(a)
        est.suffix = suffix
        return est

    if h in (512, 513):
        est = _parse_h512_bottom(a)
        est.suffix = suffix
        return est

    if h == 853 and ndim == 2:
        est = _parse_gray_853(a)
        est.suffix = suffix
        return est

    return _unknown(
        family=family,
        height=h,
        width=w,
        ndim=ndim,
        suffix=suffix,
        scale_note=f"unknown_format {a.shape}",
        notes=["no_ambrosm_family_match"],
    )
