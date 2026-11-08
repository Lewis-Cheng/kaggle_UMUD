#!/usr/bin/env python3
"""Phase 2 scale sensitivity audit (proxy geometry — NOT Tier B).

Recomputes rival-style FL/MT pixel proxies at isotropic scales
{0.05, 0.07225, 0.10} mm/px and any discovered per-image sx/sy.

Outputs are labelled ``proxy`` / ``calibration_provenance``. Never invents
competition scores.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from _common import ROOT, utc_now_iso, write_json  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.geometry.physical import physical_length  # noqa: E402

ISOTROPIC_SCALES = (0.05, 0.07225, 0.10)
SCALE_KEY_RE = re.compile(
    r"(?P<key>mm_per_px|mm/px|spacing_sx|spacing_sy|sx|sy|pixel_spacing|"
    r"PixelSpacing|x_resolution|y_resolution)\s*[:=]\s*(?P<val>[-+eE0-9.]+)",
    re.I,
)


def _load_mask(path: Path) -> np.ndarray:
    """Load mask as boolean HxW (foreground = nonzero)."""
    try:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise OSError("cv2.imread returned None")
    except Exception:
        from PIL import Image

        arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.asarray(arr) > 0


def proxy_fascicle_pca(mask: np.ndarray) -> dict[str, float] | None:
    """Rival-like fascicle proxy: PCA extent along principal axis (pixels).

    Returns pixel length and angle (deg, image coords) or None if empty.
    Label: PROXY — not expert FL / PA.
    """
    ys, xs = np.nonzero(mask)
    if ys.size < 5:
        return None
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    centered = pts - mean
    # Covariance PCA
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0]
    proj = centered @ axis
    length_px = float(proj.max() - proj.min())
    angle_deg = float(math.degrees(math.atan2(axis[1], axis[0])))
    # Acute-ish pennation vs horizontal (proxy only)
    pa_proxy = abs(angle_deg)
    pa_proxy = min(pa_proxy, 180.0 - pa_proxy)
    return {
        "fl_length_px": length_px,
        "pa_deg_proxy": pa_proxy,
        "eigval0": float(eigvals[0]),
        "n_fg": int(ys.size),
    }


def proxy_apo_peak_gap(mask: np.ndarray) -> dict[str, float] | None:
    """Rival-like MT proxy: gap between two peaks of the row-sum profile (px).

    Falls back to foreground row min/max span when peaks are unclear.
    Label: PROXY — not expert MT.
    """
    if mask.size == 0 or not mask.any():
        return None
    # Row profile: sum of foreground along columns → per-row thickness mass.
    profile = mask.sum(axis=1).astype(np.float64)
    if float(profile.max()) <= 0:
        return None
    # Smooth lightly with a box filter.
    k = 5
    kernel = np.ones(k, dtype=np.float64) / k
    smooth = np.convolve(profile, kernel, mode="same")
    # Find local maxima
    peaks = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > 0.2 * smooth.max():
            peaks.append(i)
    gap_px = None
    method = "row_minmax"
    if len(peaks) >= 2:
        # Take the two strongest peaks.
        peak_scores = sorted(((smooth[i], i) for i in peaks), reverse=True)
        i1, i2 = sorted([peak_scores[0][1], peak_scores[1][1]])
        gap_px = float(i2 - i1)
        method = "two_peak_gap"
    ys, _xs = np.nonzero(mask)
    span = float(ys.max() - ys.min()) if ys.size else 0.0
    if gap_px is None or gap_px <= 0:
        gap_px = span
        method = "row_minmax"
    return {"mt_gap_px": float(gap_px), "mt_method": method, "row_span_px": span}


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _probe_spacing_from_image(path: Path) -> dict[str, Any] | None:
    """Best-effort spacing from TIFF/PNG metadata. Returns mm/px guesses or None."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            info = dict(im.info)
            dpi = info.get("dpi")
            if dpi and len(dpi) == 2 and dpi[0] and dpi[1]:
                # inch → mm: 25.4 / dpi
                sx = 25.4 / float(dpi[0])
                sy = 25.4 / float(dpi[1])
                if sx > 0 and sy > 0:
                    return {
                        "sx": sx,
                        "sy": sy,
                        "source": "image_dpi",
                        "path": str(path),
                        "note": "dpi→mm/px conversion; ultrasound may not use print dpi",
                    }
            tag_v2 = getattr(im, "tag_v2", None)
            if tag_v2:
                # XResolution=282, YResolution=283, ResolutionUnit=296
                xres = tag_v2.get(282)
                yres = tag_v2.get(283)
                unit = tag_v2.get(296)
                if xres and yres:
                    def _rat(v):
                        if hasattr(v, "numerator"):
                            return float(v.numerator) / float(v.denominator or 1)
                        if isinstance(v, tuple) and len(v) == 2:
                            return float(v[0]) / float(v[1] or 1)
                        return float(v)

                    xr, yr = _rat(xres), _rat(yres)
                    if xr > 0 and yr > 0:
                        # unit 2 = inch, 3 = cm
                        if unit == 3:
                            sx, sy = 10.0 / xr, 10.0 / yr  # cm→mm
                        else:
                            sx, sy = 25.4 / xr, 25.4 / yr
                        return {
                            "sx": sx,
                            "sy": sy,
                            "source": "tiff_resolution_tags",
                            "resolution_unit": unit,
                            "path": str(path),
                        }
    except Exception:
        return None
    return None


def _scan_text_spacings(data_root: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".csv", ".tsv", ".yml", ".yaml", ".txt", ".md"}:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in SCALE_KEY_RE.finditer(text):
            try:
                val = float(m.group("val"))
            except ValueError:
                continue
            found.append(
                {
                    "path": str(path.relative_to(data_root)),
                    "key": m.group("key"),
                    "value": val,
                }
            )
            if len(found) >= 50:
                return found
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "phase2")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="max paired fasc+apo mask stems to probe (default 200; use all if fewer)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    layout = discover_layout(args.data_root)
    fasc_masks = {p.stem: p for p in list_images(layout.fasc_masks)}
    apo_masks = {p.stem: p for p in list_images(layout.apo_masks)}
    # Paired by identical stem when present; else independent stratified samples.
    paired_stems = sorted(set(fasc_masks) & set(apo_masks))
    fasc_only = sorted(set(fasc_masks) - set(apo_masks))
    apo_only = sorted(set(apo_masks) - set(fasc_masks))

    rng = random.Random(args.seed)
    if paired_stems:
        picked = paired_stems[:]
        rng.shuffle(picked)
        picked = picked[: args.sample_size]
        iter_items = [(s, s) for s in picked]
        pair_mode = "identical_stem"
    else:
        # No stem overlap: sample each side independently (still a sensitivity probe).
        fs = fasc_only[:]
        as_ = apo_only[:]
        rng.shuffle(fs)
        rng.shuffle(as_)
        n = min(args.sample_size, len(fs), len(as_))
        iter_items = list(zip(fs[:n], as_[:n]))
        pair_mode = "independent_sample_no_stem_overlap"

    text_spacings = _scan_text_spacings(layout.data_root)

    # Probe a few images for dpi/tiff resolution.
    image_spacings: list[dict[str, Any]] = []
    for folder in (layout.fasc_imgs, layout.apo_imgs, layout.test_imgs):
        imgs = list_images(folder)
        for p in imgs[:10]:
            hit = _probe_spacing_from_image(p)
            if hit:
                image_spacings.append(hit)

    discovered_isotropic: list[float] = []
    discovered_aniso: list[dict[str, float]] = []
    for hit in image_spacings:
        if abs(hit["sx"] - hit["sy"]) < 1e-9:
            discovered_isotropic.append(float(hit["sx"]))
        else:
            discovered_aniso.append({"sx": float(hit["sx"]), "sy": float(hit["sy"])})
    for hit in text_spacings:
        key = hit["key"].lower()
        if "sx" in key or "sy" in key:
            continue  # need pairs; record raw only
        if 1e-4 < hit["value"] < 1.0:  # plausible mm/px band
            discovered_isotropic.append(float(hit["value"]))

    # Deduplicate scales for the sweep.
    scale_list = list(ISOTROPIC_SCALES)
    for s in discovered_isotropic:
        if all(abs(s - x) > 1e-12 for x in scale_list):
            scale_list.append(s)

    if image_spacings or any(1e-4 < h["value"] < 1.0 for h in text_spacings):
        calibration_provenance = "known"
        recommended_scale: float | dict | None
        if discovered_aniso:
            recommended_scale = discovered_aniso[0]
        elif discovered_isotropic:
            recommended_scale = float(np.median(discovered_isotropic))
        else:
            # Mentions exist but values not parsed into mm/px confidently.
            calibration_provenance = "unknown"
            recommended_scale = None
    else:
        calibration_provenance = "unknown"
        recommended_scale = None

    # Accumulate proxy measurements in pixels first, then scale.
    fl_px_list: list[float] = []
    mt_px_list: list[float] = []
    pa_list: list[float] = []
    n_fail_fasc = 0
    n_fail_apo = 0
    n_used_fasc = 0
    n_used_apo = 0

    for fasc_stem, apo_stem in iter_items:
        fp = fasc_masks.get(fasc_stem)
        ap = apo_masks.get(apo_stem)
        if fp is not None:
            try:
                m = _load_mask(fp)
                prox = proxy_fascicle_pca(m)
                if prox is None:
                    n_fail_fasc += 1
                else:
                    fl_px_list.append(prox["fl_length_px"])
                    pa_list.append(prox["pa_deg_proxy"])
                    n_used_fasc += 1
            except Exception:
                n_fail_fasc += 1
        if ap is not None:
            try:
                m = _load_mask(ap)
                prox = proxy_apo_peak_gap(m)
                if prox is None:
                    n_fail_apo += 1
                else:
                    mt_px_list.append(prox["mt_gap_px"])
                    n_used_apo += 1
            except Exception:
                n_fail_apo += 1

    per_scale: dict[str, Any] = {}
    for s in scale_list:
        fl_mm = [physical_length((L, 0.0), sx=s, sy=s) for L in fl_px_list]
        # vertical gap: displacement (0, gap_px)
        mt_mm = [physical_length((0.0, G), sx=s, sy=s) for G in mt_px_list]
        per_scale[f"{s:.8g}"] = {
            "mm_per_px_isotropic": s,
            "proxy_fl_mm": _summarize(fl_mm),
            "proxy_mt_mm": _summarize(mt_mm),
            "label": "PROXY_NOT_TIER_B",
        }

    # Delta vs rival default 0.07225
    ref = 0.07225
    deltas: dict[str, Any] = {}
    if fl_px_list and mt_px_list:
        ref_fl = [physical_length((L, 0.0), sx=ref, sy=ref) for L in fl_px_list]
        ref_mt = [physical_length((0.0, G), sx=ref, sy=ref) for G in mt_px_list]
        for s in scale_list:
            if abs(s - ref) < 1e-15:
                continue
            fl = [physical_length((L, 0.0), sx=s, sy=s) for L in fl_px_list]
            mt = [physical_length((0.0, G), sx=s, sy=s) for G in mt_px_list]
            d_fl = [a - b for a, b in zip(fl, ref_fl)]
            d_mt = [a - b for a, b in zip(mt, ref_mt)]
            deltas[f"{s:.8g}_minus_{ref}"] = {
                "delta_proxy_fl_mm": _summarize(d_fl),
                "delta_proxy_mt_mm": _summarize(d_mt),
                "note": "signed proxy delta vs rival isotropic 0.07225; NOT leaderboard",
            }

    aniso_probe = None
    if discovered_aniso:
        sx, sy = discovered_aniso[0]["sx"], discovered_aniso[0]["sy"]
        fl_mm = [physical_length((L, 0.0), sx=sx, sy=sy) for L in fl_px_list]
        mt_mm = [physical_length((0.0, G), sx=sx, sy=sy) for G in mt_px_list]
        aniso_probe = {
            "sx": sx,
            "sy": sy,
            "proxy_fl_mm": _summarize(fl_mm),
            "proxy_mt_mm": _summarize(mt_mm),
            "label": "PROXY_NOT_TIER_B",
        }

    report = {
        "audit": "phase2_scale_audit",
        "timestamp_utc": utc_now_iso(),
        "data_root": str(layout.data_root),
        "label": "PROXY_SENSITIVITY_NOT_TIER_B",
        "calibration_provenance": calibration_provenance,
        "recommended_scale": recommended_scale,
        "isotropic_scales_tried": scale_list,
        "pair_mode": pair_mode,
        "n_paired_stems_available": len(paired_stems),
        "n_sample_requested": args.sample_size,
        "n_fasc_proxy_used": n_used_fasc,
        "n_apo_proxy_used": n_used_apo,
        "n_fasc_proxy_failed": n_fail_fasc,
        "n_apo_proxy_failed": n_fail_apo,
        "proxy_pa_deg": _summarize(pa_list),
        "proxy_fl_length_px": _summarize(fl_px_list),
        "proxy_mt_gap_px": _summarize(mt_px_list),
        "per_scale_mm": per_scale,
        "delta_vs_rival_0_07225": deltas,
        "anisotropic_discovered_probe": aniso_probe,
        "discovered_image_spacings_sample": image_spacings[:20],
        "discovered_text_spacings_sample": text_spacings[:20],
        "notes": [
            "FL/MT here are mask-derived proxies (PCA length / peak-gap), not expert scalars.",
            "Do not promote models on this proxy; Tier B needs independent PA/FL/MT + pinned scorer.",
            "calibration_provenance=known only when metadata/dpi/text yields a usable spacing.",
        ],
    }

    out = args.out_dir / "scale_audit.json"
    write_json(out, report)
    print(f"Wrote {out}")
    print(
        f"calibration_provenance={calibration_provenance} "
        f"recommended_scale={recommended_scale} "
        f"fasc_used={n_used_fasc} apo_used={n_used_apo}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
