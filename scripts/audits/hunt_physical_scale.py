#!/usr/bin/env python3
"""Hunt real ultrasound mm/px (scale bars, depth notes) — never invent scale.

Sources (in order):
  1. Text/README/CSV/JSON under --data-root (mm/px, depth, transducer, pix/cm)
  2. Optional --web-notes file (organizer discussion / EDA notebook notes)
  3. Vertical tick detection on a sample of train/test images
     (Paul Ritsche: vertical scale bars; some DL_Track images have NO bar)

Never uses TIFF print-dpi (dpi=1,1 → 25.4 is FAKE for ultrasound).
Writes reports/scale_hunt.json (or --out). Leaves provenance unknown when
physical tick pitch / depth is missing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from _common import ROOT, utc_now_iso, write_json  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from umud.baselines.mask_proxy import load_gray  # noqa: E402
from umud.baselines.scale_bar import detect_vertical_scale_bar  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402

TEXT_EXTS = {".json", ".csv", ".tsv", ".yml", ".yaml", ".txt", ".md", ".ipynb"}
SCALE_RE = re.compile(
    r"(?P<key>mm_per_px|mm/px|mm\s*/\s*pixel|pix(?:els)?\s*/\s*(?:mm|cm)|"
    r"pixel_spacing|PixelSpacing|spacing_sx|spacing_sy|image\s*depth|"
    r"depth\s*(?:cm|mm)|transducer|probe\s*freq|MHz|scale\s*bar|"
    r"pixels?\s*per\s*(?:mm|cm)|px/(?:mm|cm))",
    re.I,
)
NUM_RE = re.compile(
    r"(?P<label>mm_per_px|mm/px|pix(?:els)?/(?:mm|cm)|px/(?:mm|cm)|"
    r"pixel_spacing|depth)\s*[:=]\s*(?P<val>[-+eE0-9.]+)\s*(?P<unit>mm|cm|px)?",
    re.I,
)


def _scan_text_file(path: Path) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    try:
        if path.stat().st_size > 5_000_000:
            return hits
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return [{"path": str(path), "error": str(exc)}]
    for m in SCALE_RE.finditer(text):
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 120)
        snippet = " ".join(text[start:end].split())
        hits.append(
            {
                "path": str(path),
                "match": m.group("key"),
                "snippet": snippet[:240],
            }
        )
    for m in NUM_RE.finditer(text):
        hits.append(
            {
                "path": str(path),
                "match": m.group("label"),
                "value": m.group("val"),
                "unit": m.group("unit"),
                "kind": "numeric_assignment",
            }
        )
    return hits


def scan_data_root_text(data_root: Path, *, max_files: int = 5000) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    n = 0
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        n += 1
        if n > max_files:
            break
        hits.extend(_scan_text_file(path))
    return hits


def scan_web_notes(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return _scan_text_file(path)


def _parse_mm_per_tick(web_hits: list[dict[str, Any]], data_hits: list[dict[str, Any]]) -> float | None:
    """Best-effort mm_per_tick from notes (e.g. 10 mm between major ticks)."""
    for hit in web_hits + data_hits:
        if hit.get("kind") != "numeric_assignment":
            continue
        label = str(hit.get("match", "")).lower()
        try:
            val = float(hit["value"])
        except Exception:
            continue
        unit = (hit.get("unit") or "").lower()
        if "depth" in label and unit == "cm" and val > 0:
            # Depth alone is not mm/tick — skip conversion without pixel depth.
            continue
        if label in {"mm_per_px", "mm/px"} and val > 0:
            # Direct scale — not tick pitch.
            continue
    # Explicit tick pitch keys in snippets
    for hit in web_hits + data_hits:
        snip = str(hit.get("snippet", "")).lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:per|/)\s*tick", snip)
        if m:
            return float(m.group(1))
        m = re.search(r"tick[s]?\s*[:=]\s*(\d+(?:\.\d+)?)\s*mm", snip)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*cm\s*(?:per|/)\s*tick", snip)
        if m:
            return float(m.group(1)) * 10.0
    return None


def sample_scale_bars(
    data_root: Path,
    *,
    sample_size: int,
    seed: int,
    mm_per_tick: float | None,
) -> dict[str, Any]:
    layout = discover_layout(data_root)
    pools = {
        "fasc_imgs": list_images(layout.fasc_imgs),
        "apo_imgs": list_images(layout.apo_imgs),
        "test_imgs": list_images(layout.test_imgs),
    }
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    n_found = 0
    n_with_mm = 0
    spacings: list[float] = []
    for pool_name, files in pools.items():
        if not files:
            continue
        k = min(sample_size, len(files))
        idx = rng.choice(len(files), size=k, replace=False)
        for i in idx:
            path = files[int(i)]
            try:
                gray = load_gray(path)
            except Exception as exc:
                results.append(
                    {
                        "pool": pool_name,
                        "path": str(path),
                        "error": str(exc),
                        "found": False,
                    }
                )
                continue
            det = detect_vertical_scale_bar(gray, mm_per_tick=mm_per_tick)
            if det.found:
                n_found += 1
                if det.tick_spacing_px:
                    spacings.append(float(det.tick_spacing_px))
                if det.mm_per_px is not None:
                    n_with_mm += 1
            results.append(
                {
                    "pool": pool_name,
                    "stem": path.stem,
                    "found": det.found,
                    "side": det.side,
                    "tick_spacing_px": det.tick_spacing_px,
                    "n_ticks": det.n_ticks,
                    "mm_per_px": det.mm_per_px,
                    "provenance": det.provenance,
                    "notes": det.notes,
                }
            )
    summary = {
        "n_probed": len(results),
        "n_ticks_found": n_found,
        "n_with_mm_per_px": n_with_mm,
        "tick_spacing_px_p50": float(np.median(spacings)) if spacings else None,
        "tick_spacing_px_p10": float(np.percentile(spacings, 10)) if spacings else None,
        "tick_spacing_px_p90": float(np.percentile(spacings, 90)) if spacings else None,
        "fraction_with_ticks": (n_found / len(results)) if results else 0.0,
    }
    return {"summary": summary, "samples": results[:200]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <repo>/reports/scale_hunt.json",
    )
    parser.add_argument(
        "--web-notes",
        type=Path,
        default=None,
        help="optional organizer / EDA notes file (e.g. UMUD Hardcore EDA excerpts)",
    )
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mm-per-tick",
        type=float,
        default=None,
        help="if known from notes, physical pitch of major vertical ticks (mm)",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    out = args.out or (ROOT / "reports" / "scale_hunt.json")

    data_hits = scan_data_root_text(data_root)
    web_hits = scan_web_notes(args.web_notes)

    mm_per_tick = args.mm_per_tick
    if mm_per_tick is None:
        mm_per_tick = _parse_mm_per_tick(web_hits, data_hits)

    bar = sample_scale_bars(
        data_root,
        sample_size=args.sample_size,
        seed=args.seed,
        mm_per_tick=mm_per_tick,
    )

    # Direct numeric mm/px from text (still labelled; human must verify)
    direct_scales = [
        h
        for h in data_hits + web_hits
        if h.get("kind") == "numeric_assignment"
        and str(h.get("match", "")).lower() in {"mm_per_px", "mm/px"}
    ]

    recommended_mm_per_px = None
    provenance = "unknown"
    notes = [
        "TIFF print-dpi must NOT be used (dpi 1,1 → 25.4 is fake for ultrasound).",
        "Organizer: vertical scale bars; some train images (DL_Track) have no bar.",
        "Prefer pixel-space compare or per-image scale when bars exist.",
        "Public notebook reference: xiaoleilian 'UMUD Hardcore EDA' "
        "section 'Scale: pixels per millimetre' (ideas only; verify before use).",
    ]

    # If tick detection produced mm_per_px values, summarize median
    mm_vals = [
        s["mm_per_px"]
        for s in bar["samples"]
        if s.get("mm_per_px") is not None
    ]
    if mm_vals:
        recommended_mm_per_px = float(np.median(mm_vals))
        provenance = "scale_bar_ticks+mm_per_tick"
        notes.append(
            f"median mm_per_px from {len(mm_vals)} frames with known mm_per_tick"
        )
    elif direct_scales:
        # Record but do not auto-promote — still needs human verification
        provenance = "text_mention_unverified"
        notes.append(
            "Found numeric mm/px mentions in text; not auto-applied "
            f"(n={len(direct_scales)}). Human must verify."
        )
    elif bar["summary"]["n_ticks_found"] > 0:
        provenance = "scale_bar_ticks_detected_physical_unknown"
        notes.append(
            "Vertical ticks detected on some frames but physical tick pitch unknown; "
            "mm_per_px left null. Pass --mm-per-tick when known."
        )
    else:
        notes.append("No scale bars / numeric scale found; leave unknown.")

    report = {
        "audit": "hunt_physical_scale",
        "timestamp_utc": utc_now_iso(),
        "data_root": str(data_root),
        "web_notes": str(args.web_notes) if args.web_notes else None,
        "mm_per_tick_used": mm_per_tick,
        "calibration_provenance": provenance,
        "recommended_mm_per_px": recommended_mm_per_px,
        "forbid_tiff_dpi_25_4": True,
        "rival_isotropic_guess_only": 0.07225,
        "text_hits_n": len(data_hits),
        "text_hits_sample": data_hits[:50],
        "web_hits_n": len(web_hits),
        "web_hits_sample": web_hits[:50],
        "direct_mm_per_px_mentions": direct_scales[:20],
        "scale_bar_probe": bar["summary"],
        "scale_bar_samples": bar["samples"],
        "notes": notes,
        "status": (
            "scale_resolved"
            if recommended_mm_per_px is not None
            else "calibration_provenance_unknown"
        ),
    }
    write_json(out, report)
    print(
        f"Wrote {out} provenance={provenance} "
        f"recommended_mm_per_px={recommended_mm_per_px} "
        f"ticks_found={bar['summary']['n_ticks_found']}/{bar['summary']['n_probed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
