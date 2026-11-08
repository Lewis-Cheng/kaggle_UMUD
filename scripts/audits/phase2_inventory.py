#!/usr/bin/env python3
"""Phase 2 inventory audit: folder counts, stem overlap, sidecars, submission.

Does NOT claim competition scores. Writes inventory.json under --out-dir.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from _common import ROOT, utc_now_iso, write_json  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from umud.data.layout import (  # noqa: E402
    IMAGE_EXTS,
    discover_layout,
    list_images,
    stem_set,
)

SCALE_KEY_RE = re.compile(
    r"(mm_per_px|mm/px|spacing|sx|sy|pixel_spacing|pixelspacing|"
    r"x_resolution|y_resolution|calibration)",
    re.I,
)


def _count_by_ext(folder: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in list_images(folder):
        ext = p.suffix.lower()
        counts[ext] = counts.get(ext, 0) + 1
    return counts


def _overlap_report(img_dir: Path, mask_dir: Path) -> dict:
    imgs = stem_set(img_dir)
    masks = stem_set(mask_dir)
    both = imgs & masks
    return {
        "n_images": len(imgs),
        "n_masks": len(masks),
        "n_stem_overlap": len(both),
        "n_image_only": len(imgs - masks),
        "n_mask_only": len(masks - imgs),
        "image_only_sample": sorted(imgs - masks)[:20],
        "mask_only_sample": sorted(masks - imgs)[:20],
    }


def _sniff_sample_submission(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    # Prefer semicolon if header contains it (live sample uses ';').
    if lines and ";" in lines[0] and lines[0].count(";") >= lines[0].count(","):
        delim = ";"
    else:
        delim = ","
    reader = csv.reader(lines, delimiter=delim)
    rows = list(reader)
    header = rows[0] if rows else []
    return {
        "path": str(path),
        "delimiter": delim,
        "n_lines_including_header": len(rows),
        "n_data_rows": max(0, len(rows) - 1),
        "header": header,
        "first_data_row": rows[1] if len(rows) > 1 else None,
    }


def _scan_sidecars(data_root: Path) -> dict:
    """Search for DICOM/JSON/CSV/README that might carry sx/sy / mm_per_px."""
    interesting_suffixes = {".json", ".csv", ".tsv", ".yml", ".yaml", ".txt", ".md", ".dcm"}
    hits: list[dict] = []
    scale_mentions: list[dict] = []
    # Cap walk depth/files for safety on huge trees.
    n_seen = 0
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        n_seen += 1
        if n_seen > 200_000:
            break
        # Skip bulk image payloads for content scan; still note .dcm presence.
        suf = path.suffix.lower()
        if suf in IMAGE_EXTS and suf != ".tif" and suf != ".tiff":
            continue
        rel = str(path.relative_to(data_root))
        if suf in interesting_suffixes or suf in {".tif", ".tiff"}:
            entry = {"path": rel, "suffix": suf, "bytes": path.stat().st_size}
            if suf in interesting_suffixes:
                hits.append(entry)
            # Lightweight text scan for scale keys (skip large binaries).
            if suf in {".json", ".csv", ".tsv", ".yml", ".yaml", ".txt", ".md"} and path.stat().st_size < 2_000_000:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if SCALE_KEY_RE.search(text):
                    scale_mentions.append({"path": rel, "suffix": suf})
            if suf in {".tif", ".tiff"}:
                # Only probe a few TIFFs for tags (inventory; scale audit digs deeper).
                if len([h for h in hits if h.get("tiff_tags_sampled")]) >= 5:
                    continue
                try:
                    from PIL import Image

                    with Image.open(path) as im:
                        tags = {}
                        if hasattr(im, "tag_v2") and im.tag_v2:
                            # Keep only resolution-ish tags.
                            for k, v in list(im.tag_v2.items())[:30]:
                                tags[str(k)] = str(v)[:120]
                        info = {str(k): str(v)[:120] for k, v in dict(im.info).items()}
                        if tags or info:
                            entry = {
                                "path": rel,
                                "suffix": suf,
                                "tiff_tags_sampled": True,
                                "tag_v2_sample": tags,
                                "info_sample": info,
                            }
                            hits.append(entry)
                            blob = json_dump_ish(tags) + json_dump_ish(info)
                            if SCALE_KEY_RE.search(blob) or "dpi" in blob.lower() or "resolution" in blob.lower():
                                scale_mentions.append({"path": rel, "source": "tiff_tags"})
                except Exception as exc:  # noqa: BLE001 — inventory must not abort
                    hits.append({"path": rel, "suffix": suf, "tiff_open_error": str(exc)})
    return {
        "sidecar_candidates": hits[:200],
        "n_sidecar_candidates": len(hits),
        "scale_key_mentions": scale_mentions[:100],
        "n_scale_key_mentions": len(scale_mentions),
    }


def json_dump_ish(obj: object) -> str:
    return str(obj)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="UMUD folder containing apo_* / fasc_* / test_images_v2 / sample_submission.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "phase2",
        help="directory for inventory.json",
    )
    args = parser.parse_args()

    layout = discover_layout(args.data_root)
    apo = _overlap_report(layout.apo_imgs, layout.apo_masks)
    fasc = _overlap_report(layout.fasc_imgs, layout.fasc_masks)
    test_imgs = list_images(layout.test_imgs)

    sample = None
    if layout.sample_submission is not None:
        sample = _sniff_sample_submission(layout.sample_submission)

    sidecars = _scan_sidecars(layout.data_root)

    report = {
        "audit": "phase2_inventory",
        "timestamp_utc": utc_now_iso(),
        "data_root": str(layout.data_root),
        "resolved_leaf_dirs": {
            "apo_imgs": str(layout.apo_imgs),
            "apo_masks": str(layout.apo_masks),
            "fasc_imgs": str(layout.fasc_imgs),
            "fasc_masks": str(layout.fasc_masks),
            "test_imgs": str(layout.test_imgs),
        },
        "counts": {
            "apo_imgs": len(list_images(layout.apo_imgs)),
            "apo_masks": len(list_images(layout.apo_masks)),
            "fasc_imgs": len(list_images(layout.fasc_imgs)),
            "fasc_masks": len(list_images(layout.fasc_masks)),
            "test_imgs": len(test_imgs),
        },
        "extensions": {
            "apo_imgs": _count_by_ext(layout.apo_imgs),
            "apo_masks": _count_by_ext(layout.apo_masks),
            "fasc_imgs": _count_by_ext(layout.fasc_imgs),
            "fasc_masks": _count_by_ext(layout.fasc_masks),
            "test_imgs": _count_by_ext(layout.test_imgs),
        },
        "stem_overlap": {"aponeurosis": apo, "fascicle": fasc},
        "sample_submission": sample,
        "sidecars": sidecars,
        "notes": [
            "Stem overlap compares filename stems only (extension-agnostic).",
            "Sidecar scan is best-effort; TIFF tag sampling is capped.",
            "This report does not invent or claim competition scores.",
        ],
    }

    out = args.out_dir / "inventory.json"
    write_json(out, report)
    print(f"Wrote {out}")
    print(
        "counts:",
        report["counts"],
        "| apo overlap:",
        apo["n_stem_overlap"],
        "| fasc overlap:",
        fasc["n_stem_overlap"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
