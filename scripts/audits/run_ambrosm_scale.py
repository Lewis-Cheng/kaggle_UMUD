#!/usr/bin/env python3
"""Run AmbrosM / xiaoleilian per-family scale parsers on UMUD pools.

Writes reports/phase3/ambrosm_scale_v2_report.json (+ CSV) with provenance tags.
Never uses TIFF dpi. Does not train or submit.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from _common import ROOT, utc_now_iso, write_json  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from umud.calibration.ambrosm_scale import estimate_scale  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def run_pool(pool: str, files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, path in enumerate(files):
        try:
            est = estimate_scale(path)
            rec = est.to_dict()
            rec["pool"] = pool
            rec["stem"] = path.stem
            rec["name"] = path.name
            rec["path"] = str(path)
            rec["error"] = None
        except Exception as exc:  # noqa: BLE001
            rec = {
                "pool": pool,
                "stem": path.stem,
                "name": path.name,
                "path": str(path),
                "family": "unknown",
                "provenance": "unknown",
                "px_per_cm": None,
                "px_per_mm": None,
                "mm_per_px": None,
                "tick_spacing_px": None,
                "assumed_mm_per_tick": None,
                "q_provenance": "unknown",
                "scale_note": f"load_err:{exc}",
                "error": str(exc),
                "notes": ["do_not_use_tiff_dpi"],
            }
        rows.append(rec)
        if (i + 1) % 200 == 0:
            print(f"  {pool}: {i+1}/{len(files)}", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prov = Counter(str(r.get("provenance") or "unknown") for r in rows)
    by_family: dict[str, dict[str, Any]] = {}
    fam_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        fam_groups[str(r.get("family") or "unknown")].append(r)

    for fam, items in sorted(fam_groups.items()):
        px = [float(x["px_per_mm"]) for x in items if x.get("px_per_mm")]
        mm = [float(x["mm_per_px"]) for x in items if x.get("mm_per_px")]
        qs = [float(x["assumed_mm_per_tick"]) for x in items if x.get("assumed_mm_per_tick")]
        notes = Counter(str(x.get("scale_note") or "") for x in items)
        by_family[fam] = {
            "n": len(items),
            "n_measured": sum(1 for x in items if x.get("provenance") == "measured"),
            "n_family_prior": sum(1 for x in items if x.get("provenance") == "family_prior"),
            "n_unknown": sum(1 for x in items if x.get("provenance") == "unknown"),
            "median_px_per_mm": _median(px),
            "median_mm_per_px": _median(mm),
            "q_mm_values": sorted(set(round(q, 4) for q in qs)),
            "scale_notes": dict(notes.most_common(12)),
        }

    by_pool: dict[str, dict[str, Any]] = {}
    for pool in sorted({str(r.get("pool")) for r in rows}):
        items = [r for r in rows if r.get("pool") == pool]
        by_pool[pool] = {
            "n": len(items),
            "n_measured": sum(1 for x in items if x.get("provenance") == "measured"),
            "n_family_prior": sum(1 for x in items if x.get("provenance") == "family_prior"),
            "n_unknown": sum(1 for x in items if x.get("provenance") == "unknown"),
            "coverage_frac": (
                sum(1 for x in items if x.get("provenance") in {"measured", "family_prior"})
                / len(items)
                if items
                else 0.0
            ),
        }

    return {
        "n_total": len(rows),
        "provenance_counts": dict(prov),
        "n_with_px_per_mm": sum(1 for r in rows if r.get("px_per_mm")),
        "by_pool": by_pool,
        "by_family": by_family,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "pool",
        "stem",
        "name",
        "family",
        "provenance",
        "q_provenance",
        "px_per_cm",
        "px_per_mm",
        "mm_per_px",
        "tick_spacing_px",
        "assumed_mm_per_tick",
        "fov_l",
        "fov_t",
        "fov_r",
        "fov_b",
        "fov_w_mm",
        "fov_h_mm",
        "scale_note",
        "height",
        "width",
        "ndim",
        "suffix",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="default: <repo>/reports/phase3/ambrosm_scale_v2_report.json",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="default: <repo>/reports/phase3/ambrosm_scale_v2_rows.csv",
    )
    parser.add_argument(
        "--pools",
        nargs="+",
        default=["test", "fasc", "apo"],
        choices=["test", "fasc", "apo"],
    )
    parser.add_argument("--limit", type=int, default=None, help="per-pool cap (debug)")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    layout = discover_layout(data_root)
    pool_dirs = {
        "test": layout.test_imgs,
        "fasc": layout.fasc_imgs,
        "apo": layout.apo_imgs,
    }

    all_rows: list[dict[str, Any]] = []
    for pool in args.pools:
        files = list_images(pool_dirs[pool])
        if args.limit is not None:
            files = files[: args.limit]
        print(f"pool={pool} n={len(files)} dir={pool_dirs[pool]}", flush=True)
        all_rows.extend(run_pool(pool, files))

    summary = summarize(all_rows)
    out_json = args.out_json or (ROOT / "reports" / "phase3" / "ambrosm_scale_v2_report.json")
    out_csv = args.out_csv or (ROOT / "reports" / "phase3" / "ambrosm_scale_v2_rows.csv")

    report = {
        "audit": "ambrosm_per_family_scale_v2",
        "timestamp_utc": utc_now_iso(),
        "data_root": str(data_root),
        "sources": [
            "ambrosm/umud-quick-and-dirty collect_metadata",
            "xiaoleilian/umud-hardcore-eda ambrosm_test_scale + family_of",
        ],
        "forbid_tiff_dpi_25_4": True,
        "provenance_enum": ["measured", "family_prior", "unknown"],
        "summary": summary,
        "rows": all_rows,
        "notes": [
            "measured = tick positions from this image; assumed_mm_per_tick q is still AmbrosM UI prior.",
            "family_prior = fixed AmbrosM shape/HUD formula (e.g. 630.5/5).",
            "unknown = no AmbrosM family match or parser miss.",
            "Never treat TIFF print-dpi as ultrasound spacing.",
            "v2: mirror-aware 800x1200 hflip right/left parsers for train fasc; q still AmbrosM UI prior; uncovered shapes stay unknown.",
        ],
    }
    write_json(out_json, report)
    write_csv(out_csv, all_rows)
    print(
        f"Wrote {out_json} and {out_csv} "
        f"prov={summary['provenance_counts']} "
        f"with_px={summary['n_with_px_per_mm']}/{summary['n_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
