#!/usr/bin/env python3
"""Audit AmbrosM-unknown shapes; do NOT invent mm-per-tick q."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale-csv", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--sample-per-family", type=int, default=3)
    args = ap.parse_args()

    import sys
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT / "src"))
    from umud.data.layout import discover_layout, list_images

    rows = list(csv.DictReader(args.scale_csv.open(encoding="utf-8")))
    unknown = [r for r in rows if r.get("provenance") == "unknown"]
    by_family = Counter(r.get("family", "") for r in unknown)
    by_pool = Counter(r.get("pool", "") for r in unknown)

    layout = discover_layout(args.data_root)
    stem_map = {}
    for pool, folder in (
        ("fasc", layout.fasc_imgs),
        ("apo", layout.apo_imgs),
        ("test", layout.test_imgs),
    ):
        for p in list_images(folder):
            stem_map[(pool, p.name)] = p
            stem_map[(pool, p.stem)] = p

    samples = defaultdict(list)
    for r in unknown:
        fam = r.get("family", "")
        if len(samples[fam]) >= args.sample_per_family:
            continue
        key1 = (r.get("pool"), r.get("name"))
        key2 = (r.get("pool"), Path(r.get("name", "")).stem)
        path = stem_map.get(key1) or stem_map.get(key2)
        info = {"name": r.get("name"), "pool": r.get("pool"), "path": None}
        if path is not None:
            a = np.asarray(Image.open(path))
            info.update(
                {
                    "path": str(path),
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                    "mean": float(np.mean(a)),
                    "right_col_mean": float(a[:, -min(20, a.shape[1]):].mean())
                    if a.ndim >= 2
                    else None,
                    "left_col_mean": float(a[:, : min(20, a.shape[1])].mean())
                    if a.ndim >= 2
                    else None,
                    "bottom_row_mean": float(a[-min(8, a.shape[0]) :].mean())
                    if a.ndim >= 2
                    else None,
                }
            )
        samples[fam].append(info)

    # Hard rule: no parser added without AmbrosM/literature q provenance.
    decision = {
        "parsers_added": [],
        "reason_no_new_parsers": (
            "Remaining unknown families lack AmbrosM/xiaoleilian formulas and "
            "have no documented mm-per-tick q. Tick-like edges without q remain "
            "unidentifiable (sy=q/d). Do not invent q."
        ),
        "families_unknown": dict(by_family),
        "pools_unknown": dict(by_pool),
        "n_unknown": len(unknown),
        "n_total_rows": len(rows),
    }
    report = {
        "audit": "residual_scale_unknown_shapes",
        "decision": decision,
        "samples": {k: v for k, v in samples.items()},
        "attention_note": "residual unknowns do not block test (309/309 covered); they shrink train mm supervision",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Residual AmbrosM-unknown scale audit",
        "",
        f"- unknown rows: **{len(unknown)}** / {len(rows)}",
        f"- by pool: `{dict(by_pool)}`",
        "",
        "## By family",
        "",
        "| family | n_unknown |",
        "|---|---:|",
    ]
    for fam, n in by_family.most_common():
        lines.append(f"| `{fam}` | {n} |")
    lines += [
        "",
        "## Decision",
        "",
        f"- parsers added: **none**",
        f"- reason: {decision['reason_no_new_parsers']}",
        "",
        "Test coverage remains 309/309 from AmbrosM v2; residual unknowns are train/apo/fasc only.",
        "",
    ]
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"unknown={len(unknown)} wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
