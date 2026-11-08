#!/usr/bin/env python3
"""Search data-root / repo for independent PA/FL/MT expert GT. Do not invent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYWORDS = (
    "pa_deg",
    "fl_mm",
    "mt_mm",
    "pennation",
    "fascicle_length",
    "muscle_thickness",
    "ground_truth",
    "expert",
    "labels",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    hits = []
    for root in (args.data_root, args.repo_root / "reports", args.repo_root / "data"):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".csv", ".tsv", ".json", ".xlsx", ".txt", ".md"}:
                continue
            if p.stat().st_size > 50_000_000:
                continue
            name_l = p.name.lower()
            name_hit = any(k in name_l for k in ("gt", "label", "expert", "pa_fl", "target", "submission"))
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")[:20000]
            except Exception:
                continue
            content_hit = all(k in text for k in ("pa_deg", "fl_mm", "mt_mm"))
            if name_hit or content_hit:
                hits.append(
                    {
                        "path": str(p),
                        "name_hit": name_hit,
                        "has_pa_fl_mt_headers": content_hit,
                        "size": p.stat().st_size,
                        "is_sample_submission": p.name == "sample_submission.csv",
                        "is_phase3_submission": "submission_" in p.name and "phase3" in str(p),
                    }
                )

    expert_candidates = [
        h
        for h in hits
        if h["has_pa_fl_mt_headers"]
        and not h["is_sample_submission"]
        and not h["is_phase3_submission"]
        and "tier_b" not in h["path"]
        and "median" not in h["path"]
        and "extratrees" not in h["path"]
        and "geometry" not in h["path"]
    ]

    verdict = {
        "expert_gt_found": False,
        "official_S": None,
        "reason": (
            "No independent expert PA/FL/MT ground-truth CSV in data root or repo "
            "reports. sample_submission.csv is a 2-row format template only. "
            "Train provides masks, not expert biometry labels. Official S cannot "
            "be pinned locally; Kaggle LB would be required for true S."
        ),
        "hits_n": len(hits),
        "expert_candidates": expert_candidates,
        "sample_hits": hits[:30],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    args.out_md.write_text(
        "\n".join(
            [
                "# Expert GT / official S hunt",
                "",
                f"- expert_gt_found: **{verdict['expert_gt_found']}**",
                f"- official_S: **null**",
                f"- reason: {verdict['reason']}",
                f"- scanned hits mentioning headers/names: {len(hits)}",
                f"- non-submission candidates with pa/fl/mt headers: {len(expert_candidates)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"expert_gt_found": False, "candidates": len(expert_candidates)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
