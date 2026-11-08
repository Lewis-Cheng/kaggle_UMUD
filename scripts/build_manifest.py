#!/usr/bin/env python3
"""Build an immutable UMUD file manifest (schema only in scaffold).

TODO: walk real aponeurosis/fascicle folders, compute file hashes, assign
content_group_id across tasks, record calibration provenance and dims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umud.data.manifest import MANIFEST_FIELDS, build_manifest_row  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "manifest.jsonl",
        help="output JSONL path (under data/, gitignored content ok)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print schema fields and exit without writing",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("MANIFEST_FIELDS:")
        for f in MANIFEST_FIELDS:
            print(f"  - {f}")
        # Smoke-check builder with a toy row (not real competition data).
        row = build_manifest_row(
            {
                "image_id": "example",
                "image_path": "data/example.png",
                "task": "fascicle",
                "file_hash_image": "0" * 64,
                "content_group_id": "cg_example",
                "annotation_available_fasc": True,
            }
        )
        print("example_row:", json.dumps(row.to_dict(), sort_keys=True))
        return 0

    raise SystemExit(
        "TODO: point --root at audited UMUD folders and implement hashing/grouping. "
        "Use --dry-run to inspect the schema."
    )


if __name__ == "__main__":
    raise SystemExit(main())
