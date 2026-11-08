#!/usr/bin/env python3
"""Phase 2 duplicate-aware split vs rival (id // 5) % 5 == 0.

Hashes train apo+fasc images (exact SHA-256), builds content groups, assigns
val via ``assign_duplicate_aware_split``, and reports rival leak count.

Pass criterion for OUR split: 0 shared content groups train↔val.
Does not invent competition scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from _common import ROOT, utc_now_iso, write_json  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from umud.data.hashing import (  # noqa: E402
    choose_val_content_groups,
    content_group_id_from_hash,
    parse_numeric_id,
    rival_leak_content_groups,
    sha256_file,
)
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.data.splits import (  # noqa: E402
    assert_no_val_group_in_train,
    assign_duplicate_aware_split,
)


def _rows_for_folder(folder: Path, *, task: str) -> list[dict]:
    rows: list[dict] = []
    for path in list_images(folder):
        digest = sha256_file(path)
        nid = parse_numeric_id(path.stem)
        rows.append(
            {
                "image_id": path.stem,
                "path": str(path),
                "task": task,
                "file_hash_image": digest,
                "content_group_id": content_group_id_from_hash(digest),
                "numeric_id": nid,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "phase2")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="optional cap per folder for smoke runs (0 = all)",
    )
    args = parser.parse_args()

    layout = discover_layout(args.data_root)
    apo_rows = _rows_for_folder(layout.apo_imgs, task="aponeurosis")
    fasc_rows = _rows_for_folder(layout.fasc_imgs, task="fascicle")
    if args.max_files and args.max_files > 0:
        apo_rows = apo_rows[: args.max_files]
        fasc_rows = fasc_rows[: args.max_files]

    rows = apo_rows + fasc_rows
    if not rows:
        print("ERROR: no train images found under apo_imgs / fasc_imgs", file=sys.stderr)
        return 1

    # Exact-hash content groups (cross-task: same bytes → same group).
    all_groups = [r["content_group_id"] for r in rows]
    val_groups = choose_val_content_groups(
        all_groups, val_fraction=args.val_fraction, seed=args.seed
    )
    assigned = assign_duplicate_aware_split(rows, val_content_groups=val_groups)
    assert_no_val_group_in_train(assigned)

    train_groups = {r["content_group_id"] for r in assigned if r["split"] == "train"}
    val_groups_set = {r["content_group_id"] for r in assigned if r["split"] == "val"}
    our_overlap = sorted(train_groups & val_groups_set)

    rival_leaks = rival_leak_content_groups(assigned)
    n_with_numeric = sum(1 for r in assigned if r["numeric_id"] is not None)
    n_rival_val = sum(
        1
        for r in assigned
        if r["numeric_id"] is not None and (int(r["numeric_id"]) // 5) % 5 == 0
    )

    # Hash collision / duplicate stats
    hash_to_ids: dict[str, list[str]] = {}
    for r in assigned:
        hash_to_ids.setdefault(r["file_hash_image"], []).append(
            f"{r['task']}:{r['image_id']}"
        )
    duplicate_hashes = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}

    gate_pass = len(our_overlap) == 0
    report = {
        "audit": "phase2_duplicate_split",
        "timestamp_utc": utc_now_iso(),
        "data_root": str(layout.data_root),
        "hash_method": "sha256_exact_file",
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "n_rows": len(assigned),
        "n_apo": len(apo_rows),
        "n_fasc": len(fasc_rows),
        "n_unique_content_groups": len(set(all_groups)),
        "n_val_content_groups": len(val_groups_set),
        "n_train_content_groups": len(train_groups),
        "n_train_rows": sum(1 for r in assigned if r["split"] == "train"),
        "n_val_rows": sum(1 for r in assigned if r["split"] == "val"),
        "our_split_shared_groups_train_val": our_overlap,
        "our_split_gate_pass": gate_pass,
        "pass_criterion": "0 shared content-groups train↔val",
        "n_rows_with_numeric_id": n_with_numeric,
        "n_rival_val_rows_block5": n_rival_val,
        "rival_split_rule": "(numeric_id // 5) % 5 == 0",
        "rival_leak_content_group_count": len(rival_leaks),
        "rival_leak_content_groups_sample": rival_leaks[:50],
        "n_duplicate_file_hashes": len(duplicate_hashes),
        "duplicate_hash_examples": [
            {"hash": h, "members": members[:10], "n_members": len(members)}
            for h, members in list(sorted(duplicate_hashes.items(), key=lambda kv: -len(kv[1])))[:20]
        ],
        "notes": [
            "Our split uses assign_duplicate_aware_split on exact SHA-256 content groups.",
            "Rival leak count = content groups that block-of-5 would place in BOTH train and val.",
            "Subject independence is NOT established by content-hash groups alone.",
            "No competition scores are computed or claimed here.",
        ],
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "duplicate_split.json"
    write_json(out, report)

    # Also write assignments for reuse (JSONL).
    assign_path = out_dir / "split_assignments.jsonl"
    with assign_path.open("w", encoding="utf-8") as fh:
        for row in assigned:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Wrote {out}")
    print(f"Wrote {assign_path}")
    print(
        f"gate_pass={gate_pass} our_overlap={len(our_overlap)} "
        f"rival_leaks={len(rival_leaks)} "
        f"groups={report['n_unique_content_groups']} "
        f"dup_hashes={report['n_duplicate_file_hashes']}"
    )
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
