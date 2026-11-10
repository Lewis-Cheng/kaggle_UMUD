#!/usr/bin/env python3
"""Freeze the texture_rl / joint_foldpure baseline manifest.

Records hashes of the files that exist in this workspace and explicit absences
for checkpoints, prediction CSVs, and images. Does not invent per-image
measurements. Does not submit to Kaggle.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports/phase3/chase018/baseline_freeze"
MANIFEST = OUT_DIR / "manifest.json"
STAGES = OUT_DIR / "per_image_stages.jsonl"

HASH_FILES = (
    "src/umud/metric/score.py",
    "src/umud/geometry/host_faithful.py",
    "src/umud/geometry/physical.py",
    "src/umud/verification/contracts.py",
    "src/umud/seg/frundles_smp.py",
    "scripts/chase018/infer_rl_refiner.py",
    "configs/protocol.yaml",
    "reports/phase3/official_test_file_names.txt",
    "reports/phase3/chase018/submission_texture_rl.json",
    "reports/phase3/chase018/submission_joint_foldpure.json",
    "reports/phase3/chase018/multexp_chase_affine.json",
    "reports/phase3/chase018/fv_umud_contracts.json",
)

MISSING = (
    "data/external/frundles_weights/apo_resnet34_384x576_f0.pt",
    "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt",
    "reports/phase3/chase018/submission_texture_rl.csv",
    "reports/phase3/chase018/submission_texture_pa.csv",
    "reports/phase3/chase018/submission_joint_foldpure.csv",
    "reports/phase3/chase018/rl_refiner/chase018_bandit.npz",
    "data/raw/umud",
)

TEXTURE_RL = {
    "name": "texture_rl",
    "kaggle_ref": 58005465,
    "recorded_public_score": 0.29895,
    "tex_w": 0.5,
    "fl_sin_w": 0.25,
    "fl_factor": 0.9,
    "calibration": "PA+MT affine from multexp_chase_affine.json; FL left raw in infer_rl_refiner",
    "policy_action": "cal_pa_mt on 309/309 recorded in submission_texture_rl.json",
}

JOINT = {
    "name": "joint_foldpure",
    "kaggle_ref": 58058659,
    "recorded_public_score": 0.44238,
    "role": "negative control",
    "recorded_honest_nested_S": 0.1344865957876554,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{exc}"
    return out.strip()


def image_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.endswith(".csv"):
            continue
        ids.append(Path(name).name)
    return ids


def main() -> int:
    present: dict[str, str] = {}
    absent_expected: list[str] = []
    for rel in HASH_FILES:
        path = ROOT / rel
        if path.is_file():
            present[rel] = sha256_file(path)
        else:
            absent_expected.append(rel)
    missing = [rel for rel in MISSING if not (ROOT / rel).exists()]
    ids_path = ROOT / "reports/phase3/official_test_file_names.txt"
    ids = image_ids(ids_path) if ids_path.is_file() else []
    config_blob = json.dumps(
        {"texture_rl": TEXTURE_RL, "joint_foldpure": JOINT},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    config_sha = hashlib.sha256(config_blob).hexdigest()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with STAGES.open("w", encoding="utf-8") as handle:
        for image_id in ids:
            row = {
                "image_id": image_id,
                "recipes": ["texture_rl", "joint_foldpure"],
                "scale_provenance": None,
                "coordinate_transform": None,
                "apo_geometry": None,
                "fasc_orientation": None,
                "pa_deg": None,
                "fl_mm": None,
                "mt_mm": None,
                "calibration_branch": None,
                "fallback": None,
                "reason": "artifacts_absent",
            }
            handle.write(json.dumps(row) + "\n")

    manifest = {
        "git_commit": git_head(),
        "git_commit_meaning": "HEAD when this manifest was generated. The commit that adds the manifest is a child of this hash.",
        "replay_pass": False,
        "replay_reason": (
            "Checkpoint weights, prediction CSVs, bandit npz, and competition "
            "images are not in this workspace, so the 309-row CSV cannot be regenerated here."
        ),
        "scorer_sha256_pinned": "3873f613d55be3eee82229d661dd2c06c1d3c962ff63a1ec45b0a97b29cf061f",
        "config_sha256": config_sha,
        "config": {"texture_rl": TEXTURE_RL, "joint_foldpure": JOINT},
        "file_sha256": present,
        "missing_inputs": missing,
        "hash_files_absent": absent_expected,
        "n_test_ids": len(ids),
        "per_image_stages": str(STAGES.relative_to(ROOT)),
        "per_image_measurement_status": "null_artifacts_absent",
        "stopped": [
            "sin_w x factor x PA-head grids on the same maps",
            "Dice-selected fine-tunes",
            "changes aimed at MultExp S < 0.12",
            "Kaggle submit without one written hypothesis",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST), "n_test_ids": len(ids), "replay_pass": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
