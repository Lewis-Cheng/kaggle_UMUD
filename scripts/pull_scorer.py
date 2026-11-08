#!/usr/bin/env python3
"""Download + hash-check the organizer scoring notebook.

Pull URL:
  https://www.kaggle.com/api/v1/kernels/pull/paulritsche/umud-score

Pinned notebook-source SHA256 (configs/protocol.yaml):
  3873f613d55be3eee82229d661dd2c06c1d3c962ff63a1ec45b0a97b29cf061f

Behavior
--------
* Always verifies hash when a notebook source file is present / produced.
* With Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME +
  KAGGLE_KEY): ``--download`` pulls the kernel and writes under
  artifacts/scorer/, then hash-checks the extracted source.
* Without credentials: dry-run documents the contract and exits 0 unless a
  local file fails the hash check.

Does not invent competition scores. Refuses to silently accept a changed notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umud.metric.score import PINNED_NOTEBOOK_SHA256, verify_scorer_hash  # noqa: E402

PULL_URL = "https://www.kaggle.com/api/v1/kernels/pull/paulritsche/umud-score"
KERNEL_SLUG = "paulritsche/umud-score"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "scorer"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pinned_hash() -> str:
    protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
    return protocol["scorer"]["notebook_source_sha256"]


def kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cand = Path.home() / ".kaggle" / "kaggle.json"
    if cand.is_file():
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
            return bool(data.get("username") and data.get("key"))
        except (OSError, json.JSONDecodeError):
            return False
    return False


def _extract_notebook_source(download_dir: Path) -> Path:
    """Locate kernel source bytes for hashing.

    Prefer an extracted .ipynb / .py; if only a zip is present, unzip first.
    The pinned hash is over the notebook *source* as retrieved in the audit
    (ipynb JSON bytes). We hash whichever primary source file we find and
    surface the path so operators can confirm.
    """
    zips = sorted(download_dir.glob("*.zip"))
    for zpath in zips:
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(download_dir)

    # Prefer .ipynb, then kernel sources commonly named after the slug.
    candidates: list[Path] = []
    candidates.extend(sorted(download_dir.rglob("*.ipynb")))
    candidates.extend(sorted(download_dir.rglob("*.py")))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for c in candidates:
        rp = c.resolve()
        if rp not in seen and c.is_file():
            seen.add(rp)
            uniq.append(c)
    if not uniq:
        raise FileNotFoundError(
            f"no .ipynb/.py notebook source under {download_dir} after pull"
        )
    return uniq[0]


def download_kernel(out_dir: Path) -> Path:
    """Authenticated pull via Kaggle API; returns path to notebook source file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "kaggle package not installed. pip install kaggle  "
            f"(needed for live pull of {PULL_URL})"
        ) from exc

    api = KaggleApi()
    api.authenticate()
    # kernels_pull writes into path; keep quiet=False for operator visibility.
    api.kernels_pull(KERNEL_SLUG, path=str(out_dir), metadata=False)
    return _extract_notebook_source(out_dir)


def check_hash(path: Path, expected: str) -> int:
    actual = sha256_file(path)
    ok = verify_scorer_hash(actual, expected)
    print(f"notebook_source: {path}")
    print(f"actual_sha256: {actual}")
    print(f"pinned_sha256: {expected}")
    print(f"match: {ok}")
    if not ok:
        print(
            "REFUSING changed scorer notebook. Update protocol.yaml only after "
            "an explicit audit; do not tune against an unverified metric.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-notebook-source",
        type=Path,
        default=None,
        help="path to already-extracted notebook source bytes to hash-check",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="live Kaggle API download (requires kaggle package + token)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"download directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    expected = load_pinned_hash()
    assert expected == PINNED_NOTEBOOK_SHA256, "protocol.yaml hash drift vs code constant"

    print(f"pull_url: {PULL_URL}")
    print(f"pinned_sha256: {expected}")
    print(f"credentials_present: {kaggle_credentials_present()}")

    if args.download:
        if not kaggle_credentials_present():
            print(
                "DRY-RUN: no Kaggle credentials "
                "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY). "
                f"Would pull {PULL_URL} into {args.out_dir} and verify SHA256. "
                "Pass --local-notebook-source to hash-check an existing file.",
                file=sys.stderr,
            )
            return 0
        try:
            source = download_kernel(args.out_dir)
        except Exception as exc:  # noqa: BLE001 — surface API errors cleanly
            print(f"download failed: {exc}", file=sys.stderr)
            return 2
        return check_hash(source, expected)

    if args.local_notebook_source is None:
        # Also auto-check if a previous download exists.
        cached = list(args.out_dir.rglob("*.ipynb")) if args.out_dir.is_dir() else []
        if cached:
            print(f"found cached notebook under {args.out_dir}; verifying")
            return check_hash(cached[0], expected)
        print(
            "No local file provided. Place extracted notebook source and re-run with "
            "--local-notebook-source, or pass --download once API creds are configured."
        )
        return 0

    if not args.local_notebook_source.is_file():
        print(f"missing file: {args.local_notebook_source}", file=sys.stderr)
        return 2
    return check_hash(args.local_notebook_source, expected)


if __name__ == "__main__":
    raise SystemExit(main())
