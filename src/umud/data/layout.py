"""Discover UMUD competition folder layout under a data-root.

Expected (SSH live layout, Nov 2026):

    <data-root>/
      apo_imgs_v1/apo_images_new_model_v1/
      apo_masks_v1/apo_masks_new_model_v1/
      fasc_imgs_v1/fasc_images_new_model_v1/
      fasc_masks_v1/fasc_masks_new_model_v1/
      test_images_v2/test_set_v2/
      sample_submission.csv

Top-level names are stable; one nested image folder is common. Flat files
directly under the top-level dir are also accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

TOP_LEVEL = {
    "apo_imgs": "apo_imgs_v1",
    "apo_masks": "apo_masks_v1",
    "fasc_imgs": "fasc_imgs_v1",
    "fasc_masks": "fasc_masks_v1",
    "test_imgs": "test_images_v2",
}


@dataclass(frozen=True)
class UmudLayout:
    data_root: Path
    apo_imgs: Path
    apo_masks: Path
    fasc_imgs: Path
    fasc_masks: Path
    test_imgs: Path
    sample_submission: Path | None


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def list_images(folder: Path) -> list[Path]:
    """Sorted image files directly in folder (non-recursive)."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if is_image_file(p))


def resolve_leaf_dir(top: Path) -> Path:
    """If ``top`` has images, use it; else prefer a single child dir with images.

    Raises FileNotFoundError when ``top`` is missing. Returns ``top`` when no
    nested image dir is found (caller then sees zero images).
    """
    if not top.is_dir():
        raise FileNotFoundError(f"missing expected directory: {top}")
    if list_images(top):
        return top
    child_dirs = sorted(p for p in top.iterdir() if p.is_dir() and not p.name.startswith("."))
    with_images = [c for c in child_dirs if list_images(c)]
    if len(with_images) == 1:
        return with_images[0]
    if len(with_images) > 1:
        # Prefer names containing 'image' / 'mask' / 'test' / 'set'
        ranked = sorted(
            with_images,
            key=lambda p: (
                0
                if any(k in p.name.lower() for k in ("image", "mask", "test", "set", "model"))
                else 1,
                p.name,
            ),
        )
        return ranked[0]
    return top


def discover_layout(data_root: Path | str) -> UmudLayout:
    """Resolve leaf image/mask folders under data-root."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data-root does not exist: {root}")

    apo_imgs = resolve_leaf_dir(root / TOP_LEVEL["apo_imgs"])
    apo_masks = resolve_leaf_dir(root / TOP_LEVEL["apo_masks"])
    fasc_imgs = resolve_leaf_dir(root / TOP_LEVEL["fasc_imgs"])
    fasc_masks = resolve_leaf_dir(root / TOP_LEVEL["fasc_masks"])
    test_imgs = resolve_leaf_dir(root / TOP_LEVEL["test_imgs"])

    sample = root / "sample_submission.csv"
    return UmudLayout(
        data_root=root,
        apo_imgs=apo_imgs,
        apo_masks=apo_masks,
        fasc_imgs=fasc_imgs,
        fasc_masks=fasc_masks,
        test_imgs=test_imgs,
        sample_submission=sample if sample.is_file() else None,
    )


def stem_set(folder: Path) -> set[str]:
    return {p.stem for p in list_images(folder)}
