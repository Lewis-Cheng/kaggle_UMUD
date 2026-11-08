"""Image-only features matching the rival ExtraTrees notebook (32×24)."""

from __future__ import annotations

import numpy as np

FEATURE_H = 24
FEATURE_W = 32


def resize_gray(gray: np.ndarray, h: int = FEATURE_H, w: int = FEATURE_W) -> np.ndarray:
    """Resize grayscale HxW float/uint image to (h, w)."""
    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected HxW gray, got shape {arr.shape}")
    try:
        import cv2

        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        im = Image.fromarray(
            (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
        )
        im = im.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def image_feature_vector(gray: np.ndarray) -> np.ndarray:
    """Rival-style compact features from a grayscale ultrasound frame.

    Components (concatenated):
      - flattened 32×24 resize
      - row / column mean bins
      - intensity percentiles
      - gradient orientation histogram (8 bins)
    """
    small = resize_gray(gray, FEATURE_H, FEATURE_W)
    flat = small.reshape(-1)

    row_means = small.mean(axis=1)
    col_means = small.mean(axis=0)

    percentiles = np.percentile(small, [5, 10, 25, 50, 75, 90, 95]).astype(np.float64)

    # Sobel-ish gradients via numpy finite differences
    gy, gx = np.gradient(small.astype(np.float64))
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)  # [-pi, pi]
    # Orientation hist weighted by magnitude (unsigned: fold to [0, pi))
    ang_u = np.mod(ang, np.pi)
    hist, _ = np.histogram(ang_u, bins=8, range=(0.0, np.pi), weights=mag)
    if hist.sum() > 0:
        hist = hist / hist.sum()

    feat = np.concatenate(
        [
            flat.astype(np.float64),
            row_means.astype(np.float64),
            col_means.astype(np.float64),
            percentiles,
            hist.astype(np.float64),
        ]
    )
    return feat
