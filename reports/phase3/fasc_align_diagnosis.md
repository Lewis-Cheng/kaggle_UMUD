# Fasc image/mask alignment diagnosis

**Date (Asia/Hong_Kong):** 2026-11-04

## Failure modes (why Dice stuck ~0.26)

| Finding | Evidence |
|---|---|
| Extreme imbalance | fg≈0.25%, thickness≈2px (`diag_fasc_seg.json`) |
| **Size mismatch** | Val: **216 same / 350 mismatch**; top pair img 1200×800 vs mask 996×556 |
| AmbrosM FOV ≠ mask | FOV 502×708 ≠ 996×556 — FOV crop **not** the align key for fasc |
| FascUNet / pos_w≥80 | Collapsed to Dice ~0.04–0.09 |
| Micro vs macro Dice | v2 reported micro; gate needs per-image **macro** |

## Alignment audit (mismatched val sample n=40)

| Mode | fg−bg intensity | grad_ratio (fg/bg) |
|---|---:|---:|
| **resize mask → image** | **+11.8** | **2.06** |
| letterbox mask into image | −11.4 | 1.11 |
| center-crop img to mask aspect | −14.1 | 1.15 |
| matched pairs (control) | +78.1 | 1.33 |

**Conclusion:** For mismatched stems, NEAREST-resize mask to image HxW before training/eval. Then letterbox the **aligned** pair (preserve aspect). Do not use AmbrosM FOV crop for fasc (drops GT / wrong box).

## Ceiling note

Thin 2px labels + ~60% non-uniform resize correspondence may cap Dice below 0.6 even after align. Document achieved native macro Dice honestly; do not invent q / TIFF dpi.
