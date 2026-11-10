#!/usr/bin/env python3
"""Map-induced gap: frozen geometry on GT masks versus Frundles probability maps.

No training. Same millimetre scale on both, so the difference is the maps,
not a change of definition. A small gap means a better segmenter cannot close
the public gap by itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from umud.geometry.host_faithful import measure  # noqa: E402
from umud.seg.frundles_smp import load_ckpt, predict_prob  # noqa: E402

APO_IMG = ROOT / "data/raw/umud/apo_imgs_v1"
APO_MASK = ROOT / "data/raw/umud/apo_masks_v1"
FASC_IMG = ROOT / "data/raw/umud/fasc_imgs_v1"
FASC_MASK = ROOT / "data/raw/umud/fasc_masks_v1"
WEIGHTS = ROOT / "data/external/frundles_weights"
OUT = ROOT / "reports/phase3/chase018/oracle_mask_gap.json"
# Median-ish mm per pixel from the test scale table. Same value for mask and map.
MM_PER_PX = 1.0 / 14.0


def _index(root: Path) -> dict[str, Path]:
    found = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            found.setdefault(path.stem, path)
    return found


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(path)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def main() -> int:
    apo_i, apo_m = _index(APO_IMG), _index(APO_MASK)
    fasc_i, fasc_m = _index(FASC_IMG), _index(FASC_MASK)
    stems = sorted(set(apo_i) & set(apo_m) & set(fasc_i) & set(fasc_m))
    rng = np.random.default_rng(0)
    # One image per 26 stems keeps the sample under a few minutes on CPU.
    pick = list(rng.choice(stems, size=min(40, len(stems)), replace=False))
    device = "cpu"
    apo_model, apo_size = load_ckpt(WEIGHTS / "apo_resnet34_384x576_f0.pt", device)
    fasc_model, fasc_size = load_ckpt(WEIGHTS / "fasc_resnet34_384x576_f0.pt", device)
    rows = []
    for stem in pick:
        gray = cv2.imread(str(apo_i[stem]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        apo_gt = _load_mask(apo_m[stem], gray.shape)
        fasc_gt = _load_mask(fasc_m[stem], gray.shape)
        apo_pr = predict_prob(apo_model, apo_size, gray, device)
        fasc_pr = predict_prob(fasc_model, fasc_size, gray, device)
        # Resize predicted maps back to the mask grid so both geometries share pixels.
        apo_pr = cv2.resize(apo_pr, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        fasc_pr = cv2.resize(fasc_pr, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        gt = measure(apo_gt, fasc_gt, MM_PER_PX, MM_PER_PX)
        pr = measure(apo_pr, fasc_pr, MM_PER_PX, MM_PER_PX)
        if not (gt.get("ok") and pr.get("ok")):
            rows.append({"stem": stem, "ok": False})
            continue
        rows.append(
            {
                "stem": stem,
                "ok": True,
                "d_pa": abs(float(pr["pa_deg"]) - float(gt["pa_deg"])),
                "d_fl": abs(float(pr["fl_mm"]) - float(gt["fl_mm"])),
                "d_mt": abs(float(pr["mt_mm"]) - float(gt["mt_mm"])),
            }
        )
        print(stem, rows[-1], flush=True)
    ok = [r for r in rows if r["ok"]]
    def mae(key: str) -> float | None:
        if not ok:
            return None
        return float(np.mean([r[key] for r in ok]))
    summary = {
        "n_sampled": len(pick),
        "n_both_ok": len(ok),
        "mae_pa_deg": mae("d_pa"),
        "mae_fl_mm": mae("d_fl"),
        "mae_mt_mm": mae("d_mt"),
        "mm_per_px": MM_PER_PX,
        "maps": "frundles public apo+fasc",
        "masks": "competition apo+fasc GT on the 1048-image overlap",
        "note": "Gap is map versus mask under one frozen geometry and one scale. It is not an error against expert PA/FL/MT.",
    }
    # Normalized contribution if this gap were the whole public error.
    if ok:
        summary["S_if_gap_were_the_error"] = float(
            np.mean(
                [
                    summary["mae_pa_deg"] / 6.0,
                    summary["mae_fl_mm"] / 12.0,
                    summary["mae_mt_mm"] / 3.0,
                ]
            )
        )
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
