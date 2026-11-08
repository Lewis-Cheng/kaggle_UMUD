#!/usr/bin/env python3
"""Infer apo/fasc masks with TinyUNet checkpoints; write PNG masks + index JSON.

Predictions keep native image resolution (nearest upsample from network size).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.seg.unet import TinyUNet  # noqa: E402


def _load_ckpt(path: Path, device: torch.device) -> tuple[TinyUNet, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    size = int(ckpt.get("size", 256))
    base = int(ckpt.get("base_ch", 32))
    model = TinyUNet(in_ch=1, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, size


@torch.no_grad()
def predict_mask(
    model: TinyUNet,
    img_path: Path,
    *,
    size: int,
    device: torch.device,
    thr: float = 0.5,
) -> np.ndarray:
    raw = Image.open(img_path).convert("L")
    w, h = raw.size
    arr = np.asarray(raw, dtype=np.float32) / 255.0
    resized = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR
    )
    x = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0)[None, None, ...].to(
        device
    )
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    small = (prob >= thr).astype(np.uint8) * 255
    full = np.asarray(
        Image.fromarray(small).resize((w, h), Image.NEAREST), dtype=np.uint8
    )
    return full


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--apo-ckpt", type=Path, required=True)
    ap.add_argument("--fasc-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--split-assignments", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    apo_model, apo_size = _load_ckpt(args.apo_ckpt, device)
    fasc_model, fasc_size = _load_ckpt(args.fasc_ckpt, device)
    layout = discover_layout(args.data_root)

    out_apo = args.out_dir / "pred_apo_masks"
    out_fasc = args.out_dir / "pred_fasc_masks"
    out_apo.mkdir(parents=True, exist_ok=True)
    out_fasc.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, Path]] = []  # legacy; work list built below
    apo_imgs = {p.stem: p for p in list_images(layout.apo_imgs)}
    fasc_imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}

    # Val/train: track stems; inference uses task-native images when available.
    val_train_stems: set[str] = set()
    if "val" in args.splits or "train" in args.splits:
        if args.split_assignments is None:
            print("ERROR: --split-assignments required for train/val", file=sys.stderr)
            return 1
        from umud.baselines.io import load_split_assignments

        rows = load_split_assignments(args.split_assignments)
        for r in rows:
            if r.get("split") in args.splits and r.get("split") != "test":
                val_train_stems.add(str(r["image_id"]))

    # Build work list of (split, stem, apo_src|None, fasc_src|None)
    work: list[tuple[str, str, Path | None, Path | None]] = []
    if "test" in args.splits:
        # replace earlier simple test append — rebuild cleanly
        pass
    # Clear and rebuild targets-style work list
    work = []
    if "test" in args.splits:
        for p in list_images(layout.test_imgs):
            work.append(("test", p.stem, p, p))
    for stem in sorted(val_train_stems):
        split = "val"
        work.append(
            (
                split,
                stem,
                apo_imgs.get(stem),
                fasc_imgs.get(stem),
            )
        )

    if args.max_images and args.max_images > 0:
        work = work[: args.max_images]

    index = []
    for i, (split, stem, apo_src, fasc_src) in enumerate(work):
        apo = None
        fasc = None
        if apo_src is not None:
            apo = predict_mask(apo_model, apo_src, size=apo_size, device=device, thr=args.thr)
        elif fasc_src is not None:
            # fallback: still emit apo pred from fasc image so GH has something
            apo = predict_mask(apo_model, fasc_src, size=apo_size, device=device, thr=args.thr)
            apo_src = fasc_src
        if fasc_src is not None:
            fasc = predict_mask(fasc_model, fasc_src, size=fasc_size, device=device, thr=args.thr)
        elif apo_src is not None:
            fasc = predict_mask(fasc_model, apo_src, size=fasc_size, device=device, thr=args.thr)
            fasc_src = apo_src
        if apo is None or fasc is None:
            continue
        apo_out = out_apo / f"{stem}.png"
        fasc_out = out_fasc / f"{stem}.png"
        Image.fromarray(apo).save(apo_out)
        Image.fromarray(fasc).save(fasc_out)
        index.append(
            {
                "split": split,
                "stem": stem,
                "apo_source_image": str(apo_src) if apo_src else None,
                "fasc_source_image": str(fasc_src) if fasc_src else None,
                "apo_mask": str(apo_out),
                "fasc_mask": str(fasc_out),
                "apo_nnz": int((apo > 0).sum()),
                "fasc_nnz": int((fasc > 0).sum()),
            }
        )
        if (i + 1) % 50 == 0 or (i + 1) == len(work):
            print(f"inferred {i+1}/{len(work)}")

    meta = {
        "n": len(index),
        "apo_ckpt": str(args.apo_ckpt),
        "fasc_ckpt": str(args.fasc_ckpt),
        "thr": args.thr,
        "device": str(device),
        "splits": args.splits,
    }
    (args.out_dir / "pred_masks_index.json").write_text(
        json.dumps({"meta": meta, "items": index}, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(index)} mask pairs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
