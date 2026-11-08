#!/usr/bin/env python3
"""Fasc seg v3c: SAFE FOV (only if box fits image), dilate soft targets, FascUNet.

Previous v3b applied AmbrosM 800x1200 FOV onto 556x996 fasc images → dropped
~25% of GT pixels. Here FOV is used only when fully inside the image.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_split_assignments  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.seg.unet_v2 import FascUNet  # noqa: E402


def load_fov_map(scale_csv: Path) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    with scale_csv.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("pool") != "fasc":
                continue
            try:
                box = (int(r["fov_l"]), int(r["fov_t"]), int(r["fov_r"]), int(r["fov_b"]))
                ih, iw = int(float(r["height"])), int(float(r["width"]))
            except (TypeError, ValueError, KeyError):
                continue
            l, t, rr, b = box
            # SAFE: FOV must lie inside recorded image size (margin 2px)
            if l >= 0 and t >= 0 and rr <= iw + 2 and b <= ih + 2 and rr > l + 32 and b > t + 32:
                # also require FOV covers majority of image area OR matches dims family
                out[str(r["stem"])] = (max(0, l), max(0, t), min(iw, rr), min(ih, b))
    return out


def safe_crop(arr: np.ndarray, fov: tuple[int, int, int, int] | None) -> np.ndarray:
    if fov is None:
        return arr
    h, w = arr.shape[:2]
    l, t, r, b = fov
    # reject if FOV extends far outside actual array (wrong family coords)
    if r > w + 2 or b > h + 2 or l < -2 or t < -2:
        return arr
    l, t = max(0, l), max(0, t)
    r, b = min(w, r), min(h, b)
    if (r - l) < w * 0.5 or (b - t) < h * 0.5:
        # FOV too small relative to image → likely wrong template; skip
        return arr
    if r <= l or b <= t:
        return arr
    return arr[t:b, l:r]


def dilate_mask(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_u8
    return np.asarray(Image.fromarray(mask_u8).filter(ImageFilter.MaxFilter(size=2 * radius + 1)))


class FascDataset(Dataset):
    def __init__(self, pairs, fov_map, *, size=512, dilate=2, augment=False):
        self.pairs = pairs
        self.fov_map = fov_map
        self.size = int(size)
        self.dilate = int(dilate)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem, ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.uint8)
        mask = np.asarray(Image.open(mp), dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = ((mask > 0).astype(np.uint8)) * 255
        fov = self.fov_map.get(stem)
        img = safe_crop(img, fov)
        mask = safe_crop(mask, fov)
        soft = dilate_mask(mask, self.dilate) if self.dilate else mask

        img_a = np.asarray(Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255.0
        hard_a = (np.asarray(Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)) > 127).astype(np.float32)
        soft_a = (np.asarray(Image.fromarray(soft).resize((self.size, self.size), Image.NEAREST)) > 127).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                img_a = np.fliplr(img_a).copy(); hard_a = np.fliplr(hard_a).copy(); soft_a = np.fliplr(soft_a).copy()
            if random.random() < 0.4:
                img_a = np.clip(img_a * random.uniform(0.75, 1.3) + random.uniform(-0.05, 0.05), 0, 1)
            if random.random() < 0.25:
                ang = random.uniform(-10, 10)
                img_a = np.asarray(Image.fromarray((img_a * 255).astype(np.uint8)).rotate(ang, Image.BILINEAR, fillcolor=0), dtype=np.float32) / 255.0
                hard_a = (np.asarray(Image.fromarray((hard_a * 255).astype(np.uint8)).rotate(ang, Image.NEAREST, fillcolor=0)) > 127).astype(np.float32)
                soft_a = (np.asarray(Image.fromarray((soft_a * 255).astype(np.uint8)).rotate(ang, Image.NEAREST, fillcolor=0)) > 127).astype(np.float32)

        return torch.from_numpy(img_a[None]), torch.from_numpy(soft_a[None]), torch.from_numpy(hard_a[None])


def loss_fn(logits, soft, hard, pos_weight: float):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, soft, pos_weight=pw)
    probs = torch.sigmoid(logits)
    def dice(p, t, eps=1e-6):
        inter = (p * t).sum()
        return 1.0 - (2 * inter + eps) / (p.sum() + t.sum() + eps)
    # emphasize hard thin GT more than v3b
    return bce + 0.4 * dice(probs, soft) + 0.6 * dice(probs, hard)


@torch.no_grad()
def eval_dice(model, loader, device, thr=0.5):
    model.eval()
    inter = den = 0.0
    for x, _s, hard in loader:
        x, hard = x.to(device), hard.to(device)
        pred = (torch.sigmoid(model(x)) > thr).float()
        inter += float((pred * hard).sum().item())
        den += float(pred.sum().item() + hard.sum().item())
    return float((2 * inter + 1e-6) / (den + 1e-6))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--scale-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base-ch", type=int, default=48)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--pos-weight", type=float, default=40.0)
    ap.add_argument("--device", type=str, default="cuda:2")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
    masks = {p.stem: p for p in list_images(layout.fasc_masks)}
    fov_map = load_fov_map(args.scale_csv)

    def pairs(split):
        out = []
        for r in rows:
            if r.get("task") == "fascicle" and r.get("split") == split:
                s = str(r["image_id"])
                if s in imgs and s in masks:
                    out.append((s, imgs[s], masks[s]))
        return out

    train_pairs, val_pairs = pairs("train"), pairs("val")
    # count how many actually get cropped
    n_crop = 0
    for stem, ip, _ in train_pairs[:200]:
        a = np.asarray(Image.open(ip).convert("L"))
        before = a.shape
        after = safe_crop(a, fov_map.get(stem)).shape
        if after != before:
            n_crop += 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"fasc_v3c train={len(train_pairs)} val={len(val_pairs)} fov_map={len(fov_map)} "
          f"crop_in_sample200={n_crop} pos_w={args.pos_weight} size={args.size} device={device}")

    train_loader = DataLoader(FascDataset(train_pairs, fov_map, size=args.size, dilate=args.dilate, augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(FascDataset(val_pairs, fov_map, size=args.size, dilate=0, augment=False),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = FascUNet(1, args.base_ch, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_best.pt"
    best = -1.0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, soft, hard in train_loader:
            x, soft, hard = x.to(device), soft.to(device), hard.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, soft, hard, args.pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        d35 = eval_dice(model, val_loader, device, 0.35)
        d50 = eval_dice(model, val_loader, device, 0.5)
        hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "dice35": d35, "dice50": d50})
        print(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} dice@0.35={d35:.4f} dice@0.5={d50:.4f}")
        score = max(d35, d50)
        if score > best:
            best = score
            torch.save({
                "model": model.state_dict(), "task": "fasc", "arch": "FascUNet",
                "size": args.size, "base_ch": args.base_ch, "val_dice": best,
                "val_dice_thr035": d35, "val_dice_thr050": d50, "epoch": epoch,
                "variant": "fasc_v3c_safe_fov", "dilate": args.dilate,
                "pos_weight": args.pos_weight,
                "eval_thr_preferred": 0.35 if d35 >= d50 else 0.5,
            }, ckpt)
    (args.out_dir / "seg_fasc_train.json").write_text(json.dumps({
        "variant": "fasc_v3c_safe_fov", "best_val_dice": best, "history": hist,
        "fov_map": len(fov_map), "n_train": len(train_pairs), "checkpoint": str(ckpt),
    }, indent=2))
    print(f"best_dice={best:.4f} ckpt={ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
