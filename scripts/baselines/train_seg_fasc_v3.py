#!/usr/bin/env python3
"""Fasc seg v3: AmbrosM FOV crop + dilate soft targets + FascUNet + strong imbalance loss.

Eval Dice always on original (undilated) masks at native FOV resolution.
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
    out: dict[str, tuple[int, int, int, int]] = {}
    with scale_csv.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("pool") != "fasc":
                continue
            try:
                l, t, rr, b = int(r["fov_l"]), int(r["fov_t"]), int(r["fov_r"]), int(r["fov_b"])
            except (TypeError, ValueError, KeyError):
                continue
            if rr > l and b > t:
                out[str(r["stem"])] = (l, t, rr, b)
    return out


def crop_fov(arr: np.ndarray, fov: tuple[int, int, int, int] | None) -> np.ndarray:
    if fov is None:
        return arr
    l, t, r, b = fov
    h, w = arr.shape[:2]
    l, t = max(0, l), max(0, t)
    r, b = min(w, r), min(h, b)
    if r <= l or b <= t:
        return arr
    return arr[t:b, l:r]


def dilate_mask(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_u8
    img = Image.fromarray(mask_u8)
    # MaxFilter approximates binary dilation
    k = 2 * radius + 1
    return np.asarray(img.filter(ImageFilter.MaxFilter(size=k)))


class FascDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, Path, Path]],
        fov_map: dict[str, tuple[int, int, int, int]],
        *,
        size: int = 512,
        dilate: int = 2,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.fov_map = fov_map
        self.size = int(size)
        self.dilate = int(dilate)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        stem, ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.uint8)
        mask = np.asarray(Image.open(mp), dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8) * 255
        fov = self.fov_map.get(stem)
        img = crop_fov(img, fov)
        mask = crop_fov(mask, fov)
        # soft target: dilate for training stability on thin lines
        soft = dilate_mask(mask, self.dilate) if self.dilate > 0 else mask

        img_p = Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR)
        hard_p = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)
        soft_p = Image.fromarray(soft).resize((self.size, self.size), Image.NEAREST)

        img_a = np.asarray(img_p, dtype=np.float32) / 255.0
        hard_a = (np.asarray(hard_p) > 127).astype(np.float32)
        soft_a = (np.asarray(soft_p) > 127).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                img_a = np.fliplr(img_a).copy()
                hard_a = np.fliplr(hard_a).copy()
                soft_a = np.fliplr(soft_a).copy()
            if random.random() < 0.3:
                img_a = np.flipud(img_a).copy()
                hard_a = np.flipud(hard_a).copy()
                soft_a = np.flipud(soft_a).copy()
            if random.random() < 0.5:
                img_a = np.clip(img_a * random.uniform(0.75, 1.25) + random.uniform(-0.05, 0.05), 0, 1)
            if random.random() < 0.3:
                # small rotation via PIL
                ang = random.uniform(-8, 8)
                img_a = np.asarray(
                    Image.fromarray((img_a * 255).astype(np.uint8)).rotate(ang, resample=Image.BILINEAR, fillcolor=0),
                    dtype=np.float32,
                ) / 255.0
                hard_a = (
                    np.asarray(
                        Image.fromarray((hard_a * 255).astype(np.uint8)).rotate(ang, resample=Image.NEAREST, fillcolor=0)
                    )
                    > 127
                ).astype(np.float32)
                soft_a = (
                    np.asarray(
                        Image.fromarray((soft_a * 255).astype(np.uint8)).rotate(ang, resample=Image.NEAREST, fillcolor=0)
                    )
                    > 127
                ).astype(np.float32)

        return (
            torch.from_numpy(img_a[None, ...]),
            torch.from_numpy(soft_a[None, ...]),
            torch.from_numpy(hard_a[None, ...]),
        )


def loss_fn(logits, soft, hard, *, pos_weight: float, soft_w: float = 0.7):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, soft, pos_weight=pw)
    probs = torch.sigmoid(logits)
    # soft Dice on dilated target + hard Dice on thin GT
    def dice(p, t, eps=1e-6):
        inter = (p * t).sum()
        return 1.0 - (2 * inter + eps) / (p.sum() + t.sum() + eps)

    return bce + soft_w * dice(probs, soft) + (1.0 - soft_w) * dice(probs, hard)


@torch.no_grad()
def eval_dice(model, loader, device, thr=0.5):
    model.eval()
    inter = den = 0.0
    for batch in loader:
        x, _soft, hard = batch
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base-ch", type=int, default=48)
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--pos-weight", type=float, default=0.0, help="0=auto from train fg")
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-thr", type=float, default=0.35)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
    masks = {p.stem: p for p in list_images(layout.fasc_masks)}
    fov_map = load_fov_map(args.scale_csv)

    def pairs(split: str):
        out = []
        for r in rows:
            if r.get("task") == "fascicle" and r.get("split") == split:
                s = str(r["image_id"])
                if s in imgs and s in masks:
                    out.append((s, imgs[s], masks[s]))
        return out

    train_pairs, val_pairs = pairs("train"), pairs("val")
    # auto pos weight from small sample
    fg_sum = bg_sum = 0.0
    for stem, ip, mp in train_pairs[:: max(1, len(train_pairs) // 200)]:
        m = np.asarray(Image.open(mp))
        if m.ndim == 3:
            m = m[..., 0]
        m = crop_fov((m > 0).astype(np.uint8), fov_map.get(stem))
        fg_sum += float(m.sum())
        bg_sum += float(m.size - m.sum())
    auto_pw = max(bg_sum / max(fg_sum, 1.0), 1.0)
    # cap extreme weights
    auto_pw = float(min(max(auto_pw, 20.0), 120.0))
    pos_weight = float(args.pos_weight) if args.pos_weight > 0 else auto_pw

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"fasc_v3 train={len(train_pairs)} val={len(val_pairs)} "
        f"fov_stems={len(fov_map)} pos_weight={pos_weight:.1f} size={args.size} device={device}"
    )

    train_loader = DataLoader(
        FascDataset(train_pairs, fov_map, size=args.size, dilate=args.dilate, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        FascDataset(val_pairs, fov_map, size=args.size, dilate=0, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = FascUNet(in_ch=1, base=args.base_ch, use_aspp=True).to(device)
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
            loss = loss_fn(logits, soft, hard, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        dice = eval_dice(model, val_loader, device, thr=args.eval_thr)
        # also thr 0.5
        dice50 = eval_dice(model, val_loader, device, thr=0.5)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_dice_thr035": dice,
            "val_dice_thr050": dice50,
        }
        hist.append(row)
        print(
            f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} "
            f"dice@0.35={dice:.4f} dice@0.5={dice50:.4f}"
        )
        score = max(dice, dice50)
        if score > best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": "fasc",
                    "arch": "FascUNet",
                    "size": args.size,
                    "base_ch": args.base_ch,
                    "val_dice": best,
                    "val_dice_thr035": dice,
                    "val_dice_thr050": dice50,
                    "epoch": epoch,
                    "variant": "fasc_v3_fov_dilate_aspp",
                    "dilate": args.dilate,
                    "pos_weight": pos_weight,
                    "eval_thr_preferred": 0.35 if dice >= dice50 else 0.5,
                },
                ckpt,
            )

    meta = {
        "task": "fasc",
        "variant": "fasc_v3_fov_dilate_aspp",
        "best_val_dice": best,
        "pos_weight": pos_weight,
        "fov_stems": len(fov_map),
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "size": args.size,
        "dilate": args.dilate,
        "history": hist,
        "checkpoint": str(ckpt),
    }
    (args.out_dir / "seg_fasc_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_dice={best:.4f} ckpt={ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
