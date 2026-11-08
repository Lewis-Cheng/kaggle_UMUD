#!/usr/bin/env python3
"""Stronger fasc TinyUNet: weighted BCE+Dice, size 384, more epochs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_split_assignments  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.seg.unet import TinyUNet  # noqa: E402


def weighted_dice_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: float = 40.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum()
    den = probs.sum() + targets.sum()
    dice = 1.0 - (2.0 * inter + eps) / (den + eps)
    return bce + dice


class PairDataset(Dataset):
    def __init__(self, pairs, *, size=384, augment=False):
        self.pairs = pairs
        self.size = int(size)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(mp), dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        img_p = Image.fromarray((img * 255).astype(np.uint8)).resize(
            (self.size, self.size), Image.BILINEAR
        )
        mask_p = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (self.size, self.size), Image.NEAREST
        )
        img = np.asarray(img_p, dtype=np.float32) / 255.0
        mask = (np.asarray(mask_p, dtype=np.float32) > 127).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() < 0.3:
                img = np.clip(img * random.uniform(0.8, 1.2), 0, 1)
        return (
            torch.from_numpy(img[None, ...]),
            torch.from_numpy(mask[None, ...]),
        )


@torch.no_grad()
def eval_dice(model, loader, device):
    model.eval()
    inter = den = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        probs = (torch.sigmoid(model(x)) > 0.5).float()
        inter += float((probs * y).sum().item())
        den += float(probs.sum().item() + y.sum().item())
    return float((2 * inter + 1e-6) / (den + 1e-6))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--pos-weight", type=float, default=40.0)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
    masks = {p.stem: p for p in list_images(layout.fasc_masks)}

    def pairs(split):
        out = []
        for r in rows:
            if r.get("task") == "fascicle" and r.get("split") == split:
                s = str(r["image_id"])
                if s in imgs and s in masks:
                    out.append((imgs[s], masks[s]))
        return out

    train_pairs, val_pairs = pairs("train"), pairs("val")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"fasc_v2 train={len(train_pairs)} val={len(val_pairs)} device={device}")

    train_loader = DataLoader(
        PairDataset(train_pairs, size=args.size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs, size=args.size, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = TinyUNet(1, args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_best.pt"
    best = -1.0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = weighted_dice_bce(model(x), y, pos_weight=args.pos_weight)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        dice = eval_dice(model, val_loader, device)
        hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_dice": dice})
        print(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} val_dice={dice:.4f}")
        if dice > best:
            best = dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": "fasc",
                    "size": args.size,
                    "base_ch": args.base_ch,
                    "val_dice": best,
                    "epoch": epoch,
                    "variant": "fasc_v2_weighted",
                },
                ckpt,
            )
    meta = {
        "task": "fasc",
        "variant": "fasc_v2_weighted",
        "best_val_dice": best,
        "history": hist,
        "checkpoint": str(ckpt),
    }
    (args.out_dir / "seg_fasc_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_dice={best:.4f} ckpt={ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
