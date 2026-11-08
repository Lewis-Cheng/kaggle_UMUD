#!/usr/bin/env python3
"""Train TinyUNet binary seg for apo or fasc masks (duplicate-aware split).

Writes checkpoint + val Dice. Attention remains blocked; this is the Tier-B
seg → GeometryHead prerequisite, not an LB claim.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_split_assignments  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.seg.unet import TinyUNet, dice_bce_loss  # noqa: E402


def _stem_index(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in list_images(folder)}


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        *,
        size: int = 256,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.size = int(size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(mp), dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)

        # Resize with PIL (nearest for mask)
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
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            # mild brightness
            if random.random() < 0.5:
                img = np.clip(img * random.uniform(0.85, 1.15), 0.0, 1.0)

        x = torch.from_numpy(img[None, ...])  # 1,H,W
        y = torch.from_numpy(mask[None, ...])
        return x, y


@torch.no_grad()
def eval_dice(model: TinyUNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    inter = 0.0
    den = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = (torch.sigmoid(logits) > 0.5).float()
        inter += float((probs * y).sum().item())
        den += float(probs.sum().item() + y.sum().item())
    return float((2.0 * inter + 1e-6) / (den + 1e-6))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--task", choices=("apo", "fasc"), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--max-train", type=int, default=0, help="smoke cap")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    task_name = "aponeurosis" if args.task == "apo" else "fascicle"
    img_folder = layout.apo_imgs if args.task == "apo" else layout.fasc_imgs
    mask_folder = layout.apo_masks if args.task == "apo" else layout.fasc_masks
    imgs = _stem_index(img_folder)
    masks = _stem_index(mask_folder)

    train_stems = [
        str(r["image_id"])
        for r in rows
        if r.get("task") == task_name and r.get("split") == "train"
    ]
    val_stems = [
        str(r["image_id"])
        for r in rows
        if r.get("task") == task_name and r.get("split") == "val"
    ]

    def pairs(stems: list[str]) -> list[tuple[Path, Path]]:
        out = []
        for s in stems:
            if s in imgs and s in masks:
                out.append((imgs[s], masks[s]))
        return out

    train_pairs = pairs(train_stems)
    val_pairs = pairs(val_stems)
    if args.max_train and args.max_train > 0:
        train_pairs = train_pairs[: args.max_train]
        val_pairs = val_pairs[: max(8, args.max_train // 4)]

    if len(train_pairs) < 10:
        print(f"ERROR: too few train pairs: {len(train_pairs)}", file=sys.stderr)
        return 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"task={args.task} train={len(train_pairs)} val={len(val_pairs)} device={device}"
    )

    train_ds = PairDataset(train_pairs, size=args.size, augment=True)
    val_ds = PairDataset(val_pairs, size=args.size, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TinyUNet(in_ch=1, base=args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    history: list[dict] = []
    ckpt_path = args.out_dir / f"seg_{args.task}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = dice_bce_loss(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        dice = eval_dice(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)) if losses else None,
            "val_dice": dice,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs} loss={row['train_loss_mean']:.4f} "
            f"val_dice={dice:.4f}"
        )
        if dice > best_dice:
            best_dice = dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": args.task,
                    "size": args.size,
                    "base_ch": args.base_ch,
                    "val_dice": best_dice,
                    "epoch": epoch,
                    "seed": args.seed,
                },
                ckpt_path,
            )

    meta = {
        "task": args.task,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "best_val_dice": best_dice,
        "epochs": args.epochs,
        "size": args.size,
        "base_ch": args.base_ch,
        "device": str(device),
        "checkpoint": str(ckpt_path),
        "history": history,
        "attention_still_blocked": True,
        "note": "seg baseline for Tier B GeometryHead; NOT LB / NOT official S",
    }
    (args.out_dir / f"seg_{args.task}_train.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"best_dice={best_dice:.4f} ckpt={ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
