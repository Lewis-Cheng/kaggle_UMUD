#!/usr/bin/env python3
"""Fasc TinyUNet + dilate soft targets + pos_w=80 @384 — aim to beat 0.26 toward 0.6."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from umud.baselines.io import load_split_assignments
from umud.data.layout import discover_layout, list_images
from umud.seg.unet import TinyUNet


def dilate_mask(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_u8
    return np.asarray(Image.fromarray(mask_u8).filter(ImageFilter.MaxFilter(size=2 * radius + 1)))


class PairDataset(Dataset):
    def __init__(self, pairs, *, size=384, dilate=2, augment=False):
        self.pairs = pairs
        self.size = int(size)
        self.dilate = int(dilate)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem, ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        hard = (np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8)
        soft = dilate_mask(hard * 255, self.dilate).astype(np.float32) / 255.0
        hard_f = hard.astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                soft = np.ascontiguousarray(soft[:, ::-1])
                hard_f = np.ascontiguousarray(hard_f[:, ::-1])
            if random.random() < 0.4:
                gain = 0.85 + 0.3 * random.random()
                bias = (random.random() - 0.5) * 0.08
                img = np.clip(img * gain + bias, 0, 1)
        img_p = Image.fromarray((img * 255).astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        soft_p = Image.fromarray((soft * 255).astype(np.uint8)).resize((self.size, self.size), Image.NEAREST)
        hard_p = Image.fromarray((hard_f * 255).astype(np.uint8)).resize((self.size, self.size), Image.NEAREST)
        x = torch.from_numpy(np.asarray(img_p, dtype=np.float32) / 255.0).unsqueeze(0)
        soft_t = torch.from_numpy(np.asarray(soft_p, dtype=np.float32) / 255.0).unsqueeze(0)
        hard_t = torch.from_numpy((np.asarray(hard_p) > 127).astype(np.float32)).unsqueeze(0)
        return x, soft_t, hard_t


def loss_fn(logits, soft, hard, pos_weight: float):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, soft, pos_weight=pw)
    probs = torch.sigmoid(logits)
    inter = (probs * soft).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + soft.sum(dim=(2, 3))
    dice_s = 1.0 - (2 * inter + 1) / (den + 1)
    inter_h = (probs * hard).sum(dim=(2, 3))
    den_h = probs.sum(dim=(2, 3)) + hard.sum(dim=(2, 3))
    dice_h = 1.0 - (2 * inter_h + 1) / (den_h + 1)
    return bce + dice_s.mean() + 0.5 * dice_h.mean()


@torch.no_grad()
def eval_dice(model, loader, device, thr=0.5):
    model.eval()
    dices = []
    for x, soft, hard in loader:
        x, hard = x.to(device), hard.to(device)
        pred = (torch.sigmoid(model(x)) > thr).float()
        inter = (pred * hard).sum(dim=(1, 2, 3))
        den = pred.sum(dim=(1, 2, 3)) + hard.sum(dim=(1, 2, 3))
        dices.extend(((2 * inter + 1e-6) / (den + 1e-6)).cpu().tolist())
    return float(np.mean(dices)) if dices else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=80.0)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
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
                    out.append((s, imgs[s], masks[s]))
        return out

    train_pairs, val_pairs = pairs("train"), pairs("val")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"fasc_v2f train={len(train_pairs)} val={len(val_pairs)} size={args.size} dilate={args.dilate} pos_w={args.pos_weight} device={device}", flush=True)
    train_loader = DataLoader(PairDataset(train_pairs, size=args.size, dilate=args.dilate, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(PairDataset(val_pairs, size=args.size, dilate=0, augment=False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = TinyUNet(1, args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_v2f_best.pt"
    best = -1.0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, soft, hard in train_loader:
            x, soft, hard = x.to(device), soft.to(device), hard.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), soft, hard, args.pos_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.item()))
        sched.step()
        d35 = eval_dice(model, val_loader, device, 0.35)
        d50 = eval_dice(model, val_loader, device, 0.5)
        score = max(d35, d50)
        hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "dice35": d35, "dice50": d50})
        print(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} dice@0.35={d35:.4f} dice@0.5={d50:.4f}", flush=True)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "task": "fasc", "arch": "TinyUNet", "size": args.size, "base_ch": args.base_ch, "val_dice": best, "val_dice_thr035": d35, "val_dice_thr050": d50, "epoch": epoch, "variant": "fasc_v2f_dilate", "dilate": args.dilate, "pos_weight": args.pos_weight, "eval_thr_preferred": 0.35 if d35 >= d50 else 0.5}, ckpt)
    (args.out_dir / "seg_fasc_v2f_train.json").write_text(json.dumps({"variant": "fasc_v2f_dilate", "best_val_dice": best, "history": hist, "checkpoint": str(ckpt)}, indent=2))
    print(f"best_dice={best:.4f} ckpt={ckpt}")


if __name__ == "__main__":
    main()
