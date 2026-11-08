#!/usr/bin/env python3
"""Fine-tune Frundles fasc ResNet34 U-Net on competition fasc masks.

- Start from data/external/frundles_weights/fasc_resnet34_384x576_f0.pt
- Keep 384x576; Dice+BCE + soft-clDice (orientation/skeleton aware)
- dHash group split (leak-safe) via Hamming on images
- Save best under reports/phase3/chase018/seg_ckpts/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.geometry.host_faithful import dhash  # noqa: E402
from umud.seg.frundles_smp import load_ckpt  # noqa: E402


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def soft_skel(x: torch.Tensor, iters: int = 8) -> torch.Tensor:
    img = x
    skel = F.relu(img - soft_erode(img))
    for _ in range(iters):
        img = soft_erode(img)
        delta = F.relu(img - soft_erode(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(probs: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    skel_p = soft_skel(probs)
    skel_g = soft_skel(gt)
    tprec = (skel_p * gt).sum((2, 3)) / (skel_p.sum((2, 3)) + eps)
    tsens = (skel_g * probs).sum((2, 3)) / (skel_g.sum((2, 3)) + eps)
    cl = (2 * tprec * tsens) / (tprec + tsens + eps)
    return 1.0 - cl.mean()


def dice_loss(probs: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = 2 * (probs * gt).sum((2, 3)) + eps
    den = probs.sum((2, 3)) + gt.sum((2, 3)) + eps
    return 1.0 - (num / den).mean()


def groups_from_hashes(hs: list[np.ndarray], max_dist: int = 12) -> np.ndarray:
    parent = list(range(len(hs)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    arr = np.array(hs, dtype=np.uint8)
    for i in range(len(arr)):
        for j in np.nonzero((arr[i + 1 :] != arr[i]).sum(1) <= max_dist)[0] + i + 1:
            parent[find(int(j))] = find(i)
    return np.array([find(i) for i in range(len(hs))])


def discover_pairs(img_dir: Path, mask_dir: Path) -> list[tuple[str, Path, Path]]:
    imgs = sorted(list(img_dir.rglob("*.tif")) + list(img_dir.rglob("*.png")) + list(img_dir.rglob("*.jpg")))
    pairs = []
    for ip in imgs:
        # try same relative name under mask_dir
        candidates = [
            mask_dir / ip.name,
            mask_dir / (ip.stem + ".png"),
            mask_dir / (ip.stem + ".tif"),
            mask_dir / (ip.stem + "_mask.png"),
            mask_dir / (ip.stem + "_mask.tif"),
        ]
        # also search by stem recursively
        hits = list(mask_dir.rglob(ip.name)) + list(mask_dir.rglob(ip.stem + ".*"))
        for h in hits:
            if h.suffix.lower() in {".png", ".tif", ".jpg", ".jpeg"}:
                candidates.append(h)
        mp = next((c for c in candidates if c.exists()), None)
        if mp is not None:
            pairs.append((ip.stem, ip, mp))
    # unique by stem
    seen = set()
    out = []
    for stem, ip, mp in pairs:
        if stem in seen:
            continue
        seen.add(stem)
        out.append((stem, ip, mp))
    return out


class FascDS(Dataset):
    def __init__(self, pairs, size=(384, 576), augment=False):
        self.pairs = pairs
        self.size = size  # (H, W)
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, ip, mp = self.pairs[idx]
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if g is None or m is None:
            raise RuntimeError(f"bad pair {ip} {mp}")
        if m.shape[:2] != g.shape[:2]:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        hard = (m > 127).astype(np.uint8)
        if self.augment:
            if random.random() < 0.5:
                g = np.ascontiguousarray(g[:, ::-1])
                hard = np.ascontiguousarray(hard[:, ::-1])
            if random.random() < 0.4:
                alpha = random.uniform(0.8, 1.25)
                beta = random.uniform(-12, 12)
                g = np.clip(g.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                ang = random.uniform(-8, 8)
                H, W = g.shape
                M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
                g = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                hard = cv2.warpAffine(hard, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        oh, ow = self.size
        g = cv2.resize(g, (ow, oh), interpolation=cv2.INTER_AREA)
        hard = cv2.resize(hard, (ow, oh), interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(g.astype(np.float32) / 255.0)[None]
        y = torch.from_numpy(hard.astype(np.float32))[None]
        return x, y


@torch.no_grad()
def eval_dice(model, loader, device) -> float:
    model.eval()
    dices = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = torch.sigmoid(model(x))
        # also TTA hflip average for more stable val? skip for speed
        inter = (p * y).sum((2, 3))
        den = p.sum((2, 3)) + y.sum((2, 3)) + 1e-6
        dices.append((2 * inter / den).mean().item())
    return float(np.mean(dices)) if dices else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=Path, default=ROOT / "data/raw/umud/fasc_imgs_v1")
    ap.add_argument("--mask-dir", type=Path, default=ROOT / "data/raw/umud/fasc_masks_v1")
    ap.add_argument("--init", type=Path, default=ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/phase3/chase018/seg_ckpts")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cldice-w", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = discover_pairs(args.img_dir, args.mask_dir)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"pairs={len(pairs)} img_dir={args.img_dir} mask_dir={args.mask_dir}", flush=True)
    if len(pairs) < 20:
        # print a few samples for debug
        print("sample imgs", list(args.img_dir.rglob('*'))[:10], flush=True)
        print("sample masks", list(args.mask_dir.rglob('*'))[:10], flush=True)
        raise SystemExit("too few pairs")

    print("computing dHash groups …", flush=True)
    hs = []
    for _, ip, _ in pairs:
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        hs.append(dhash(g))
    groups = groups_from_hashes(hs)
    uniq = sorted(set(groups.tolist()))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * args.val_frac)))
    val_g = set(uniq[:n_val])
    train_pairs = [p for p, g in zip(pairs, groups) if g not in val_g]
    val_pairs = [p for p, g in zip(pairs, groups) if g in val_g]
    print(f"groups={len(uniq)} train={len(train_pairs)} val={len(val_pairs)}", flush=True)

    train_ds = FascDS(train_pairs, augment=True)
    val_ds = FascDS(val_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = args.device
    model, size = load_ckpt(args.init, device)
    print("init", args.init, "size", size, "device", device, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    hist = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            loss = dice_loss(probs, y) + F.binary_cross_entropy_with_logits(logits, y) + args.cldice_w * soft_cldice(probs, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        vd = eval_dice(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": vd, "lr": float(sched.get_last_lr()[0])}
        hist.append(row)
        print(json.dumps(row), flush=True)
        if vd > best:
            best = vd
            ckpt = {
                "encoder": "resnet34",
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "size": list(size),
                "val_dice": best,
                "epoch": epoch,
                "init": str(args.init),
                "note": "chase018 fasc FT Dice+BCE+clDice dHash split",
            }
            out = args.out_dir / "fasc_resnet34_384x576_ft_best.pt"
            torch.save(ckpt, out)
            print("saved", out, "val_dice", best, flush=True)
        (args.out_dir / "fasc_ft_train.json").write_text(json.dumps({"best": best, "hist": hist}, indent=2))

    print(f"done in {time.time()-t0:.0f}s best_val_dice={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
