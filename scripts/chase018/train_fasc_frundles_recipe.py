#!/usr/bin/env python3
"""Train fasc ResNet34 U-Net from ImageNet using Frundles train_seg.py recipe.

Key Frundles knobs we were missing in the first FT:
  * ImageNet init (not fine-tune of OSF-trained Frundles weights)
  * aspect-jitter random crop + gamma/brightness + speckle
  * mask resize INTER_AREA then thr>0.25 (preserves thin fascicles)
  * OneCycleLR, BCE + soft Dice, hard Dice@0.5 for selection
  * optional pos_weight for thin positives
dHash group split for leak-safety.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.geometry.host_faithful import dhash  # noqa: E402


def groups_from_hashes(hs, max_dist=12):
    parent = list(range(len(hs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    arr = np.array(hs, dtype=np.uint8)
    for i in range(len(arr)):
        for j in np.nonzero((arr[i + 1 :] != arr[i]).sum(1) <= max_dist)[0] + i + 1:
            parent[find(int(j))] = find(i)
    return np.array([find(i) for i in range(len(hs))])


def discover_pairs(img_dir: Path, mask_dir: Path):
    imgs = sorted(list(img_dir.glob("*.tif")) + list(img_dir.glob("*.png")))
    pairs = []
    for ip in imgs:
        mp = mask_dir / ip.name
        if not mp.exists():
            mp = mask_dir / (ip.stem + ".png")
        if mp.exists():
            pairs.append((ip.stem, ip, mp))
    return pairs


def load_xy(ip: Path, mp: Path):
    x = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
    y = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if x is None or y is None:
        raise RuntimeError(f"bad {ip}")
    if y.shape[:2] != x.shape[:2]:
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_NEAREST)
    y = (y > 127).astype(np.uint8)
    return x, y


def augment(x, y, rng, out_hw, crop_min=0.6, rot_deg=8.0, rot_prob=0.5):
    H, W = x.shape
    oh, ow = out_hw
    crop_min = float(max(0.4, min(1.0, crop_min)))
    rot_deg = float(max(0.0, min(8.0, rot_deg)))
    ch, cw = int(H * rng.uniform(crop_min, 1.0)), int(W * rng.uniform(crop_min, 1.0))
    y0, x0 = int(rng.integers(0, H - ch + 1)), int(rng.integers(0, W - cw + 1))
    x = x[y0 : y0 + ch, x0 : x0 + cw]
    y = y[y0 : y0 + ch, x0 : x0 + cw]
    if rng.random() < rot_prob:
        a = float(rng.uniform(-rot_deg, rot_deg))
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), a, 1.0)
        x = cv2.warpAffine(x, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        y = cv2.warpAffine(y, M, (cw, ch), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    x = cv2.resize(x, (ow, oh), interpolation=cv2.INTER_AREA)
    y = cv2.resize(y.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA) > 0.25
    if rng.random() < 0.5:
        x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
    x = x.astype(np.float32) / 255
    x = np.clip((x ** rng.uniform(0.6, 1.6)) * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1), 0, 1)
    if rng.random() < 0.3:
        x = np.clip(x * rng.normal(1, 0.1, x.shape), 0, 1)
    return x.astype(np.float32), y.astype(np.float32)


def prep_eval(x, y, out_hw):
    oh, ow = out_hw
    xx = cv2.resize(x, (ow, oh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    yy = (cv2.resize(y.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA) > 0.25).astype(np.float32)
    return xx, yy


def dice_loss(logit, y, eps=1.0):
    p = torch.sigmoid(logit)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/phase3/chase018/seg_ckpts")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--size", type=int, nargs=2, default=[384, 576])
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="fasc_resnet34_384x576_scratch")
    ap.add_argument("--crop-min", type=float, default=0.6, help="random crop scale low (high=1.0)")
    ap.add_argument("--rot-deg", type=float, default=8.0, help="tiny rotation half-range, capped at 8")
    ap.add_argument("--rot-prob", type=float, default=0.5, help="probability of applying rotation")
    ap.add_argument(
        "--init",
        type=Path,
        default=None,
        help="optional ckpt to continue from (state dict); else ImageNet init",
    )
    args = ap.parse_args()

    pairs = discover_pairs(args.img_dir, args.mask_dir)
    print(f"pairs={len(pairs)}", flush=True)
    print("loading images into RAM …", flush=True)
    Xs, Ys, hs = [], [], []
    for _, ip, mp in pairs:
        x, y = load_xy(ip, mp)
        Xs.append(x)
        Ys.append(y)
        hs.append(dhash(x))
    groups = groups_from_hashes(hs)
    uniq = sorted(set(groups.tolist()))
    rng_split = np.random.default_rng(args.seed)
    rng_split.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * args.val_frac)))
    val_g = set(uniq[:n_val])
    tr = [i for i, g in enumerate(groups) if g not in val_g]
    va = [i for i, g in enumerate(groups) if g in val_g]
    print(f"groups={len(uniq)} train={len(tr)} val={len(va)}", flush=True)

    out_hw = tuple(args.size)
    print("prep val tensors …", flush=True)
    Xv, Yv = zip(*[prep_eval(Xs[i], Ys[i], out_hw) for i in va])
    Xv = torch.from_numpy(np.stack(Xv))[:, None]
    Yv = torch.from_numpy(np.stack(Yv))[:, None]

    device = torch.device(args.device)
    if args.init is not None and args.init.exists():
        print(f"continue from {args.init}", flush=True)
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        enc = ck.get("encoder", "resnet34")
        model = smp.Unet(enc, encoder_weights=None, in_channels=1, classes=1).to(device)
        model.load_state_dict(ck["state"])
        if "val_dice" in ck:
            print(f"prior val_dice/hard={ck.get('val_dice')}", flush=True)
    else:
        if args.init is not None:
            print(f"WARN init missing {args.init}; falling back to ImageNet", flush=True)
        model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(tr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1
    )
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    hist = []
    pw = torch.tensor(args.pos_weight, device=device)
    t0_all = time.time()
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tl = 0.0
        perm = rng.permutation(tr)
        for b in range(steps_per_epoch):
            idx = perm[b * args.bs : (b + 1) * args.bs]
            xb, yb = zip(*[augment(Xs[i], Ys[i], rng, out_hw, args.crop_min, args.rot_deg, args.rot_prob) for i in idx])
            xb = torch.from_numpy(np.stack(xb))[:, None].to(device)
            yb = torch.from_numpy(np.stack(yb))[:, None].to(device)
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw) + dice_loss(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss.item())
        model.eval()
        dices = []
        with torch.no_grad():
            for b in range(0, len(Xv), 16):
                p = torch.sigmoid(model(Xv[b : b + 16].to(device))).cpu() > 0.5
                y = Yv[b : b + 16] > 0.5
                inter = (p & y).sum((1, 2, 3)).float()
                dices += ((2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).tolist()
        vd = float(np.mean(dices))
        row = {"epoch": ep, "train_loss": tl / steps_per_epoch, "val_hard_dice": vd, "best": max(best, vd)}
        hist.append(row)
        print(json.dumps(row) + f" ({time.time()-t0:.0f}s)", flush=True)
        if vd > best:
            best = vd
            ckpt = {
                "encoder": "resnet34",
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "size": list(out_hw),
                "val_dice": best,
                "epoch": ep,
                "note": ("chase018 Frundles-recipe fasc; init=" + (str(args.init) if args.init else "imagenet")
                         + f"; crop_min={args.crop_min} rot_deg={args.rot_deg} rot_prob={args.rot_prob}"),
            }
            out = args.out_dir / f"{args.tag}_best.pt"
            torch.save(ckpt, out)
            print("saved", out, "val_hard_dice", best, flush=True)
        (args.out_dir / f"{args.tag}_train.json").write_text(json.dumps({"best": best, "hist": hist}, indent=2))
    print(f"done in {time.time()-t0_all:.0f}s best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
