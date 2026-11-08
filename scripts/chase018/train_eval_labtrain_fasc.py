#!/usr/bin/env python3
"""Train fasc U-Net on OSF LabTrain (DL_Track Example Dataset), dump MultExp maps, eval.

- ImageNet-init SMP ResNet34 U-Net (NOT Frundles weights) — outside-family
- Train on LabTrain fasc imgs/masks only; MultExp never in train
- Keep Frundles apo cache; write maps_multexp_labtrain_fasc
- Frozen recipe honest LOOCV vs hold 0.13195 / gate 0.12
No Kaggle. No attention.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from umud.geometry.host_faithful import dhash  # noqa: E402
from umud.seg.frundles_smp import load_ckpt, load_pair, predict_prob  # noqa: E402
from adapt_multexp_fasc import FROZEN, loocv_cal_pa_mt, predict_one  # noqa: E402

HK = timezone(timedelta(hours=8))
HOLD = 0.13195144580335014
GATE = 0.12
H, W = 384, 576


def now():
    return datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z")


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
    imgs = sorted(
        list(img_dir.rglob("*.tif"))
        + list(img_dir.rglob("*.png"))
        + list(img_dir.rglob("*.jpg"))
    )
    pairs = []
    for ip in imgs:
        cands = [mask_dir / ip.name]
        cands += list(mask_dir.rglob(ip.name))
        mp = next((c for c in cands if c.exists() and c.is_file()), None)
        if mp is not None:
            pairs.append((ip.stem, ip, mp))
    # unique
    seen, out = set(), []
    for s, ip, mp in pairs:
        if s in seen:
            continue
        seen.add(s)
        out.append((s, ip, mp))
    return out


def soft_dice(probs, gt, eps=1e-6):
    num = 2 * (probs * gt).sum((2, 3)) + eps
    den = probs.sum((2, 3)) + gt.sum((2, 3)) + eps
    return 1.0 - (num / den).mean()


class FascDS(Dataset):
    def __init__(self, pairs, train=True, seed=0):
        self.pairs = pairs
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        _, ip, mp = self.pairs[i]
        x = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if x is None or y is None:
            raise RuntimeError(str(ip))
        if y.shape[:2] != x.shape[:2]:
            y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.train:
            H0, W0 = x.shape
            ch = int(H0 * self.rng.uniform(0.65, 1.0))
            cw = int(W0 * self.rng.uniform(0.65, 1.0))
            y0 = int(self.rng.integers(0, max(1, H0 - ch + 1)))
            x0 = int(self.rng.integers(0, max(1, W0 - cw + 1)))
            x = x[y0 : y0 + ch, x0 : x0 + cw]
            y = y[y0 : y0 + ch, x0 : x0 + cw]
            if self.rng.random() < 0.5:
                x = np.fliplr(x).copy()
                y = np.fliplr(y).copy()
            if self.rng.random() < 0.3:
                g = float(self.rng.uniform(0.7, 1.3))
                x = np.clip((x.astype(np.float32) / 255.0) ** g * 255.0, 0, 255).astype(np.uint8)
        x = cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA)
        y = cv2.resize((y > 127).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        y = (y > 0.25).astype(np.float32)
        xt = torch.from_numpy(x.astype(np.float32) / 255.0)[None]
        yt = torch.from_numpy(y)[None]
        return xt, yt


def train(args):
    pairs = discover_pairs(args.img_dir, args.mask_dir)
    print(f"[train] pairs={len(pairs)} img={args.img_dir} mask={args.mask_dir}", flush=True)
    if len(pairs) < 10:
        raise SystemExit(f"too few pairs: {len(pairs)}")
    # dHash group split
    hs = []
    for _, ip, _ in pairs:
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        hs.append(dhash(g))
    grp = groups_from_hashes(hs)
    uniq = np.unique(grp)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * args.val_frac)))
    val_g = set(uniq[:n_val].tolist())
    train_pairs = [p for p, g in zip(pairs, grp) if g not in val_g]
    val_pairs = [p for p, g in zip(pairs, grp) if g in val_g]
    print(f"[train] train={len(train_pairs)} val={len(val_pairs)} groups={len(uniq)}", flush=True)

    device = args.device
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=max(1, len(train_pairs) // args.batch)
    )
    bce = torch.nn.BCEWithLogitsLoss()
    tr_loader = DataLoader(FascDS(train_pairs, True, args.seed), batch_size=args.batch, shuffle=True, num_workers=2)
    va_loader = DataLoader(FascDS(val_pairs, False, args.seed + 1), batch_size=args.batch, shuffle=False, num_workers=2)

    best = -1.0
    hist = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "fasc_resnet34_384x576_labtrain_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            probs = torch.sigmoid(logits)
            loss = 0.5 * bce(logits, y) + 0.5 * soft_dice(probs, y)
            loss.backward()
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            losses.append(float(loss.item()))
        model.eval()
        dices = []
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                probs = torch.sigmoid(model(x))
                pred = (probs > 0.5).float()
                inter = (pred * y).sum((2, 3))
                den = pred.sum((2, 3)) + y.sum((2, 3)) + 1e-6
                dices.append((2 * inter / den).mean().item())
        vd = float(np.mean(dices)) if dices else 0.0
        hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_dice": vd})
        print(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} val_dice={vd:.4f}", flush=True)
        if vd > best:
            best = vd
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "smp_unet_resnet34",
                    "in_ch": 1,
                    "size": (H, W),
                    "best_val_dice": best,
                    "source": "osf_labtrain_dltrack_example",
                    "encoder_weights": "imagenet",
                },
                ckpt_path,
            )
            print("saved", ckpt_path, "best", best, flush=True)
    (args.out_dir / "fasc_labtrain_train.json").write_text(
        json.dumps({"best": best, "hist": hist, "n_train": len(train_pairs), "n_val": len(val_pairs)}, indent=2)
    )
    return ckpt_path, best


def dump_multexp(ckpt_path: Path, cache_dir: Path, device: str):
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    apo_cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # load fasc from labtrain ckpt; apo from Frundles via existing cache
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model = model.to(device).eval()
    n = 0
    with torch.no_grad():
        for _, er in expert.iterrows():
            name = er["image_file"]
            g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
            zsrc = apo_cache / f"{name}.npz"
            if g is None or not zsrc.exists():
                continue
            apo = np.load(zsrc)["apo"].astype(np.float32)
            x = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            t = torch.from_numpy(x)[None, None].to(device)
            fasc = torch.sigmoid(model(t))[0, 0].detach().cpu().numpy().astype(np.float32)
            # resize fasc to apo shape if needed
            if fasc.shape != apo.shape:
                fasc = cv2.resize(fasc, (apo.shape[1], apo.shape[0]), interpolation=cv2.INTER_LINEAR)
            np.savez_compressed(cache_dir / f"{name}.npz", apo=apo.astype(np.float16), fasc=fasc.astype(np.float16))
            n += 1
    print(f"[dump] wrote {n} maps -> {cache_dir}", flush=True)
    return n


def eval_frozen(cache_dir: Path):
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    thr_grid = [0.25, 0.30, 0.35, 0.40, 0.45]
    results = []
    best = None
    best_df = None
    for thr in thr_grid:
        kw = dict(FROZEN)
        kw["fasc_thr"] = float(thr)
        rows = []
        for _, er in expert.iterrows():
            name = er["image_file"]
            g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
            zp = cache_dir / f"{name}.npz"
            if g is None or not zp.exists():
                continue
            z = np.load(zp)
            pred = predict_one(
                g, z["apo"].astype(np.float32), z["fasc"].astype(np.float32), float(er["Scale_pixel_per_cm"]) / 10.0, kw
            )
            rows.append(
                dict(
                    image_file=name,
                    ok=pred["ok"],
                    pred_pa=pred["pred_pa"],
                    pred_fl=pred["pred_fl"],
                    pred_mt=pred["pred_mt"],
                    gt_pa=float(er["median_PA"]),
                    gt_fl=float(er["median_FL"]),
                    gt_mt=float(er["median_MT"]),
                )
            )
        df = pd.DataFrame(rows)
        comps, mae, *_ = loocv_cal_pa_mt(df)
        summary = {
            "thr": thr,
            "S": float(comps["S"]),
            "MAE": mae,
            "n": len(df),
            "gate": float(comps["S"]) < GATE,
            "beats_hold": float(comps["S"]) < HOLD,
        }
        results.append(summary)
        print(f"[eval] thr={thr} S={summary['S']:.5f} MAE={mae}", flush=True)
        if best is None or summary["S"] < best["S"]:
            best = summary
            best_df = df
    return results, best, best_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/phase3/chase018/seg_ckpts")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "reports/phase3/chase018/cache/maps_multexp_labtrain_fasc")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--ckpt", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.skip_train:
        ckpt = args.ckpt or (args.out_dir / "fasc_resnet34_384x576_labtrain_best.pt")
        best_dice = None
    else:
        ckpt, best_dice = train(args)
    dump_multexp(ckpt, args.cache_dir, args.device)
    results, best, best_df = eval_frozen(args.cache_dir)
    out = ROOT / "reports/phase3/chase018"
    if best_df is not None:
        best_df.to_csv(out / "multexp_nonfrundles_fasc_labtrain_best_rows.csv", index=False)
    payload = {
        "ts": now(),
        "hold": HOLD,
        "gate": GATE,
        "ckpt": str(ckpt),
        "best_val_dice": best_dice,
        "train_img": str(args.img_dir),
        "train_mask": str(args.mask_dir),
        "cache": str(args.cache_dir),
        "results": results,
        "best": best,
        "promoted": bool(best and best["S"] < HOLD),
        "gate_pass": bool(best and best["S"] < GATE),
        "elapsed_s": time.time() - t0,
        "note": "ImageNet-init LabTrain fasc + Frundles apo; MultExp never in train",
    }
    (out / "multexp_nonfrundles_fasc_labtrain.json").write_text(json.dumps(payload, indent=2))
    status = (
        f"# LabTrain fasc MultExp\nUpdated: {now()}\n\n"
        f"- Train: OSF LabTrain fasc (ImageNet ResNet34 U-Net), MultExp never in train\n"
        f"- Best MultExp S: **{best['S']:.5f}** thr={best['thr']} PA={best['MAE']['pa_deg']:.3f} "
        f"FL={best['MAE']['fl_mm']:.3f} MT={best['MAE']['mt_mm']:.3f}\n"
        f"- Hold 0.13195: {'BEAT' if payload['promoted'] else 'MISS'}; Gate 0.12: {'PASS' if payload['gate_pass'] else 'FAIL'}; no Kaggle\n"
        f"- ckpt: `{ckpt}` val_dice={best_dice}\n"
    )
    (out / "multexp_nonfrundles_fasc_status.md").write_text(status)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
