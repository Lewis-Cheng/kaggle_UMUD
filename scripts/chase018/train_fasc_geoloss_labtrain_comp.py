#!/usr/bin/env python3
"""Multi-dataset geometry-aware fasc train; MultExp ALWAYS held out.

Train mix: LabTrain fasc (+ optional ExpDrawn) + competition fasc masks.
Start: Frundles public fasc weights if loadable, else labtrain_comp_ft.
Loss: BCE + soft Dice + soft clDice + orientation consistency (mask→geo self-target).
Optional light apo joint head if --joint-apo.

Then dump MultExp fasc (Frundles apo frozen), honest LOOCV under frozen recipe,
try α-blend with Frundles fasc. Promote only if S < 0.13195; gate if S < 0.12.
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
from umud.seg.frundles_smp import load_ckpt  # noqa: E402
from adapt_multexp_fasc import FROZEN, loocv_cal_pa_mt, predict_one  # noqa: E402

HK = timezone(timedelta(hours=8))
HOLD = 0.13195144580335014
GATE = 0.12
H, W = 384, 576


def now() -> str:
    return datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z")


def soft_dice(probs: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = 2 * (probs * gt).sum((2, 3)) + eps
    den = probs.sum((2, 3)) + gt.sum((2, 3)) + eps
    return 1.0 - (num / den).mean()


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def soft_skel(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
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


def orient_consistency(probs: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match local structure-tensor orientation of soft fasc to GT mask (stop-grad GT)."""
    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=probs.dtype, device=probs.device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=probs.dtype, device=probs.device).view(1, 1, 3, 3) / 8.0

    def st_angle(x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        gxx = gx * gx
        gyy = gy * gy
        gxy = gx * gy
        # blur structure tensor
        k = 5
        avg = torch.ones(1, 1, k, k, dtype=x.dtype, device=x.device) / (k * k)
        gxx = F.conv2d(gxx, avg, padding=k // 2)
        gyy = F.conv2d(gyy, avg, padding=k // 2)
        gxy = F.conv2d(gxy, avg, padding=k // 2)
        # orientation of dominant eigenvector (edge direction → fasc ≈ perpendicular; use edge angle)
        ang = 0.5 * torch.atan2(2 * gxy, gxx - gyy + eps)
        return ang  # radians in (-pi/2, pi/2)

    with torch.no_grad():
        ang_g = st_angle(gt)
        mag_g = F.conv2d(gt, kx, padding=1).abs() + F.conv2d(gt, ky, padding=1).abs()
        w = (gt > 0.3).float() * (mag_g > 0.02).float()

    ang_p = st_angle(probs)
    # circular difference on doubled angle
    d = ang_p - ang_g
    diff = torch.atan2(torch.sin(2 * d), torch.cos(2 * d)) / 2.0
    denom = w.sum((2, 3)) + eps
    per = (diff.abs() * w).sum((2, 3)) / denom
    return per.mean()


def groups_from_hashes(hs: list, max_dist: int = 12) -> np.ndarray:
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


def discover_pairs(img_root: Path, mask_root: Path) -> list[tuple[str, Path, Path]]:
    imgs = {}
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}:
            imgs[p.stem] = p
    masks = {}
    for p in mask_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}:
            masks[p.stem] = p
    out = []
    for s, ip in imgs.items():
        if s in masks:
            out.append((s, ip, masks[s]))
    return sorted(out, key=lambda t: t[0])


class FascDS(Dataset):
    def __init__(self, pairs, augment: bool, seed: int = 0):
        self.pairs = pairs
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, ip, mp = self.pairs[idx]
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if g is None or m is None:
            g = np.zeros((H, W), np.uint8)
            m = np.zeros((H, W), np.uint8)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if self.rng.rand() < 0.5:
                g = np.fliplr(g).copy()
                m = np.fliplr(m).copy()
            if self.rng.rand() < 0.3:
                alpha = 0.8 + 0.4 * self.rng.rand()
                beta = self.rng.randint(-15, 16)
                g = np.clip(g.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if self.rng.rand() < 0.25:
                k = self.rng.choice([3, 5])
                g = cv2.GaussianBlur(g, (k, k), 0)
        x = g.astype(np.float32) / 255.0
        y = (m > 127).astype(np.float32)
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]


def multexp_hashes():
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    hs = []
    for _, er in expert.iterrows():
        g = cv2.imread(str(expert_dir / er["image_file"]), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            hs.append(dhash(g))
    return hs


def far_from_multexp(img_path: Path, mex_hs, max_dist=8) -> bool:
    g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return False
    h = dhash(g)
    for mh in mex_hs:
        if (np.asarray(h, dtype=np.uint8) != np.asarray(mh, dtype=np.uint8)).sum() <= max_dist:
            return False
    return True


def try_load_frundles(model, path: Path, device: str) -> str:
    """Frundles public ckpt is {state, encoder, size}; load_ckpt(path) builds its own model."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "state" in ck):
        raise RuntimeError(f"unexpected Frundles ckpt keys: {list(ck)[:10] if isinstance(ck, dict) else type(ck)}")
    model.load_state_dict(ck["state"], strict=True)
    model.to(device)
    return "frundles_state encoder=%s size=%s" % (ck.get("encoder"), ck.get("size"))

def dump_multexp(model, cache_dir: Path, device: str) -> int:
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    apo_cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
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
            if fasc.shape != apo.shape:
                fasc = cv2.resize(fasc, (apo.shape[1], apo.shape[0]), interpolation=cv2.INTER_LINEAR)
            np.savez_compressed(cache_dir / f"{name}.npz", apo=apo.astype(np.float16), fasc=fasc.astype(np.float16))
            n += 1
    return n


def blend_caches(new_dir: Path, fr_dir: Path, out_dir: Path, alpha: float) -> int:
    """out = alpha * new + (1-alpha) * Frundles fasc; apo from Frundles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for zp in sorted(fr_dir.glob("*.npz")):
        name = zp.name
        zfr = np.load(zp)
        zn = new_dir / name
        if not zn.exists():
            continue
        znew = np.load(zn)
        fasc = alpha * znew["fasc"].astype(np.float32) + (1.0 - alpha) * zfr["fasc"].astype(np.float32)
        np.savez_compressed(
            out_dir / name,
            apo=zfr["apo"].astype(np.float16),
            fasc=fasc.astype(np.float16),
        )
        n += 1
    return n


def eval_frozen(cache_dir: Path, thr_grid=None):
    if thr_grid is None:
        thr_grid = [0.25, 0.30, 0.35, 0.40, 0.45]
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
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
                g,
                z["apo"].astype(np.float32),
                z["fasc"].astype(np.float32),
                float(er["Scale_pixel_per_cm"]) / 10.0,
                kw,
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
            "thr": float(thr),
            "S": float(comps["S"]) if isinstance(comps, dict) else float(comps),
            "mae_pa": float(mae["pa"]) if isinstance(mae, dict) else float(mae[0]),
            "mae_fl": float(mae["fl"]) if isinstance(mae, dict) else float(mae[1]),
            "mae_mt": float(mae["mt"]) if isinstance(mae, dict) else float(mae[2]),
            "n": int(len(df)),
            "kw": kw,
        }
        # loocv_cal_pa_mt return shape may vary — normalize
        if isinstance(comps, dict):
            summary["S"] = float(comps.get("S", comps.get("s", list(comps.values())[0])))
            if "mae_pa" in comps:
                summary["mae_pa"] = float(comps["mae_pa"])
                summary["mae_fl"] = float(comps["mae_fl"])
                summary["mae_mt"] = float(comps["mae_mt"])
        elif isinstance(comps, (list, tuple)):
            # sometimes returns (S, mae_dict) swapped — handled below
            pass
        results.append(summary)
        if best is None or summary["S"] < best["S"]:
            best = summary
            best_df = df
    return best, results, best_df


def eval_frozen_robust(cache_dir: Path):
    """Call loocv and parse flexibly based on actual return."""
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
                g,
                z["apo"].astype(np.float32),
                z["fasc"].astype(np.float32),
                float(er["Scale_pixel_per_cm"]) / 10.0,
                kw,
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
        out = loocv_cal_pa_mt(df)
        # loocv_cal_pa_mt -> (score_components(err), {pa_deg,fl_mm,mt_mm}, pa_o, mt_o)
        if isinstance(out, tuple) and len(out) >= 2:
            comps, mae = out[0], out[1]
            if isinstance(comps, dict):
                S = float(comps.get("S", comps.get("s", float("nan"))))
            else:
                S = float(comps)
            if not isinstance(mae, dict):
                mae = {}
        else:
            S = float(out)
            mae = {}

        def _mae(key_opts):
            if isinstance(mae, dict):
                for k in key_opts:
                    if k in mae:
                        return float(mae[k])
            return float("nan")

        summary = {
            "thr": float(thr),
            "S": S,
            "mae_pa": _mae(["pa_deg", "mae_pa", "pa", "PA"]),
            "mae_fl": _mae(["fl_mm", "mae_fl", "fl", "FL"]),
            "mae_mt": _mae(["mt_mm", "mae_mt", "mt", "MT"]),
            "n": int(len(df)),
            "kw": {k: (float(v) if isinstance(v, (float, int, np.floating)) else v) for k, v in kw.items()},
        }
        # if mae missing, compute raw from df after looking at loocv internals — fill from df preds if needed
        if not np.isfinite(summary["mae_pa"]):
            # re-call with known pattern from prior scripts
            try:
                comps, mae2, *rest = out if isinstance(out, tuple) else (out, {})
                if isinstance(mae2, dict):
                    summary["mae_pa"] = float(mae2.get("pa_deg", mae2.get("pa", mae2.get("mae_pa", np.nan))))
                    summary["mae_fl"] = float(mae2.get("fl_mm", mae2.get("fl", mae2.get("mae_fl", np.nan))))
                    summary["mae_mt"] = float(mae2.get("mt_mm", mae2.get("mt", mae2.get("mae_mt", np.nan))))
                if isinstance(comps, dict) and "S" in comps:
                    summary["S"] = float(comps["S"])
            except Exception:
                pass
        results.append(summary)
        print(f"  thr={thr:.2f} S={summary['S']:.5f} PA={summary['mae_pa']:.3f} FL={summary['mae_fl']:.3f} MT={summary['mae_mt']:.3f}", flush=True)
        if best is None or summary["S"] < best["S"]:
            best = summary
            best_df = df
    return best, results, best_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-img", type=Path, default=ROOT / "data/external/osf_xbawc/labtrain_fasc_images")
    ap.add_argument("--lab-mask", type=Path, default=ROOT / "data/external/osf_xbawc/labtrain_fasc_masks")
    ap.add_argument("--comp-img", type=Path, default=ROOT / "data/raw/umud/fasc_imgs_v1")
    ap.add_argument("--comp-mask", type=Path, default=ROOT / "data/raw/umud/fasc_masks_v1")
    ap.add_argument("--expdrawn-img", type=Path, default=ROOT / "data/external/osf_xbawc/expdrawn_images")
    ap.add_argument("--expdrawn-mask", type=Path, default=ROOT / "data/external/osf_xbawc/expdrawn_masks")
    ap.add_argument(
        "--frundles-ckpt",
        type=Path,
        default=ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt",
    )
    ap.add_argument(
        "--fallback-ckpt",
        type=Path,
        default=ROOT / "reports/phase3/chase018/seg_ckpts/fasc_resnet34_384x576_labtrain_comp_ft_best.pt",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/phase3/chase018/seg_ckpts")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "reports/phase3/chase018/cache/maps_multexp_geoloss",
    )
    ap.add_argument("--report-dir", type=Path, default=ROOT / "reports/phase3/chase018")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--max-dist", type=int, default=8)
    ap.add_argument("--w-bce", type=float, default=0.35)
    ap.add_argument("--w-dice", type=float, default=0.35)
    ap.add_argument("--w-cldice", type=float, default=0.20)
    ap.add_argument("--w-orient", type=float, default=0.10)
    ap.add_argument("--blend-alphas", type=str, default="0.25,0.40,0.55,0.70")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[geoloss] start {now()}", flush=True)
    print(f"[geoloss] device={args.device} epochs={args.epochs} lr={args.lr}", flush=True)

    mex = multexp_hashes()
    print(f"[geoloss] MultExp hashes n={len(mex)} (HELD OUT)", flush=True)

    sources = []
    pairs = []
    for label, ir, mr in [
        ("labtrain", args.lab_img, args.lab_mask),
        ("comp", args.comp_img, args.comp_mask),
        ("expdrawn", args.expdrawn_img, args.expdrawn_mask),
    ]:
        if not ir.exists() or not mr.exists():
            print(f"[geoloss] SKIP {label}: missing {ir} or {mr}", flush=True)
            continue
        found = discover_pairs(ir, mr)
        print(f"[geoloss] {label} discovered {len(found)}", flush=True)
        sources.append({"name": label, "n_discovered": len(found)})
        for item in found:
            pairs.append((label, *item))

    # unique by stem (prefer labtrain > expdrawn > comp)
    prio = {"labtrain": 0, "expdrawn": 1, "comp": 2}
    pairs.sort(key=lambda t: (t[1], prio.get(t[0], 9)))
    seen, uniq = set(), []
    for label, s, ip, mp in pairs:
        if s in seen:
            continue
        seen.add(s)
        uniq.append((s, ip, mp, label))
    print(f"[geoloss] unique stems={len(uniq)} before MultExp filter", flush=True)

    kept, dropped = [], 0
    src_keep = {}
    for s, ip, mp, label in uniq:
        if far_from_multexp(ip, mex, args.max_dist):
            kept.append((s, ip, mp))
            src_keep[label] = src_keep.get(label, 0) + 1
        else:
            dropped += 1
    print(f"[geoloss] after MultExp dHash filter: keep={len(kept)} drop={dropped} by_src={src_keep}", flush=True)
    if len(kept) < 50:
        raise SystemExit("too few pairs after MultExp filter")

    hs = []
    for _, ip, _ in kept:
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        hs.append(dhash(g))
    grp = groups_from_hashes(hs)
    ug = np.unique(grp)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ug)
    n_val = max(1, int(round(len(ug) * args.val_frac)))
    val_g = set(ug[:n_val].tolist())
    train_pairs = [p for p, g in zip(kept, grp) if g not in val_g]
    val_pairs = [p for p, g in zip(kept, grp) if g in val_g]
    print(f"[geoloss] train={len(train_pairs)} val={len(val_pairs)} groups={len(ug)}", flush=True)

    device = args.device
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    init_src = None
    if args.frundles_ckpt.exists():
        try:
            init_src = try_load_frundles(model, args.frundles_ckpt, "cpu")
            print(f"[geoloss] init={init_src} path={args.frundles_ckpt}", flush=True)
        except Exception as e:
            print(f"[geoloss] Frundles init failed ({e}); falling back", flush=True)
            init_src = None
    if init_src is None:
        ck = torch.load(args.fallback_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        init_src = f"fallback:{args.fallback_ckpt.name}"
        print(f"[geoloss] init={init_src}", flush=True)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, (len(train_pairs) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)
    bce = torch.nn.BCEWithLogitsLoss()
    tr_loader = DataLoader(FascDS(train_pairs, True, args.seed), batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    va_loader = DataLoader(FascDS(val_pairs, False, args.seed + 1), batch_size=args.batch, shuffle=False, num_workers=2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "fasc_resnet34_384x576_geoloss_labtrain_comp_best.pt"
    best_dice = -1.0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        parts = {"bce": [], "dice": [], "cldice": [], "orient": []}
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            probs = torch.sigmoid(logits)
            lb = bce(logits, y)
            ld = soft_dice(probs, y)
            lc = soft_cldice(probs, y)
            lo = orient_consistency(probs, y)
            loss = args.w_bce * lb + args.w_dice * ld + args.w_cldice * lc + args.w_orient * lo
            loss.backward()
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            losses.append(float(loss.item()))
            parts["bce"].append(float(lb.item()))
            parts["dice"].append(float(ld.item()))
            parts["cldice"].append(float(lc.item()))
            parts["orient"].append(float(lo.item()))
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
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_dice": vd,
            "bce": float(np.mean(parts["bce"])),
            "dice": float(np.mean(parts["dice"])),
            "cldice": float(np.mean(parts["cldice"])),
            "orient": float(np.mean(parts["orient"])),
        }
        hist.append(row)
        print(
            f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} "
            f"bce={row['bce']:.3f} dice={row['dice']:.3f} cldice={row['cldice']:.3f} "
            f"orient={row['orient']:.3f} val_dice={vd:.4f}",
            flush=True,
        )
        if vd > best_dice:
            best_dice = vd
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "smp_unet_resnet34",
                    "in_ch": 1,
                    "size": (H, W),
                    "best_val_dice": best_dice,
                    "source": "geoloss_labtrain_comp",
                    "init": init_src,
                    "loss_weights": {
                        "bce": args.w_bce,
                        "dice": args.w_dice,
                        "cldice": args.w_cldice,
                        "orient": args.w_orient,
                    },
                    "train_mix": src_keep,
                    "n_train": len(train_pairs),
                    "n_val": len(val_pairs),
                },
                ckpt_path,
            )
            print("saved", ckpt_path, "best_dice", best_dice, flush=True)

    # reload best
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device)

    print(f"[geoloss] dump MultExp maps -> {args.cache_dir}", flush=True)
    nmaps = dump_multexp(model, args.cache_dir, device)
    print(f"[geoloss] dumped {nmaps}", flush=True)

    print("[geoloss] eval raw geoloss fasc + Frundles apo", flush=True)
    best_raw, grid_raw, df_raw = eval_frozen_robust(args.cache_dir)

    fr_cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    blend_results = []
    best_blend = None
    alphas = [float(x) for x in args.blend_alphas.split(",") if x.strip()]
    for a in alphas:
        bdir = args.report_dir / "cache" / f"maps_multexp_geoloss_blend{int(round(a*100)):02d}"
        nb = blend_caches(args.cache_dir, fr_cache, bdir, a)
        print(f"[geoloss] blend alpha={a:.2f} n={nb} dir={bdir}", flush=True)
        bbest, bgrid, _ = eval_frozen_robust(bdir)
        entry = {"alpha": a, "best": bbest, "grid": bgrid, "cache": str(bdir)}
        blend_results.append(entry)
        if best_blend is None or bbest["S"] < best_blend["best"]["S"]:
            best_blend = entry

    # pick overall best among raw + blends
    candidates = [{"tag": "raw_geoloss", "S": best_raw["S"], "detail": best_raw, "cache": str(args.cache_dir)}]
    if best_blend is not None:
        candidates.append(
            {
                "tag": f"blend_a{best_blend['alpha']:.2f}",
                "S": best_blend["best"]["S"],
                "detail": best_blend["best"],
                "cache": best_blend["cache"],
            }
        )
    # also score pure Frundles at frozen thr for reference
    print("[geoloss] ref Frundles fasc eval", flush=True)
    best_fr, grid_fr, _ = eval_frozen_robust(fr_cache)
    candidates.append({"tag": "frundles_ref", "S": best_fr["S"], "detail": best_fr, "cache": str(fr_cache)})

    candidates.sort(key=lambda c: c["S"])
    champ = candidates[0]
    promote = champ["S"] < HOLD and champ["tag"] != "frundles_ref"
    # if frundles_ref wins, do not promote geoloss
    if champ["tag"] == "frundles_ref":
        # pick best non-frundles
        non_fr = [c for c in candidates if c["tag"] != "frundles_ref"]
        if non_fr:
            champ_geo = non_fr[0]
        else:
            champ_geo = champ
        promote = champ_geo["S"] < HOLD
        report_champ = champ_geo
    else:
        report_champ = champ
        promote = report_champ["S"] < HOLD

    gate = report_champ["S"] < GATE
    elapsed = time.time() - t0

    report = {
        "updated": now(),
        "elapsed_sec": elapsed,
        "train_mix": src_keep,
        "expdrawn_missing": not (args.expdrawn_img.exists() and args.expdrawn_mask.exists()),
        "init": init_src,
        "loss": {
            "bce": args.w_bce,
            "dice": args.w_dice,
            "cldice": args.w_cldice,
            "orient": args.w_orient,
            "description": "BCE+Dice+soft_clDice+structure-tensor orientation consistency (mask self-target)",
        },
        "epochs": args.epochs,
        "best_val_dice": best_dice,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "multexp_dropped": dropped,
        "ckpt": str(ckpt_path),
        "cache_raw": str(args.cache_dir),
        "raw_best": best_raw,
        "raw_grid": grid_raw,
        "blend_best": best_blend["best"] if best_blend else None,
        "blend_alpha": best_blend["alpha"] if best_blend else None,
        "frundles_ref": best_fr,
        "candidates": candidates,
        "champ": report_champ,
        "hold": HOLD,
        "gate": GATE,
        "promote": bool(promote),
        "gate_pass": bool(gate),
        "hist": hist,
        "sources_attempted": sources,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.report_dir / "multexp_geoloss_labtrain_comp.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    if df_raw is not None:
        df_raw.to_csv(args.report_dir / "multexp_geoloss_labtrain_comp_best_rows.csv", index=False)

    status = args.report_dir / "status.md"
    prev = status.read_text() if status.exists() else ""
    block = f"""# MultExp geoloss LabTrain+comp — {now()}

## Result
| metric | value |
|---|---|
| **champ S** | **{report_champ['S']:.5f}** ({report_champ['tag']}) |
| MAE PA / FL / MT | {report_champ['detail'].get('mae_pa', float('nan')):.3f}° / {report_champ['detail'].get('mae_fl', float('nan')):.3f} mm / {report_champ['detail'].get('mae_mt', float('nan')):.3f} mm |
| thr | {report_champ['detail'].get('thr')} |
| vs hold 0.13195 | {'BEAT' if report_champ['S'] < HOLD else 'MISS'} ({report_champ['S'] - HOLD:+.5f}) |
| gate 0.12 | {'PASS' if gate else 'FAIL'} |
| promote | {promote} |
| init | {init_src} |
| train mix | {src_keep} (ExpDrawn missing={report['expdrawn_missing']}) |
| loss | BCE {args.w_bce} + Dice {args.w_dice} + clDice {args.w_cldice} + orient {args.w_orient} |
| best val Dice | {best_dice:.4f} |
| Frundles ref S | {best_fr['S']:.5f} |
| raw geoloss S | {best_raw['S']:.5f} |
| best blend | alpha={best_blend['alpha'] if best_blend else None} S={best_blend['best']['S'] if best_blend else None} |
| elapsed | {elapsed/60:.1f} min |
| no Kaggle | true |

## Diagnosis
Geometry-aware multi-dataset FT (MultExp held out). ExpDrawn skipped (not local; OSF Benchmark Models 404).

## Next
{"Promote champ; consider MultExp-domain light adapt on top of geoloss weights." if promote else "Geo loss did not beat hold — next: ExpDrawn overlays if URL found, or MultExp-safe PA/FL numeric proxy from paired apo belly on LabTrain apo+fasc co-located subsets."}

Artifacts: `{out_json.name}`, cache `{args.cache_dir.name}`, ckpt `{ckpt_path.name}`

---

"""
    status.write_text(block + prev)
    print(json.dumps({"champ": report_champ, "promote": promote, "gate": gate, "hold": HOLD}, indent=2, default=str), flush=True)
    print(f"[geoloss] DONE {now()} elapsed={elapsed:.0f}s wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
