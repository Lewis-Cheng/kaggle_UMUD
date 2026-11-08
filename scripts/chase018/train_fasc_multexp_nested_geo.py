#!/usr/bin/env python3
"""Nested/group-safe MultExp fasc fine-tune with PA/FL geometry proxy.

Protocol (honest, no held-out leak):
  - Split MultExp into K group-folds (dHash groups → K folds, default 5).
  - For each fold: fine-tune Frundles fasc U-Net on TRAIN folds only, using a
    stop-grad geometry-constrained fasc target built from MultExp GT PA/FL/MT
    + Frundles apo belly (apo frozen / never trained).
  - Infer fasc maps for HELD-OUT images with that fold's adapted weights.
  - Apo maps stay Frundles cache unchanged.
  - Final honest LOOCV S (affine PA+MT, FL raw) under frozen geo recipe.

Start weights: data/external/frundles_weights/fasc_resnet34_384x576_f0.pt
NOT lab scratch_best. No Kaggle. No attention.

Promote only if S < 0.13195; gate pass iff S < 0.12.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from umud.geometry.host_faithful import (  # noqa: E402
    aponeuroses_rich,
    dhash,
    slope_deg,
)
from umud.seg.frundles_smp import load_ckpt  # noqa: E402
from adapt_multexp_fasc import (  # noqa: E402
    FROZEN,
    apo_belly_mask,
    loocv_cal_pa_mt,
    predict_one,
)

HK = timezone(timedelta(hours=8))
PRIOR_C = 0.13195144580335014
PRIOR_FIXED = 0.13293555901039555


def dice_loss(probs: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
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


def assign_folds(group_ids: np.ndarray, n_folds: int, seed: int = 0) -> np.ndarray:
    """Map each sample's group → fold id in 0..n_folds-1 (group-safe)."""
    uniq = sorted(set(int(g) for g in group_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    g2f = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([g2f[int(g)] for g in group_ids], dtype=int)


def deep_apo_slope_px(apo: np.ndarray, apo_thr: float = 0.60) -> float:
    """Median deep-apo slope in degrees (image coords, +y down)."""
    rich = aponeuroses_rich(apo, 1.0, 1.0, thr=apo_thr)
    if rich is None:
        return 0.0
    deep = rich[1]
    xs = np.linspace(deep["lo"] + 0.2 * (deep["hi"] - deep["lo"]), deep["hi"] - 0.2 * (deep["hi"] - deep["lo"]), 7)
    slopes = [float(slope_deg(deep["fit"]["c"], float(x))) for x in xs]
    return float(np.median(slopes)) if slopes else 0.0


def frundles_fasc_angle_hint(fasc: np.ndarray, belly: np.ndarray, thr: float = 0.35) -> float:
    """Rough absolute fascicle angle from Frundles map edges (deg, image)."""
    m = ((fasc > thr) & (belly > 0)).astype(np.uint8) * 255
    if m.sum() < 50:
        return float("nan")
    edges = cv2.Canny(m, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=12, maxLineGap=6)
    if lines is None:
        return float("nan")
    angs = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if abs(x2 - x1) + abs(y2 - y1) < 5:
            continue
        ang = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 90) % 180 - 90
        angs.append(ang)
    return float(np.median(angs)) if angs else float("nan")


def paint_geo_target(
    apo: np.ndarray,
    fasc_fr: np.ndarray,
    gt_pa: float,
    gt_fl: float,
    gt_mt: float,
    px_per_mm: float,
    *,
    blend: float = 0.55,
    apo_thr: float = 0.60,
) -> np.ndarray:
    """Stop-grad geometry-constrained fasc target (train folds only).

    Paints soft parallel fascicle stripes in the apo belly at absolute angle
    ≈ deep_apo_slope ± GT_PA (sign matched to Frundles fasc), with spacing
    tied to GT FL / sin(PA) thickness heuristic. Blends with Frundles fasc
    for stability. Leakage risk: GT PA/FL/MT enter the TARGET on train only;
    held-out images never contribute GT to that fold's FT or target build.
    """
    h, w = apo.shape
    belly = apo_belly_mask(apo, apo_thr, 2).astype(np.float32)
    deep_s = deep_apo_slope_px(apo, apo_thr)
    # pennation: absolute fasc angle ≈ deep_slope + signed PA
    hint = frundles_fasc_angle_hint(fasc_fr, belly)
    cand_p = deep_s + gt_pa
    cand_m = deep_s - gt_pa
    if np.isfinite(hint):
        abs_ang = cand_p if abs(((cand_p - hint + 90) % 180) - 90) <= abs(((cand_m - hint + 90) % 180) - 90) else cand_m
    else:
        abs_ang = cand_m  # MultExp often has fasc rising left→right with -PA convention; fallback
    # stripe period from FL geometry: along-fasc period ~ a few fascicles across belly
    # physical period ~ MT / tan(PA) projected; convert to map px via mm_per_map_px ≈ q * (full/map)
    # Use a robust proxy: period_px ≈ clamp( 0.35 * gt_fl / max(sin(pa),0.15) * px_per_mm * (w/512) , 8, 48)
    pa_rad = np.radians(np.clip(gt_pa, 5.0, 45.0))
    period_mm = float(np.clip(0.45 * gt_fl * np.sin(pa_rad), 2.0, 18.0))
    # map mm/px roughly: Frundles maps are 384x576; scale from original via px_per_mm
    # approximate: 1 map-px ≈ (orig_w/576) / px_per_mm mm — unknown orig; use px_per_mm directly with 0.7 shrink
    period_px = float(np.clip(period_mm * px_per_mm * 0.55, 8.0, 56.0))
    # stripe width ~ fasc thickness proxy
    width_px = float(np.clip(gt_mt * px_per_mm * 0.12, 1.5, 6.0))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = np.radians(abs_ang)
    # coordinate along normal to fascicles
    proj = xx * np.cos(theta + np.pi / 2) + yy * np.sin(theta + np.pi / 2)
    # soft stripes
    phase = (proj / period_px) * 2 * np.pi
    stripes = 0.5 * (1.0 + np.cos(phase))
    # sharpen with width
    stripes = np.clip((stripes - 0.35) / max(0.15, 0.5 - 0.02 * width_px), 0, 1)
    stripes = stripes * belly

    # keep Frundles mass as consistency anchor
    fr = fasc_fr.astype(np.float32)
    if fr.shape != (h, w):
        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
    fr_b = fr * (0.15 + 0.85 * belly)

    # geometry mass should not evaporate: boost where stripes fire
    geo = np.maximum(stripes * 0.85, stripes * fr_b)
    target = (1.0 - blend) * fr_b + blend * geo
    # light morph connect on target (numpy, stop-grad)
    m = (target > 0.20).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    fill = (m2 > 0) & (m == 0)
    if fill.any():
        blur = cv2.GaussianBlur(target, (0, 0), 1.0)
        target = np.where(fill, np.maximum(target, blur * 0.85), target)
    return np.clip(target, 0, 1).astype(np.float32)


class GeoProxyDS(Dataset):
    def __init__(self, items, size=(384, 576), augment=True):
        # items: list of dicts with gpath, apo, fasc_fr, gt_pa, gt_fl, gt_mt, px_per_mm, name
        self.items = items
        self.H, self.W = size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        g = cv2.imread(str(it["gpath"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"bad image {it['gpath']}")
        apo = it["apo"]
        fasc = it["fasc"]
        target = paint_geo_target(
            apo,
            fasc,
            it["gt_pa"],
            it["gt_fl"],
            it["gt_mt"],
            it["px_per_mm"],
            blend=it.get("blend", 0.55),
        )
        g_r = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if target.shape != (self.H, self.W):
            target = cv2.resize(target, (self.W, self.H), interpolation=cv2.INTER_AREA)
        if self.augment:
            if np.random.rand() < 0.5:
                g_r = g_r[:, ::-1].copy()
                target = target[:, ::-1].copy()
            if np.random.rand() < 0.45:
                alpha = 0.85 + 0.35 * np.random.rand()
                beta = (np.random.rand() - 0.5) * 0.12
                g_r = np.clip(g_r * alpha + beta, 0, 1)
            if np.random.rand() < 0.20:
                noise = np.random.randn(*g_r.shape).astype(np.float32) * 0.02
                g_r = np.clip(g_r + noise, 0, 1)
        x = torch.from_numpy(g_r[None]).float()
        y = torch.from_numpy(target[None]).float()
        return x, y


def load_items(expert_csv, expert_dir, cache_dir):
    expert = pd.read_csv(expert_csv)
    expert_dir = Path(expert_dir)
    cache_dir = Path(cache_dir)
    items = []
    hashes = []
    for _, er in expert.iterrows():
        name = er["image_file"]
        gpath = expert_dir / name
        zpath = cache_dir / f"{name}.npz"
        if not gpath.exists() or not zpath.exists():
            continue
        z = np.load(zpath)
        g = cv2.imread(str(gpath), cv2.IMREAD_GRAYSCALE)
        hs = dhash(g)
        items.append(
            dict(
                name=name,
                gpath=gpath,
                apo=z["apo"].astype(np.float32),
                fasc=z["fasc"].astype(np.float32),
                gt_pa=float(er["median_PA"]),
                gt_fl=float(er["median_FL"]),
                gt_mt=float(er["median_MT"]),
                px_per_mm=float(er["Scale_pixel_per_cm"]) / 10.0,
                ImageID=str(er["ImageID"]),
            )
        )
        hashes.append(hs)
    return items, hashes


@torch.no_grad()
def infer_heldout(model, held_items, apo_cache_lookup, out_dir, device, size):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    H, W = size
    for it in held_items:
        name = it["name"]
        g = cv2.imread(str(it["gpath"]), cv2.IMREAD_GRAYSCALE)
        apo = apo_cache_lookup[name]
        g_r = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = torch.from_numpy(g_r[None, None]).float().to(device)
        # TTA hflip
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits_f = model(x.flip(-1))
        if isinstance(logits_f, (tuple, list)):
            logits_f = logits_f[0]
        prob = ((torch.sigmoid(logits) + torch.sigmoid(logits_f).flip(-1)) / 2)[0, 0].cpu().numpy().astype(np.float32)
        if apo.shape != prob.shape:
            apo_r = cv2.resize(apo, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            apo_r = apo
        np.savez_compressed(
            out_dir / f"{name}.npz",
            apo=apo_r.astype(np.float16),
            fasc=prob.astype(np.float16),
        )


def eval_maps(cache_dir, expert_csv, expert_dir):
    expert = pd.read_csv(expert_csv)
    expert_dir = Path(expert_dir)
    cache_dir = Path(cache_dir)
    kw = dict(FROZEN)
    rows = []
    for _, er in expert.iterrows():
        name = er["image_file"]
        zpath = cache_dir / f"{name}.npz"
        if not zpath.exists():
            continue
        g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
        z = np.load(zpath)
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
    comps, mae, _, _ = loocv_cal_pa_mt(df)
    return {
        "S_loocv_cal_pa_mt": float(comps["S"]),
        "MAE_loocv": mae,
        "n": len(df),
        "n_ok": int(df["ok"].sum()),
        "gate_0.12": float(comps["S"]) < 0.12,
        "beats_0.13195": float(comps["S"]) < PRIOR_C,
        "beats_0.13294": float(comps["S"]) < PRIOR_FIXED,
    }, df


def train_fold(model, train_items, device, epochs, bs, lr, time_budget_s, blend, size):
    for it in train_items:
        it["blend"] = blend
    ds = GeoProxyDS(train_items, size=size, augment=True)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.time()
    hist = []
    model.train()
    for ep in range(1, epochs + 1):
        losses = []
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            loss = (
                0.45 * dice_loss(probs, y)
                + 0.35 * F.binary_cross_entropy(probs, y)
                + 0.20 * soft_cldice(probs, y)
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        mean_l = float(np.mean(losses)) if losses else 9.0
        hist.append({"ep": ep, "loss": mean_l, "sec": time.time() - t0})
        if time.time() - t0 >= time_budget_s:
            break
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--blend", type=float, default=0.55, help="geo paint blend vs Frundles")
    ap.add_argument("--time-limit-min", type=float, default=50.0)
    ap.add_argument("--fold-time-min", type=float, default=8.0)
    ap.add_argument("--hash-dist", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="nested_geo_proxy_v1")
    args = ap.parse_args()

    expert_csv = ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv"
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    ckpt_in = ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt"
    out_dir = ROOT / "reports/phase3/chase018"
    maps_out = out_dir / "cache" / f"maps_multexp_fasc_ft_{args.tag}"
    seg_dir = out_dir / "seg_ckpts"
    seg_dir.mkdir(parents=True, exist_ok=True)
    maps_out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} folds={args.n_folds} epochs={args.epochs} blend={args.blend}", flush=True)
    print(f"ckpt_in={ckpt_in}", flush=True)

    items, hashes = load_items(expert_csv, expert_dir, cache)
    print(f"n_items={len(items)}", flush=True)
    assert len(items) >= 10, "too few MultExp items"
    gids = groups_from_hashes(hashes, max_dist=args.hash_dist)
    folds = assign_folds(gids, args.n_folds, seed=args.seed)
    n_groups = len(set(int(g) for g in gids))
    print(f"n_groups={n_groups} fold_counts={np.bincount(folds, minlength=args.n_folds).tolist()}", flush=True)

    apo_lookup = {it["name"]: it["apo"] for it in items}
    size = (384, 576)
    t_global = time.time()
    fold_reports = []
    remaining_min = args.time_limit_min

    for f in range(args.n_folds):
        if remaining_min < 1.5:
            print(f"global time nearly exhausted before fold {f}; stopping", flush=True)
            break
        train_idx = [i for i in range(len(items)) if folds[i] != f]
        held_idx = [i for i in range(len(items)) if folds[i] == f]
        train_items = [items[i] for i in train_idx]
        held_items = [items[i] for i in held_idx]
        print(f"\n=== fold {f}: train={len(train_items)} held={len(held_items)} names={[it['name'] for it in held_items]} ===", flush=True)

        # fresh Frundles init each fold (nested-safe, no cross-fold weight reuse as teacher of held-out)
        model, size = load_ckpt(ckpt_in, device)
        model.train()
        fold_budget = min(args.fold_time_min * 60.0, remaining_min * 60.0 * 0.9)
        hist = train_fold(
            model,
            train_items,
            device,
            epochs=args.epochs,
            bs=args.bs,
            lr=args.lr,
            time_budget_s=fold_budget,
            blend=args.blend,
            size=size,
        )
        ckpt_path = seg_dir / f"fasc_multexp_ft_{args.tag}_fold{f}.pt"
        torch.save(
            {
                "encoder": "resnet34",
                "size": list(size),
                "state": model.state_dict(),
                "fold": f,
                "held": [it["name"] for it in held_items],
                "hist": hist,
                "blend": args.blend,
            },
            ckpt_path,
        )
        infer_heldout(model, held_items, apo_lookup, maps_out, device, size)
        elapsed = (time.time() - t_global) / 60.0
        remaining_min = args.time_limit_min - elapsed
        fold_reports.append(
            {
                "fold": f,
                "n_train": len(train_items),
                "n_held": len(held_items),
                "held": [it["name"] for it in held_items],
                "epochs_ran": len(hist),
                "last_loss": hist[-1]["loss"] if hist else None,
                "fold_min": (hist[-1]["sec"] / 60.0) if hist else 0.0,
                "ckpt": str(ckpt_path),
            }
        )
        print(f"fold {f} done ep={len(hist)} last_loss={fold_reports[-1]['last_loss']} remaining_min={remaining_min:.1f}", flush=True)

    # Ensure all images have maps; if some folds skipped, fall back to Frundles fasc for missing
    missing = []
    for it in items:
        p = maps_out / f"{it['name']}.npz"
        if not p.exists():
            missing.append(it["name"])
            np.savez_compressed(
                p,
                apo=it["apo"].astype(np.float16),
                fasc=it["fasc"].astype(np.float16),
            )
    if missing:
        print(f"WARNING: fell back to Frundles fasc for {len(missing)} images: {missing}", flush=True)

    print("eval frozen recipe on nested maps...", flush=True)
    summary, df = eval_maps(maps_out, expert_csv, expert_dir)
    print(summary, flush=True)
    df.to_csv(out_dir / f"multexp_fasc_ft_nested_{args.tag}_rows.csv", index=False)

    base_s, _ = eval_maps(cache, expert_csv, expert_dir)

    promote = bool(summary["S_loocv_cal_pa_mt"] < PRIOR_C)
    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "tag": args.tag,
        "protocol": (
            f"group-safe {args.n_folds}-fold nested MultExp fasc FT; "
            "dHash groups; Frundles fasc init; apo frozen from Frundles cache; "
            "geo-proxy target from train-fold GT PA/FL/MT + apo belly stripes; "
            "held-out maps inferred with fold weights only; "
            "honest LOOCV affine PA+MT, FL raw; frozen recipe hough_tex0.4_thr035_apo60 fl_sin_w=0.35"
        ),
        "leakage_notes": (
            "GT PA/FL/MT used ONLY to paint train-fold targets. "
            "Held-out image GT never enters that fold's loss or target. "
            "Final S uses honest LOOCV cal on nested maps (same as eval_promoted_hough_multexp). "
            "Risk: geo paint is a surrogate — may not match true fasc geometry; "
            "group split may still correlate similar MultExp anatomy across folds."
        ),
        "frozen_recipe_kw": {k: FROZEN[k] for k in ("fasc_thr", "apo_thr", "tex_w", "fl_factor", "fl_sin_w", "agree_thr", "pa_mode")},
        "ckpt_init": str(ckpt_in),
        "lab_scratch_best": False,
        "maps": str(maps_out),
        "n_folds": args.n_folds,
        "n_groups": n_groups,
        "fold_counts": np.bincount(folds, minlength=args.n_folds).tolist(),
        "fold_reports": fold_reports,
        "hyper": {
            "epochs": args.epochs,
            "bs": args.bs,
            "lr": args.lr,
            "blend": args.blend,
            "hash_dist": args.hash_dist,
            "seed": args.seed,
            "fold_time_min": args.fold_time_min,
            "time_limit_min": args.time_limit_min,
        },
        "train_wall_min": (time.time() - t_global) / 60.0,
        "device": str(device),
        "S_nested_ft": summary,
        "S_frundles_baseline_recheck": base_s,
        "prior_C_0.13195": PRIOR_C,
        "prior_fixed_0.13294": PRIOR_FIXED,
        "delta_vs_C": float(summary["S_loocv_cal_pa_mt"] - PRIOR_C),
        "gate_0.12": summary["gate_0.12"],
        "promote": promote,
        "kaggle_submit": False,
        "missing_fallback_frundles": missing,
    }
    out_json = out_dir / f"multexp_fasc_ft_nested_{args.tag}.json"
    out_json.write_text(json.dumps(payload, indent=2))

    status = out_dir / "multexp_fasc_ft_nested_status.md"
    status.write_text(
        f"""# MultExp nested fasc FT (geo proxy) — {payload['asof']}

## Result
- **Nested-safe S = {summary['S_loocv_cal_pa_mt']:.5f}** (`{args.tag}`)
  PA {summary['MAE_loocv']['pa_deg']:.3f}° / FL {summary['MAE_loocv']['fl_mm']:.3f} mm / MT {summary['MAE_loocv']['mt_mm']:.3f} mm
- vs prior C **{PRIOR_C:.5f}** (Δ {payload['delta_vs_C']:+.5f})
- vs fixed Frundles **{PRIOR_FIXED:.5f}**
- **Gate 0.12: {'PASS' if summary['gate_0.12'] else 'FAIL'}**
- Promote recipe: **{promote}**
- Train wall: {payload['train_wall_min']:.1f} min on {device}
- Folds ran: {len(fold_reports)}/{args.n_folds}; groups={n_groups}
- **No Kaggle**

## Protocol
{payload['protocol']}

## Leakage notes
{payload['leakage_notes']}

## Next lever
See JSON diagnosis fields / parent report.
"""
    )

    if promote:
        recipe = {
            "asof": payload["asof"],
            "name": f"nested_fasc_ft_{args.tag}",
            "S_loocv_cal_pa_mt": summary["S_loocv_cal_pa_mt"],
            "MAE_loocv": summary["MAE_loocv"],
            "kw": payload["frozen_recipe_kw"],
            "fasc_adapt": {
                "type": "nested_group_fasc_ft_geo_proxy",
                "tag": args.tag,
                "n_folds": args.n_folds,
                "maps": str(maps_out),
                "note": "Group-safe nested FT; apo frozen Frundles; geo-proxy targets on train only",
            },
            "cache_dir": str(maps_out),
            "maps": f"Nested FT fasc ({args.tag}) + Frundles apo",
            "prior_C_S": PRIOR_C,
            "beats_prior_C": True,
            "gate_0.12": summary["gate_0.12"],
            "gap_to_0.12": float(summary["S_loocv_cal_pa_mt"] - 0.12),
            "kaggle_submit": False,
            "lab_fasc_ckpt": False,
            "protocol": payload["protocol"],
        }
        (out_dir / "math_map_promoted_recipe.json").write_text(json.dumps(recipe, indent=2))
        print("PROMOTED math_map_promoted_recipe.json", flush=True)
    else:
        print("NOT promoted (S did not beat 0.13195)", flush=True)

    print(
        f"FT_NESTED_RESULT S={summary['S_loocv_cal_pa_mt']:.5f} "
        f"PA={summary['MAE_loocv']['pa_deg']:.3f} FL={summary['MAE_loocv']['fl_mm']:.3f} MT={summary['MAE_loocv']['mt_mm']:.3f} "
        f"vs_C={PRIOR_C:.5f} gate={summary['gate_0.12']} promote={promote} wall_min={payload['train_wall_min']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
