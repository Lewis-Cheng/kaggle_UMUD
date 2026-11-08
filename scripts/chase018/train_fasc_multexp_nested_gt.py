#!/usr/bin/env python3
"""Nested MultExp fasc FT with GT-selected geometry teachers (fold-safe).

Proxy: on TRAIN folds only, for each image pick the A-variant fasc map that
minimizes raw normalized PA+FL error vs MultExp GT (apo frozen). Distill those
teachers into Frundles fasc U-Net; infer held-out with fold weights.

Unlike failed self-train (fixed morph+ridge teacher), teachers are GT-scored
per image on train only. Held-out GT never enters that fold's FT.

Promote iff S < 0.13195; gate iff S < 0.12. No Kaggle. No attention.
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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from umud.geometry.host_faithful import dhash  # noqa: E402
from umud.seg.frundles_smp import load_ckpt  # noqa: E402
from adapt_multexp_fasc import (  # noqa: E402
    FROZEN,
    VARIANTS,
    adapt_fasc,
    loocv_cal_pa_mt,
    predict_one,
)

HK = timezone(timedelta(hours=8))
PRIOR_C = 0.13195144580335014
PRIOR_FIXED = 0.13293555901039555

# Mild pool: baseline + best C partners + light blends
TEACHER_VARIANTS = [
    "baseline",
    "morph_connect",
    "ridge_soft02",
    "ridge_blend03",
    "corridor_dilate",
    "belly_gate",
    "blend_base_morph",
]


def adapt_or_blend(g, apo, fasc, variant: str) -> np.ndarray:
    if variant == "ridge_soft02":
        # soft ridge from adapt_multexp_fasc v1 naming in C pool
        try:
            return adapt_fasc(g, apo, fasc, "ridge_blend03") * 0.0 + (
                0.8 * fasc.astype(np.float32)
                + 0.2 * adapt_fasc(g, apo, fasc, "ridge_blend03")
            )
        except Exception:
            return fasc.astype(np.float32)
    if variant == "blend_base_morph":
        a = fasc.astype(np.float32)
        b = adapt_fasc(g, apo, fasc, "morph_connect")
        return np.clip(0.5 * a + 0.5 * b, 0, 1).astype(np.float32)
    if variant == "blend_base_morph25":
        a = fasc.astype(np.float32)
        b = adapt_fasc(g, apo, fasc, "morph_connect")
        return np.clip(0.75 * a + 0.25 * b, 0, 1).astype(np.float32)
    if variant == "blend_base_morph75":
        a = fasc.astype(np.float32)
        b = adapt_fasc(g, apo, fasc, "morph_connect")
        return np.clip(0.25 * a + 0.75 * b, 0, 1).astype(np.float32)
    if variant in VARIANTS or variant in (
        "baseline",
        "morph_connect",
        "ridge_blend03",
        "corridor_dilate",
        "belly_gate",
    ):
        return adapt_fasc(g, apo, fasc, variant).astype(np.float32)
    raise ValueError(variant)


TEACHER_VARIANTS = [
    "baseline",
    "morph_connect",
    "ridge_blend03",
    "corridor_dilate",
    "belly_gate",
    "blend_base_morph",
    "blend_base_morph25",
    "blend_base_morph75",
]


def raw_geo_cost(pred, gt_pa, gt_fl) -> float:
    """Normalized raw PA+FL cost (no cal); MT ignored (apo-fixed)."""
    return abs(pred["pred_pa"] - gt_pa) / 6.0 + abs(pred["pred_fl"] - gt_fl) / 12.0


def select_teacher(g, apo, fasc, gt_pa, gt_fl, px_per_mm, kw):
    best_v, best_c, best_m = "baseline", 1e9, fasc.astype(np.float32)
    for v in TEACHER_VARIANTS:
        try:
            m = adapt_or_blend(g, apo, fasc, v)
        except Exception:
            continue
        pred = predict_one(g, apo, m, px_per_mm, kw)
        c = raw_geo_cost(pred, gt_pa, gt_fl)
        if c < best_c:
            best_v, best_c, best_m = v, c, m
    return best_v, best_c, best_m


def dice_loss(probs, gt, eps=1e-6):
    num = 2 * (probs * gt).sum((2, 3)) + eps
    den = probs.sum((2, 3)) + gt.sum((2, 3)) + eps
    return 1.0 - (num / den).mean()


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


def assign_folds(group_ids, n_folds, seed=0):
    uniq = sorted(set(int(g) for g in group_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    g2f = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([g2f[int(g)] for g in group_ids], dtype=int)


def load_items(expert_csv, expert_dir, cache_dir):
    expert = pd.read_csv(expert_csv)
    expert_dir = Path(expert_dir)
    cache_dir = Path(cache_dir)
    items, hashes = [], []
    for _, er in expert.iterrows():
        name = er["image_file"]
        gpath = expert_dir / name
        zpath = cache_dir / f"{name}.npz"
        if not gpath.exists() or not zpath.exists():
            continue
        z = np.load(zpath)
        g = cv2.imread(str(gpath), cv2.IMREAD_GRAYSCALE)
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
            )
        )
        hashes.append(dhash(g))
    return items, hashes


class TeacherDS(Dataset):
    def __init__(self, rows, size=(384, 576), augment=True, frundles_mix=0.35):
        # rows: list of (gpath, teacher_map HxW, frundles_fasc HxW)
        self.rows = rows
        self.H, self.W = size
        self.augment = augment
        self.frundles_mix = frundles_mix

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        gpath, teach, fr = self.rows[idx]
        g = cv2.imread(str(gpath), cv2.IMREAD_GRAYSCALE)
        g_r = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        y = teach.astype(np.float32)
        if y.shape != (self.H, self.W):
            y = cv2.resize(y, (self.W, self.H), interpolation=cv2.INTER_AREA)
        fr_r = fr.astype(np.float32)
        if fr_r.shape != (self.H, self.W):
            fr_r = cv2.resize(fr_r, (self.W, self.H), interpolation=cv2.INTER_AREA)
        # mix keeps student near Frundles (prevents self-train collapse)
        y = (1.0 - self.frundles_mix) * y + self.frundles_mix * fr_r
        if self.augment:
            if np.random.rand() < 0.5:
                g_r = g_r[:, ::-1].copy()
                y = y[:, ::-1].copy()
            if np.random.rand() < 0.4:
                alpha = 0.85 + 0.3 * np.random.rand()
                beta = (np.random.rand() - 0.5) * 0.1
                g_r = np.clip(g_r * alpha + beta, 0, 1)
        return torch.from_numpy(g_r[None]).float(), torch.from_numpy(y[None]).float()


@torch.no_grad()
def infer_heldout(model, held_items, out_dir, device, size):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    H, W = size
    for it in held_items:
        g = cv2.imread(str(it["gpath"]), cv2.IMREAD_GRAYSCALE)
        apo = it["apo"]
        g_r = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = torch.from_numpy(g_r[None, None]).float().to(device)
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits_f = model(x.flip(-1))
        if isinstance(logits_f, (tuple, list)):
            logits_f = logits_f[0]
        prob = ((torch.sigmoid(logits) + torch.sigmoid(logits_f).flip(-1)) / 2)[0, 0].cpu().numpy().astype(np.float32)
        apo_r = apo if apo.shape == prob.shape else cv2.resize(apo, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_AREA)
        np.savez_compressed(out_dir / f"{it['name']}.npz", apo=apo_r.astype(np.float16), fasc=prob.astype(np.float16))


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
        pred = predict_one(g, z["apo"].astype(np.float32), z["fasc"].astype(np.float32), float(er["Scale_pixel_per_cm"]) / 10.0, kw)
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
    }, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--frundles-mix", type=float, default=0.40)
    ap.add_argument("--time-limit-min", type=float, default=45.0)
    ap.add_argument("--fold-time-min", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="nested_gt_teacher_v1")
    ap.add_argument("--oracle-targets-only", action="store_true", help="skip FT; dump GT-selected teachers for all (leaky upper bound)")
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
    kw = dict(FROZEN)
    items, hashes = load_items(expert_csv, expert_dir, cache)
    print(f"device={device} n={len(items)} tag={args.tag}", flush=True)

    # Precompute teachers for ALL images (uses GT) — ONLY used for train indices per fold.
    # For leaky upper-bound diagnostic, also eval all-teacher maps.
    print("selecting GT teachers...", flush=True)
    teachers = {}
    teacher_meta = {}
    for it in items:
        g = cv2.imread(str(it["gpath"]), cv2.IMREAD_GRAYSCALE)
        v, c, m = select_teacher(g, it["apo"], it["fasc"], it["gt_pa"], it["gt_fl"], it["px_per_mm"], kw)
        teachers[it["name"]] = m
        teacher_meta[it["name"]] = {"variant": v, "raw_cost": float(c)}
    from collections import Counter
    print("teacher_counts", dict(Counter(t["variant"] for t in teacher_meta.values())), flush=True)

    # Leaky oracle of teachers (diagnostic only — NOT promoted)
    leak_dir = out_dir / "cache" / f"maps_multexp_gt_teacher_leaky_{args.tag}"
    leak_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        np.savez_compressed(
            leak_dir / f"{it['name']}.npz",
            apo=it["apo"].astype(np.float16),
            fasc=teachers[it["name"]].astype(np.float16),
        )
    leak_s, _ = eval_maps(leak_dir, expert_csv, expert_dir)
    print("LEAKY_TEACHER_ORACLE", leak_s, flush=True)

    if args.oracle_targets_only:
        payload = {
            "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
            "tag": args.tag,
            "leaky_teacher_oracle": leak_s,
            "teacher_meta": teacher_meta,
            "note": "leaky upper bound only",
            "promote": False,
            "kaggle_submit": False,
        }
        (out_dir / f"multexp_fasc_ft_nested_{args.tag}.json").write_text(json.dumps(payload, indent=2))
        return 0

    gids = groups_from_hashes(hashes, 12)
    folds = assign_folds(gids, args.n_folds, args.seed)
    print(f"n_groups={len(set(map(int,gids)))} fold_counts={np.bincount(folds, minlength=args.n_folds).tolist()}", flush=True)

    t_global = time.time()
    fold_reports = []
    remaining = args.time_limit_min

    for f in range(args.n_folds):
        if remaining < 1.5:
            print(f"time exhausted before fold {f}", flush=True)
            break
        train_items = [items[i] for i in range(len(items)) if folds[i] != f]
        held_items = [items[i] for i in range(len(items)) if folds[i] == f]
        print(f"\n=== fold {f} train={len(train_items)} held={len(held_items)} ===", flush=True)

        rows = []
        for it in train_items:
            rows.append((it["gpath"], teachers[it["name"]], it["fasc"]))
        ds = TeacherDS(rows, size=(384, 576), augment=True, frundles_mix=args.frundles_mix)
        dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0)

        model, size = load_ckpt(ckpt_in, device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        fold_budget = min(args.fold_time_min * 60.0, remaining * 60.0 * 0.85)
        t0 = time.time()
        hist = []
        for ep in range(1, args.epochs + 1):
            losses = []
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probs = torch.sigmoid(logits)
                loss = 0.5 * dice_loss(probs, y) + 0.5 * F.binary_cross_entropy(probs, y)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            mean_l = float(np.mean(losses)) if losses else 9.0
            hist.append({"ep": ep, "loss": mean_l, "sec": time.time() - t0})
            if time.time() - t0 >= fold_budget:
                break
        ckpt_path = seg_dir / f"fasc_multexp_ft_{args.tag}_fold{f}.pt"
        torch.save({"encoder": "resnet34", "size": list(size), "state": model.state_dict(), "fold": f, "hist": hist}, ckpt_path)
        infer_heldout(model, held_items, maps_out, device, size)
        elapsed = (time.time() - t_global) / 60.0
        remaining = args.time_limit_min - elapsed
        fold_reports.append(
            {
                "fold": f,
                "n_train": len(train_items),
                "n_held": len(held_items),
                "held": [it["name"] for it in held_items],
                "epochs_ran": len(hist),
                "last_loss": hist[-1]["loss"] if hist else None,
                "fold_min": (hist[-1]["sec"] / 60.0) if hist else 0.0,
            }
        )
        print(f"fold {f} done ep={len(hist)} loss={fold_reports[-1]['last_loss']:.4f} rem={remaining:.1f}m", flush=True)

    missing = []
    for it in items:
        p = maps_out / f"{it['name']}.npz"
        if not p.exists():
            missing.append(it["name"])
            np.savez_compressed(p, apo=it["apo"].astype(np.float16), fasc=it["fasc"].astype(np.float16))

    summary, df = eval_maps(maps_out, expert_csv, expert_dir)
    base_s, _ = eval_maps(cache, expert_csv, expert_dir)
    df.to_csv(out_dir / f"multexp_fasc_ft_nested_{args.tag}_rows.csv", index=False)
    promote = bool(summary["S_loocv_cal_pa_mt"] < PRIOR_C)

    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "tag": args.tag,
        "protocol": (
            f"group-safe {args.n_folds}-fold nested fasc FT; Frundles init; apo frozen; "
            "GT-selected A-variant teachers on train folds only (raw PA/6+FL/12); "
            f"frundles_mix={args.frundles_mix} in target; honest LOOCV under frozen recipe"
        ),
        "leakage_notes": (
            "GT used only to pick train-fold teachers among A post-filters. "
            "Held-out GT never in that fold's loss. "
            "leaky_teacher_oracle scores all-image GT teachers (diagnostic upper bound, not nested). "
            "Teacher pool limited to Frundles A variants — cannot beat A-oracle ~0.12 ceiling without new map generators."
        ),
        "teacher_pool": TEACHER_VARIANTS,
        "teacher_counts": dict(Counter(t["variant"] for t in teacher_meta.values())),
        "teacher_meta": teacher_meta,
        "leaky_teacher_oracle": leak_s,
        "frozen_recipe_kw": {k: FROZEN[k] for k in ("fasc_thr", "apo_thr", "tex_w", "fl_factor", "fl_sin_w", "agree_thr", "pa_mode")},
        "ckpt_init": str(ckpt_in),
        "lab_scratch_best": False,
        "maps": str(maps_out),
        "n_folds": args.n_folds,
        "fold_counts": np.bincount(folds, minlength=args.n_folds).tolist(),
        "fold_reports": fold_reports,
        "hyper": {
            "epochs": args.epochs,
            "bs": args.bs,
            "lr": args.lr,
            "frundles_mix": args.frundles_mix,
            "seed": args.seed,
        },
        "train_wall_min": (time.time() - t_global) / 60.0,
        "device": str(device),
        "S_nested_ft": summary,
        "S_frundles_baseline_recheck": base_s,
        "prior_C_0.13195": PRIOR_C,
        "delta_vs_C": float(summary["S_loocv_cal_pa_mt"] - PRIOR_C),
        "gate_0.12": summary["gate_0.12"],
        "promote": promote,
        "kaggle_submit": False,
        "missing_fallback_frundles": missing,
        "stripe_proxy_abandoned": {
            "reason": "geo-paint/orient targets themselves scored S=0.14-0.29 > Frundles 0.133",
            "smoke_S": 0.5789967919128125,
        },
    }
    (out_dir / f"multexp_fasc_ft_nested_{args.tag}.json").write_text(json.dumps(payload, indent=2))

    status = out_dir / "multexp_fasc_ft_nested_status.md"
    status.write_text(
        f"""# MultExp nested fasc FT — {payload['asof']}

## Result
- **Nested-safe S = {summary['S_loocv_cal_pa_mt']:.5f}** (`{args.tag}`)
  PA {summary['MAE_loocv']['pa_deg']:.3f}° / FL {summary['MAE_loocv']['fl_mm']:.3f} mm / MT {summary['MAE_loocv']['mt_mm']:.3f} mm
- vs prior C **{PRIOR_C:.5f}** (Δ {payload['delta_vs_C']:+.5f})
- Leaky GT-teacher oracle (diagnostic): **{leak_s['S_loocv_cal_pa_mt']:.5f}**
- **Gate 0.12: {'PASS' if summary['gate_0.12'] else 'FAIL'}**
- Promote: **{promote}** | wall {payload['train_wall_min']:.1f} min on {device}
- Stripe/orient geo-paint proxy abandoned (targets worse than Frundles; smoke S=0.579)
- **No Kaggle**

## Protocol
{payload['protocol']}

## Leakage
{payload['leakage_notes']}
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
                "type": "nested_gt_teacher_fasc_ft",
                "tag": args.tag,
                "maps": str(maps_out),
            },
            "cache_dir": str(maps_out),
            "prior_C_S": PRIOR_C,
            "beats_prior_C": True,
            "gate_0.12": summary["gate_0.12"],
            "gap_to_0.12": float(summary["S_loocv_cal_pa_mt"] - 0.12),
            "kaggle_submit": False,
            "lab_fasc_ckpt": False,
            "protocol": payload["protocol"],
        }
        (out_dir / "math_map_promoted_recipe.json").write_text(json.dumps(recipe, indent=2))
        print("PROMOTED", flush=True)
    else:
        print("NOT promoted", flush=True)

    print(
        f"FT_NESTED_RESULT S={summary['S_loocv_cal_pa_mt']:.5f} "
        f"PA={summary['MAE_loocv']['pa_deg']:.3f} FL={summary['MAE_loocv']['fl_mm']:.3f} MT={summary['MAE_loocv']['mt_mm']:.3f} "
        f"leaky_oracle={leak_s['S_loocv_cal_pa_mt']:.5f} vs_C={PRIOR_C:.5f} gate={summary['gate_0.12']} promote={promote}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
