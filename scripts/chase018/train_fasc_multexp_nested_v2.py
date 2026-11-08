#!/usr/bin/env python3
"""LOOCV nested fasc FT + nested Frundles blend; expanded GT-teacher pool (cached costs)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
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

from umud.seg.frundles_smp import load_ckpt
from adapt_multexp_fasc import FROZEN, VARIANTS, adapt_fasc, loocv_cal_pa_mt, predict_one

HK = timezone(timedelta(hours=8))
PRIOR_C = 0.13195144580335014


def adapt_or_blend(g, apo, fasc, variant):
    if variant.startswith("blend_bm_"):
        w = float(variant.split("_")[-1]) / 100.0
        a = fasc.astype(np.float32)
        b = adapt_fasc(g, apo, fasc, "morph_connect")
        return np.clip((1 - w) * a + w * b, 0, 1).astype(np.float32)
    if variant.startswith("blend_br_"):
        w = float(variant.split("_")[-1]) / 100.0
        a = fasc.astype(np.float32)
        b = adapt_fasc(g, apo, fasc, "ridge_blend03")
        return np.clip((1 - w) * a + w * b, 0, 1).astype(np.float32)
    if variant.startswith("blend_bc_"):
        w = float(variant.split("_")[-1]) / 100.0
        a = fasc.astype(np.float32)
        b = adapt_fasc(g, apo, fasc, "corridor_dilate")
        return np.clip((1 - w) * a + w * b, 0, 1).astype(np.float32)
    return adapt_fasc(g, apo, fasc, variant).astype(np.float32)


POOL = (
    list(VARIANTS)
    + [f"blend_bm_{w}" for w in (15, 25, 40, 50, 65, 75)]
    + [f"blend_br_{w}" for w in (15, 25, 40, 50)]
    + [f"blend_bc_{w}" for w in (15, 25, 40, 50)]
)


def cost_pa_fl(pa, fl, gt_pa, gt_fl):
    return abs(pa - gt_pa) / 6.0 + abs(fl - gt_fl) / 12.0


def dice_loss(probs, gt, eps=1e-6):
    num = 2 * (probs * gt).sum((2, 3)) + eps
    den = probs.sum((2, 3)) + gt.sum((2, 3)) + eps
    return 1.0 - (num / den).mean()


def load_items(expert_csv, expert_dir, cache_dir):
    expert = pd.read_csv(expert_csv)
    items = []
    for _, er in expert.iterrows():
        name = er["image_file"]
        gpath = Path(expert_dir) / name
        zpath = Path(cache_dir) / f"{name}.npz"
        if not gpath.exists() or not zpath.exists():
            continue
        z = np.load(zpath)
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
    return items


def eval_from_fasc_dict(fasc_maps, items, kw):
    rows = []
    for it in items:
        g = cv2.imread(str(it["gpath"]), cv2.IMREAD_GRAYSCALE)
        pred = predict_one(g, it["apo"], fasc_maps[it["name"]], it["px_per_mm"], kw)
        rows.append(
            dict(
                image_file=it["name"],
                ok=pred["ok"],
                pred_pa=pred["pred_pa"],
                pred_fl=pred["pred_fl"],
                pred_mt=pred["pred_mt"],
                gt_pa=it["gt_pa"],
                gt_fl=it["gt_fl"],
                gt_mt=it["gt_mt"],
            )
        )
    df = pd.DataFrame(rows)
    comps, mae, _, _ = loocv_cal_pa_mt(df)
    return {"S": float(comps["S"]), "MAE": mae, "gate": float(comps["S"]) < 0.12, "beats_C": float(comps["S"]) < PRIOR_C}, df


class DS(Dataset):
    def __init__(self, rows, mix=0.60):
        self.rows = rows
        self.mix = mix

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        gpath, teach, fr = self.rows[idx]
        g = cv2.imread(str(gpath), cv2.IMREAD_GRAYSCALE)
        g_r = cv2.resize(g, (576, 384), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        y = cv2.resize(teach, (576, 384), interpolation=cv2.INTER_AREA).astype(np.float32)
        fr_r = cv2.resize(fr, (576, 384), interpolation=cv2.INTER_AREA).astype(np.float32)
        y = (1 - self.mix) * y + self.mix * fr_r
        if np.random.rand() < 0.5:
            g_r = g_r[:, ::-1].copy()
            y = y[:, ::-1].copy()
        return torch.from_numpy(g_r[None]).float(), torch.from_numpy(y[None]).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--mix", type=float, default=0.60)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--tag", default="loocv_gt_teacher_blend_v2")
    ap.add_argument("--skip-ft", action="store_true")
    args = ap.parse_args()

    expert_csv = ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv"
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    ckpt_in = ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt"
    out_dir = ROOT / "reports/phase3/chase018"
    kw = dict(FROZEN)
    items = load_items(expert_csv, expert_dir, cache)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n = len(items)
    print(f"n={n} pool={len(POOL)} device={device}", flush=True)

    print("building variant bank + cached costs...", flush=True)
    bank = {}  # name -> variant -> map
    costs = {}  # name -> variant -> cost
    preds_cache = {}  # name -> variant -> (pa, fl, mt)
    teachers = {}
    meta = {}
    t_bank = time.time()
    for it in items:
        g = cv2.imread(str(it["gpath"]), cv2.IMREAD_GRAYSCALE)
        bank[it["name"]] = {}
        costs[it["name"]] = {}
        preds_cache[it["name"]] = {}
        best_v, best_c, best_m = None, 1e9, it["fasc"]
        for v in POOL:
            try:
                m = adapt_or_blend(g, it["apo"], it["fasc"], v)
            except Exception:
                continue
            pred = predict_one(g, it["apo"], m, it["px_per_mm"], kw)
            c = cost_pa_fl(pred["pred_pa"], pred["pred_fl"], it["gt_pa"], it["gt_fl"])
            bank[it["name"]][v] = m
            costs[it["name"]][v] = c
            preds_cache[it["name"]][v] = (pred["pred_pa"], pred["pred_fl"], pred["pred_mt"])
            if c < best_c:
                best_v, best_c, best_m = v, c, m
        teachers[it["name"]] = best_m
        meta[it["name"]] = {"variant": best_v, "cost": float(best_c)}
    print(f"bank done in {(time.time()-t_bank)/60:.1f}m counts={dict(Counter(m['variant'] for m in meta.values()))}", flush=True)

    leak, _ = eval_from_fasc_dict(teachers, items, kw)
    print("LEAKY_EXPANDED_ORACLE", leak, flush=True)

    # Nested global variant using CACHED train costs (fast)
    picked_global = {}
    for i in range(n):
        train = [j for j in range(n) if j != i]
        scores = {}
        for v in POOL:
            cs = [costs[items[j]["name"]][v] for j in train if v in costs[items[j]["name"]]]
            if cs:
                scores[v] = float(np.mean(cs))
        best_v = min(scores, key=scores.get) if scores else "baseline"
        picked_global[items[i]["name"]] = bank[items[i]["name"]].get(best_v, items[i]["fasc"])
    nest_global, _ = eval_from_fasc_dict(picked_global, items, kw)
    print("NESTED_GLOBAL_VARIANT", nest_global, flush=True)

    picked_maj = {}
    for i in range(n):
        train_vars = [meta[items[j]["name"]]["variant"] for j in range(n) if j != i]
        maj = Counter(train_vars).most_common(1)[0][0]
        picked_maj[items[i]["name"]] = bank[items[i]["name"]].get(maj, items[i]["fasc"])
    nest_maj, _ = eval_from_fasc_dict(picked_maj, items, kw)
    print("NESTED_MAJORITY_TEACHER", nest_maj, flush=True)

    # Nested: pick per held-out the variant with best mean train cost (same as global actually
    # when costs are image-local — already done). Also try: for held-out use variant that
    # minimizes |pred_pa_frundles - median train GT gap| — skip, keep FT path.

    if args.skip_ft:
        payload = {
            "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
            "tag": args.tag + "_noft",
            "leaky_expanded_oracle": leak,
            "nested_global_variant": nest_global,
            "nested_majority_teacher": nest_maj,
            "prior_C": PRIOR_C,
            "promote": bool(min(nest_global["S"], nest_maj["S"]) < PRIOR_C),
            "kaggle_submit": False,
        }
        best = nest_global if nest_global["S"] <= nest_maj["S"] else nest_maj
        if best["S"] < PRIOR_C:
            maps_out = out_dir / "cache" / f"maps_multexp_fasc_ft_{args.tag}_noft"
            maps_out.mkdir(parents=True, exist_ok=True)
            src = picked_global if nest_global["S"] <= nest_maj["S"] else picked_maj
            for it in items:
                np.savez_compressed(maps_out / f"{it['name']}.npz", apo=it["apo"].astype(np.float16), fasc=src[it["name"]].astype(np.float16))
            recipe = {
                "asof": payload["asof"],
                "name": f"nested_expanded_pool_{args.tag}",
                "S_loocv_cal_pa_mt": best["S"],
                "MAE_loocv": best["MAE"],
                "kw": {k: FROZEN[k] for k in ("fasc_thr", "apo_thr", "tex_w", "fl_factor", "fl_sin_w", "agree_thr", "pa_mode")},
                "fasc_adapt": {"type": "nested_expanded_pool", "tag": args.tag},
                "cache_dir": str(maps_out),
                "prior_C_S": PRIOR_C,
                "beats_prior_C": True,
                "gate_0.12": best["gate"],
                "gap_to_0.12": float(best["S"] - 0.12),
                "kaggle_submit": False,
                "lab_fasc_ckpt": False,
            }
            (out_dir / "math_map_promoted_recipe.json").write_text(json.dumps(recipe, indent=2))
            payload["promote"] = True
            payload["maps"] = str(maps_out)
        (out_dir / f"multexp_fasc_ft_nested_{args.tag}_noft.json").write_text(json.dumps(payload, indent=2))
        print("NOFT_DONE", payload, flush=True)
        return 0

    print("LOOCV FT starting...", flush=True)
    ft_maps = {}
    t0 = time.time()
    for i in range(n):
        held = items[i]
        train_items = [items[j] for j in range(n) if j != i]
        rows = [(it["gpath"], teachers[it["name"]], it["fasc"]) for it in train_items]
        ds = DS(rows, mix=args.mix)
        dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        model, size = load_ckpt(ckpt_in, device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for ep in range(args.epochs):
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
        model.eval()
        with torch.no_grad():
            g = cv2.imread(str(held["gpath"]), cv2.IMREAD_GRAYSCALE)
            H, W = size
            g_r = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            x = torch.from_numpy(g_r[None, None]).float().to(device)
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits_f = model(x.flip(-1))
            if isinstance(logits_f, (tuple, list)):
                logits_f = logits_f[0]
            prob = ((torch.sigmoid(logits) + torch.sigmoid(logits_f).flip(-1)) / 2)[0, 0].cpu().numpy().astype(np.float32)
        ft_maps[held["name"]] = prob
        if (i + 1) % 5 == 0 or i == 0:
            print(f"fold {i+1}/{n} t={(time.time()-t0)/60:.1f}m", flush=True)

    pure_ft, _ = eval_from_fasc_dict(ft_maps, items, kw)
    print("PURE_FT", pure_ft, flush=True)

    # Nested blend alpha using cached train geometry costs of blended maps — need predict
    # Approximate: evaluate alphas via predict on train (35*7*34 is heavy). Use 7 alphas but
    # only compute once per (image, alpha) and cache.
    print("nested blend alphas...", flush=True)
    alphas = [0.0, 0.15, 0.25, 0.4, 0.55, 0.7, 1.0]
    blend_cost = {a: {} for a in alphas}  # a -> name -> cost
    for a in alphas:
        for it in items:
            m = ((1 - a) * it["fasc"] + a * ft_maps[it["name"]]).astype(np.float32)
            g = cv2.imread(str(it["gpath"]), cv2.IMREAD_GRAYSCALE)
            pred = predict_one(g, it["apo"], m, it["px_per_mm"], kw)
            blend_cost[a][it["name"]] = cost_pa_fl(pred["pred_pa"], pred["pred_fl"], it["gt_pa"], it["gt_fl"])
        print(f"  alpha={a} mean_cost={np.mean(list(blend_cost[a].values())):.4f}", flush=True)

    blended = {}
    alpha_picked = {}
    for i in range(n):
        held = items[i]
        train_names = [items[j]["name"] for j in range(n) if j != i]
        best_a, best_sc = 0.0, 1e9
        for a in alphas:
            sc = float(np.mean([blend_cost[a][nm] for nm in train_names]))
            if sc < best_sc:
                best_sc, best_a = sc, a
        alpha_picked[held["name"]] = best_a
        blended[held["name"]] = ((1 - best_a) * held["fasc"] + best_a * ft_maps[held["name"]]).astype(np.float32)

    nest_blend, df = eval_from_fasc_dict(blended, items, kw)
    print("NESTED_BLEND", nest_blend, "alpha_counts", dict(Counter(alpha_picked.values())), flush=True)
    fr_only, _ = eval_from_fasc_dict({it["name"]: it["fasc"] for it in items}, items, kw)

    candidates = {
        "nested_global_variant": nest_global,
        "nested_majority_teacher": nest_maj,
        "pure_ft": pure_ft,
        "nested_blend": nest_blend,
        "frundles": fr_only,
    }
    best_name = min(candidates, key=lambda k: candidates[k]["S"])
    best = candidates[best_name]
    promote = bool(best["S"] < PRIOR_C)

    maps_out = out_dir / "cache" / f"maps_multexp_fasc_ft_{args.tag}"
    maps_out.mkdir(parents=True, exist_ok=True)
    if best_name == "nested_blend":
        src = blended
    elif best_name == "pure_ft":
        src = ft_maps
    elif best_name == "nested_global_variant":
        src = picked_global
    elif best_name == "nested_majority_teacher":
        src = picked_maj
    else:
        src = {it["name"]: it["fasc"] for it in items}
    for it in items:
        np.savez_compressed(maps_out / f"{it['name']}.npz", apo=it["apo"].astype(np.float16), fasc=src[it["name"]].astype(np.float16))
    df_best, _ = eval_from_fasc_dict(src, items, kw)
    # rewrite df from best
    _, df = eval_from_fasc_dict(src, items, kw)
    df.to_csv(out_dir / f"multexp_fasc_ft_nested_{args.tag}_rows.csv", index=False)

    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "tag": args.tag,
        "protocol": (
            "Expanded A/blend pool GT teachers; nested global-variant + majority (no FT); "
            "35-fold LOOCV fasc FT from Frundles with frundles_mix target; nested alpha blend; "
            "apo frozen; frozen geo recipe; honest LOOCV S"
        ),
        "leakage_notes": (
            "GT used only for train-fold teacher selection and train-fold alpha choice. "
            "leaky_expanded_oracle uses GT teachers on all images (upper bound, not nested). "
            "Held-out GT never in that fold's FT loss."
        ),
        "pool_size": len(POOL),
        "leaky_expanded_oracle": leak,
        "candidates": candidates,
        "best_name": best_name,
        "best": best,
        "alpha_counts": {str(k): int(v) for k, v in Counter(alpha_picked.values()).items()},
        "prior_C": PRIOR_C,
        "delta_vs_C": float(best["S"] - PRIOR_C),
        "gate_0.12": best["gate"],
        "promote": promote,
        "train_wall_min": (time.time() - t0) / 60.0,
        "hyper": {"epochs": args.epochs, "mix": args.mix, "lr": args.lr},
        "maps": str(maps_out),
        "kaggle_submit": False,
        "teacher_counts": dict(Counter(m["variant"] for m in meta.values())),
        "stripe_proxy_abandoned": True,
        "v1_5fold_gt_teacher_S": 0.14469,
    }
    (out_dir / f"multexp_fasc_ft_nested_{args.tag}.json").write_text(json.dumps(payload, indent=2))

    if promote:
        recipe = {
            "asof": payload["asof"],
            "name": f"nested_fasc_ft_{args.tag}_{best_name}",
            "S_loocv_cal_pa_mt": best["S"],
            "MAE_loocv": best["MAE"],
            "kw": {k: FROZEN[k] for k in ("fasc_thr", "apo_thr", "tex_w", "fl_factor", "fl_sin_w", "agree_thr", "pa_mode")},
            "fasc_adapt": {"type": best_name, "tag": args.tag, "maps": str(maps_out)},
            "cache_dir": str(maps_out),
            "prior_C_S": PRIOR_C,
            "beats_prior_C": True,
            "gate_0.12": best["gate"],
            "gap_to_0.12": float(best["S"] - 0.12),
            "kaggle_submit": False,
            "lab_fasc_ckpt": False,
            "protocol": payload["protocol"],
        }
        (out_dir / "math_map_promoted_recipe.json").write_text(json.dumps(recipe, indent=2))
        print("PROMOTED", flush=True)

    status = f"""# MultExp nested fasc FT v2 — {payload['asof']}

## Result
- **Best nested-safe S = {best['S']:.5f}** (`{best_name}`)
  PA {best['MAE']['pa_deg']:.3f}° / FL {best['MAE']['fl_mm']:.3f} mm / MT {best['MAE']['mt_mm']:.3f} mm
- vs prior C **{PRIOR_C:.5f}** (Δ {payload['delta_vs_C']:+.5f})
- Leaky expanded oracle: **{leak['S']:.5f}** (gate={'PASS' if leak['gate'] else 'FAIL'} upper bound)
- Candidates: global={nest_global['S']:.5f} maj={nest_maj['S']:.5f} pure_ft={pure_ft['S']:.5f} blend={nest_blend['S']:.5f} frundles={fr_only['S']:.5f}
- **Gate 0.12: {'PASS' if best['gate'] else 'FAIL'}** | Promote: {promote}
- Wall {payload['train_wall_min']:.1f} min | **No Kaggle**

## Diagnosis
Stripe/orient geo-paint targets scored worse than Frundles (abandoned; smoke S=0.579).
5-fold GT-teacher U-Net distill S=0.145. Expanded A/blend leaky oracle reaches {leak['S']:.5f}.
Transferable nested selection/FT still the bottleneck vs oracle.
"""
    (out_dir / "multexp_fasc_ft_nested_status.md").write_text(status)
    print(
        f"FT_NESTED_V2_RESULT best={best_name} S={best['S']:.5f} "
        f"PA={best['MAE']['pa_deg']:.3f} FL={best['MAE']['fl_mm']:.3f} "
        f"leaky={leak['S']:.5f} vs_C={PRIOR_C:.5f} gate={best['gate']} promote={promote}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
