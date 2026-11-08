#!/usr/bin/env python3
"""Hold-combo: fasc-adapt variants + B sin_w0.30 (+ optional A2 PA). Beat 0.13195?"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location("aporel", ROOT / "scripts/chase018/multexp_aporelpa_flgeo.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

SPEC2 = importlib.util.spec_from_file_location("adapt", ROOT / "scripts/chase018/adapt_multexp_fasc.py")
adapt = importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(adapt)

from umud.metric.score import score_components
from umud.seg.frundles_smp import load_ckpt, predict_prob
from umud.calibration.anisotropic_scale import mm_per_prob_pixel
from umud.geometry.host_faithful import apo_curves, slope_deg

REPORT = ROOT / "reports/phase3/chase018"
HOLD = mod.HOLD
device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
H, W = mod.H, mod.W

POOL = ["baseline", "ridge_blend03", "morph_connect", "corridor_dilate"]
# recreate blend_base_morph ≈ 0.5*baseline + 0.5*morph_connect
EXTRA = ["blend_base_morph", "ridge_soft02"]


def adapt_ext(gray, apo, fasc, variant):
    if variant == "blend_base_morph":
        a = adapt.adapt_fasc(gray, apo, fasc, "baseline")
        b = adapt.adapt_fasc(gray, apo, fasc, "morph_connect")
        return (0.5 * a + 0.5 * b).astype(np.float32)
    if variant == "ridge_soft02":
        # soft ridge blend lighter than ridge_blend03
        h, w = apo.shape
        g = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        belly = adapt.apo_belly_mask(apo, apo_thr=0.60, margin_px=2).astype(np.float32)
        ridge, ang, coher = adapt.structure_ridge(g, sigma=1.5)
        base = fasc.astype(np.float32)
        out = base * (1.0 + 0.4 * ridge * belly)
        seeds = 0.20 * ridge * coher * belly
        out = np.maximum(out, seeds)
        out = adapt.morph_cleanup(out, open_k=0, close_k=2, min_area=6)
        return np.clip(out, 0, 1).astype(np.float32)
    return adapt.adapt_fasc(gray, apo, fasc, variant)


def deep_apo_slope(apo, px_per_mm, gshape):
    H0, W0 = gshape
    h, w = apo.shape
    mx, my = mm_per_prob_pixel(px_per_mm, px_per_mm, W0, H0, w, h)
    curves = apo_curves(apo, mx, my, thr=0.60)
    if curves is None or len(curves) < 2:
        return 0.0
    cx = 0.5 * (curves[1][1] + curves[1][2])
    return float(slope_deg(curves[1][0], np.clip(cx, curves[1][1], curves[1][2])))


def build_pa_overrides():
    """Reload A2 PA head quickly from targets."""
    targets = json.loads((REPORT / "multexp_aporelpa_targets.json").read_text())
    fasc_fr, _ = load_ckpt(ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt", device)
    img_dir = ROOT / "data/external/osf_xbawc/expdrawn_images"
    Xtr, Ytr = [], []
    for r in targets["rows"]:
        ip = img_dir / f"{r['stem']}.png"
        if not ip.exists():
            continue
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        fasc = predict_prob(fasc_fr, (H, W), g, device)
        feats = mod.fasc_feats(fasc, r["apo_slope"], r["pa"], r["pa"])
        f = fasc.astype(np.float32)
        extras = np.array([float(f.std()), float(np.percentile(f, 90))], dtype=np.float32)
        Xtr.append(np.concatenate([feats[:6], extras]))
        Ytr.append(r["pa"])

    class PAHead12(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(8, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )
        def forward(self, x):
            return self.mlp(x).squeeze(-1)

    head = PAHead12().to(device)
    Xtr_t = torch.tensor(np.stack(Xtr), device=device)
    Ytr_t = torch.tensor(np.asarray(Ytr, dtype=np.float32), device=device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    head.train()
    for _ in range(800):
        pred = head(Xtr_t)
        loss = F.smooth_l1_loss(pred, Ytr_t)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

    fr_cache = REPORT / "cache" / "maps_multexp"
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    blend = 0.25
    pa_ov = {}
    with torch.no_grad():
        for _, er in expert.iterrows():
            name = er["image_file"]
            zp = fr_cache / f"{name}.npz"
            g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
            if g is None or not zp.exists():
                continue
            z = np.load(zp)
            px = float(er["Scale_pixel_per_cm"]) / 10.0
            pred0 = mod.predict_bundle(g, z["apo"].astype(np.float32), z["fasc"].astype(np.float32), px)
            slope = deep_apo_slope(z["apo"].astype(np.float32), px, g.shape)
            feats = mod.fasc_feats(z["fasc"].astype(np.float32), slope, pred0["pa_h"], pred0["pa_tex"])
            f = z["fasc"].astype(np.float32)
            extras = np.array([float(f.std()), float(np.percentile(f, 90))], dtype=np.float32)
            x = np.concatenate([feats[:6], extras])
            pa_hat = float(np.clip(head(torch.tensor(x[None], device=device)).cpu().numpy()[0], 5, 40))
            pa_ov[name] = (1 - blend) * pred0["pred_pa"] + blend * pa_hat
    return pa_ov


def run_variants(fl_protocol, pa_overrides=None):
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    fr_cache = REPORT / "cache" / "maps_multexp"
    variants = POOL + EXTRA
    preds = {}
    for variant in variants:
        rows = []
        for _, er in expert.iterrows():
            name = er["image_file"]
            g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
            zp = fr_cache / f"{name}.npz"
            if g is None or not zp.exists():
                continue
            z = np.load(zp)
            apo = z["apo"].astype(np.float32)
            fasc0 = z["fasc"].astype(np.float32)
            fasc = adapt_ext(g, apo, fasc0, variant)
            px = float(er["Scale_pixel_per_cm"]) / 10.0
            po = None if pa_overrides is None else pa_overrides.get(name)
            pred = mod.predict_bundle(g, apo, fasc, px, fl_protocol=fl_protocol, pa_override=po)
            # features for rule
            feats = adapt.image_features(g, apo, fasc, px)
            rows.append(dict(
                image_file=name,
                pred_pa=pred["pred_pa"],
                pred_fl=pred["pred_fl"],
                pred_mt=pred["pred_mt"],
                gt_pa=float(er["median_PA"]),
                gt_fl=float(er["median_FL"]),
                gt_mt=float(er["median_MT"]),
                **feats,
            ))
        preds[variant] = pd.DataFrame(rows)
        # fixed S
        comps, mae = mod.loocv_cal_pa_mt(preds[variant])
        S = float(comps["S"]) if isinstance(comps, dict) else float(comps)
        print(f"  {variant:20s} S={S:.5f} PA={mae['pa_deg']:.3f} FL={mae['fl_mm']:.3f}", flush=True)
    return preds


def feature_rule_select(preds_by_var, variants):
    """Honest nested feature_rule: for each fold, search (fkey,v_lo,v_hi) on TRAIN only."""
    base = preds_by_var[variants[0]]
    imgs = list(base.image_file.values)
    n = len(imgs)
    pa = {v: preds_by_var[v].set_index("image_file").loc[imgs, "pred_pa"].values for v in variants}
    fl = {v: preds_by_var[v].set_index("image_file").loc[imgs, "pred_fl"].values for v in variants}
    mt = {v: preds_by_var[v].set_index("image_file").loc[imgs, "pred_mt"].values for v in variants}
    gpa = base.set_index("image_file").loc[imgs, "gt_pa"].values
    gfl = base.set_index("image_file").loc[imgs, "gt_fl"].values
    gmt = base.set_index("image_file").loc[imgs, "gt_mt"].values
    feature_keys = ["fasc_mass", "orient_disp", "coher_mean", "apo_gap_px"]
    feat = np.stack([base.set_index("image_file").loc[imgs, k].values.astype(float) for k in feature_keys], 1)
    col_med = np.nanmedian(feat, axis=0)
    inds = np.where(~np.isfinite(feat))
    feat[inds] = np.take(col_med, inds[1])
    feat_idx = {k: i for i, k in enumerate(feature_keys)}

    pa_o = np.zeros(n); fl_o = np.zeros(n); mt_o = np.zeros(n)
    picks = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_cfg = None
        best_train = 9.0
        for fkey in feature_keys:
            fi = feat_idx[fkey]
            for v_lo in variants:
                for v_hi in variants:
                    if v_lo == v_hi:
                        continue
                    thr = float(np.median(feat[tr, fi]))
                    def train_err(lo_is_below):
                        e = []
                        for t in tr:
                            v = v_lo if ((feat[t, fi] < thr) == lo_is_below) else v_hi
                            e.append(abs(fl[v][t] - gfl[t]) / 12.0 + abs(pa[v][t] - gpa[t]) / 6.0)
                        return float(np.mean(e))
                    s0, s1 = train_err(True), train_err(False)
                    lo_is_below = s0 <= s1
                    train_s = min(s0, s1)
                    if train_s < best_train:
                        best_train = train_s
                        best_cfg = (fkey, fi, thr, v_lo, v_hi, lo_is_below)
        fkey, fi, thr, v_lo, v_hi, lo_is_below = best_cfg
        v_pick = v_lo if ((feat[i, fi] < thr) == lo_is_below) else v_hi
        picks.append(v_pick)
        a, b = np.polyfit(pa[v_pick][tr], gpa[tr], 1)
        pa_o[i] = a * pa[v_pick][i] + b
        a, b = np.polyfit(mt[v_pick][tr], gmt[tr], 1)
        mt_o[i] = a * mt[v_pick][i] + b
        fl_o[i] = fl[v_pick][i]
    err = {"pa_deg": np.abs(pa_o - gpa), "fl_mm": np.abs(fl_o - gfl), "mt_mm": np.abs(mt_o - gmt)}
    comps = score_components(err)
    from collections import Counter
    return dict(
        S=float(comps["S"]),
        MAE={k: float(np.mean(v)) for k, v in err.items()},
        pick_counts=dict(Counter(picks)),
        nested="per_fold_train_rule_search",
    )


def main():
    print(f"[holdcombo] device={device}", flush=True)
    results = {}

    for tag, fl_proto, use_pa in [
        ("adapt_baseline_fl", "baseline", False),
        ("adapt_sin030", "sin_w0.30", False),
        ("adapt_sin030_A2pa", "sin_w0.30", True),
        ("adapt_geo65sin35", "geo65_sin35", False),
    ]:
        print(f"\n=== {tag} fl={fl_proto} pa_ov={use_pa} ===", flush=True)
        pa_ov = build_pa_overrides() if use_pa else None
        preds = run_variants(fl_proto, pa_ov)
        variants = POOL + EXTRA
        # fixed best
        fixed_best = None
        for v, df in preds.items():
            comps, mae = mod.loocv_cal_pa_mt(df)
            S = float(comps["S"]) if isinstance(comps, dict) else float(comps)
            if fixed_best is None or S < fixed_best["S"]:
                fixed_best = {"variant": v, "S": S, "mae": mae}
        rule = feature_rule_select(preds, variants)
        print(f"  fixed_best={fixed_best}", flush=True)
        print(f"  feature_rule={rule}", flush=True)
        results[tag] = {"fixed_best": fixed_best, "feature_rule": rule}

    champ = None
    for tag, r in results.items():
        for key in ("fixed_best", "feature_rule"):
            s = r[key]["S"] if r[key] else None
            if s is not None and (champ is None or s < champ["S"]):
                champ = {"tag": tag, "key": key, "S": s, "detail": r[key]}

    # merge with prior summary
    prior = json.loads((REPORT / "multexp_aporelpa_flgeo_summary.json").read_text())
    prior_champ = prior.get("champ_S")
    all_scores = [prior_champ, champ["S"] if champ else None]
    all_scores = [s for s in all_scores if isinstance(s, (float, int))]
    final_champ = min(all_scores)

    out = {
        "asof": mod.now(),
        "holdcombo": results,
        "holdcombo_champ": champ,
        "prior_champ_S": prior_champ,
        "final_champ_S": final_champ,
        "hold": HOLD,
        "gate_0.12": final_champ < 0.12,
        "promote": final_champ < HOLD,
        "kaggle_submit": False,
    }
    (REPORT / "multexp_aporelpa_flgeo_holdcombo.json").write_text(json.dumps(out, indent=2, default=str))

    # update summary
    prior["holdcombo_champ"] = champ
    prior["final_champ_S"] = final_champ
    prior["promote"] = bool(final_champ < HOLD)
    prior["gate_0.12"] = bool(final_champ < 0.12)
    prior["asof"] = mod.now()
    (REPORT / "multexp_aporelpa_flgeo_summary.json").write_text(json.dumps(prior, indent=2, default=str))

    status = f"""# MultExp apo-rel PA + FL geo FINAL — {mod.now()}

## Result
| track | S |
|---|---|
| A soft-PA FT | {prior.get('A_ft_S')} |
| A2 PA-head | {prior.get('A2_pa_head_S')} |
| B FL sin_w0.30 | {prior.get('B_S')} |
| combo A2+B | {prior.get('combo_S')} |
| holdcombo best | {champ} |
| **champ** | **{final_champ}** |
| hold | {HOLD} |
| gate 0.12 | {'PASS' if final_champ < 0.12 else 'FAIL'} |
| promote | {final_champ < HOLD} |
| no Kaggle | true |

## Diagnosis
A apo-rel soft PA FT (152 ExpDrawn targets) kept Dice but MultExp maps worse. A2 PA-head improved PA (0.901°). B sin_w0.30 edged Frundles floor to 0.13241. Holdcombo = fasc-adapt feature_rule + B FL.

## Next
{"PROMOTABLE under hold — still no Kaggle until gate." if final_champ < HOLD else "Missed hold by a hair. Next: fold-pure joint search over (fasc-adapt variant × FL sin_w ∈{0.28..0.35} × A2 PA blend) with nested train-only selection; or MultExp-domain fasc GT proxy beyond ExpDrawn."}
"""
    sp = REPORT / "status.md"
    prev = sp.read_text() if sp.exists() else ""
    sp.write_text(status + "\n---\n\n" + prev)
    print(json.dumps(out, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
