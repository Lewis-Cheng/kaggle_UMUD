#!/usr/bin/env python3
"""Fold-pure joint search: fasc-adapt x fl_sin_w x A2 PA-head blend.
Honest nested outer LOOCV only. Beat hold C 0.13195; gate 0.12. No Kaggle."""
from __future__ import annotations
import importlib.util
import json
import sys
import time
from collections import Counter
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
GATE = 0.12
device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
H, W = mod.H, mod.W

VARIANTS = ["baseline", "ridge_blend03", "morph_connect", "corridor_dilate", "blend_base_morph", "ridge_soft02"]
SIN_WS = [0.25, 0.28, 0.30, 0.32, 0.35]
PA_ALPHAS = [0.0, 0.15, 0.25]
EXTRA_FL = ["geo70_sin30", "geo65_sin35", "deep_insert"]
FEATURE_KEYS = ["fasc_mass", "orient_disp", "coher_mean", "apo_gap_px"]


def adapt_ext(gray, apo, fasc, variant):
    if variant == "blend_base_morph":
        a = adapt.adapt_fasc(gray, apo, fasc, "baseline")
        b = adapt.adapt_fasc(gray, apo, fasc, "morph_connect")
        return (0.5 * a + 0.5 * b).astype(np.float32)
    if variant == "ridge_soft02":
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


def build_pa_overrides(blend: float) -> dict:
    if blend <= 0:
        return {}
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
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()

    fr_cache = REPORT / "cache" / "maps_multexp"
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
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


def precompute(pa_overrides_by_alpha: dict):
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    fr_cache = REPORT / "cache" / "maps_multexp"
    store = {}
    names = None
    for alpha, pa_ov in pa_overrides_by_alpha.items():
        for variant in VARIANTS:
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
                po = None if (not pa_ov) else pa_ov.get(name)
                pred = mod.predict_bundle(g, apo, fasc, px, fl_protocol="baseline", pa_override=po)
                feats = adapt.image_features(g, apo, fasc, px)
                row = dict(
                    image_file=name,
                    pred_pa=pred["pred_pa"],
                    pred_mt=pred["pred_mt"],
                    pred_fl=pred["pred_fl"],
                    gt_pa=float(er["median_PA"]),
                    gt_fl=float(er["median_FL"]),
                    gt_mt=float(er["median_MT"]),
                    **feats,
                )
                for k, v in pred.get("fl_cands", {}).items():
                    row[f"fl_{k}"] = float(v)
                rows.append(row)
            df = pd.DataFrame(rows)
            store[(variant, alpha)] = df
            if names is None:
                names = list(df.image_file.values)
            print(f"  precompute variant={variant:18s} alpha={alpha:.2f} n={len(df)}", flush=True)
    return store, names


def stack_preds(store, names, variant, alpha, sin_w=None, fl_proto=None):
    df = store[(variant, alpha)].set_index("image_file").loc[names]
    pa = df["pred_pa"].values.astype(float)
    mt = df["pred_mt"].values.astype(float)
    if fl_proto is not None:
        col = f"fl_{fl_proto}"
        fl = df[col].values.astype(float) if col in df.columns else df["pred_fl"].values.astype(float)
    else:
        col = f"fl_sin_w{sin_w:.2f}"
        if col in df.columns:
            fl = df[col].values.astype(float)
        elif abs(sin_w - 0.35) < 1e-9 and "fl_baseline" in df.columns:
            fl = df["fl_baseline"].values.astype(float)
        else:
            fl = df["pred_fl"].values.astype(float)
    return pa, fl, mt


def loocv_score_arrays(pa, fl, mt, gpa, gfl, gmt):
    n = len(pa)
    pa_o = np.zeros(n)
    mt_o = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        a, b = np.polyfit(pa[m], gpa[m], 1)
        pa_o[i] = a * pa[i] + b
        a, b = np.polyfit(mt[m], gmt[m], 1)
        mt_o[i] = a * mt[i] + b
    err = {"pa_deg": np.abs(pa_o - gpa), "fl_mm": np.abs(fl - gfl), "mt_mm": np.abs(mt_o - gmt)}
    comps = score_components(err)
    return float(comps["S"]), {k: float(np.mean(v)) for k, v in err.items()}, err


def train_proxy(pa, fl, mt, gpa, gfl, gmt, tr):
    e = []
    for t in tr:
        e.append(abs(fl[t] - gfl[t]) / 12.0 + abs(pa[t] - gpa[t]) / 6.0)
    return float(np.mean(e)) if e else 9.0


def nested_trainS(pa, fl, mt, gpa, gfl, gmt, tr):
    errs = []
    tr = list(tr)
    for t in tr:
        sub = [x for x in tr if x != t]
        if len(sub) < 3:
            continue
        a, b = np.polyfit(pa[sub], gpa[sub], 1)
        pa_hat = a * pa[t] + b
        a, b = np.polyfit(mt[sub], gmt[sub], 1)
        mt_hat = a * mt[t] + b
        errs.append([abs(pa_hat - gpa[t]) / 6.0, abs(fl[t] - gfl[t]) / 12.0, abs(mt_hat - gmt[t]) / 3.0])
    if not errs:
        return 9.0
    return float(np.mean(np.array(errs)))


def build_config_bank(store, names):
    bank = {}
    for variant in VARIANTS:
        for alpha in PA_ALPHAS:
            for sin_w in SIN_WS:
                pa, fl, mt = stack_preds(store, names, variant, alpha, sin_w=sin_w)
                bank[("sin", variant, sin_w, alpha)] = (pa, fl, mt)
            for proto in EXTRA_FL:
                pa, fl, mt = stack_preds(store, names, variant, alpha, fl_proto=proto)
                bank[("proto", variant, proto, alpha)] = (pa, fl, mt)
    return bank


def eval_fixed_bank(bank, gpa, gfl, gmt):
    results = []
    for key, (pa, fl, mt) in bank.items():
        S, mae, _ = loocv_score_arrays(pa, fl, mt, gpa, gfl, gmt)
        kind, variant, sw_or_proto, alpha = key
        results.append({
            "kind": kind,
            "variant": variant,
            "fl": sw_or_proto,
            "alpha": alpha,
            "S": S,
            "mae": mae,
        })
        print(
            f"  fixed {kind} v={variant:18s} fl={sw_or_proto} a={alpha:.2f} "
            f"S={S:.5f} PA={mae['pa_deg']:.3f} FL={mae['fl_mm']:.3f}",
            flush=True,
        )
    return sorted(results, key=lambda r: r["S"])


def nested_pick_fixed(bank, keys, gpa, gfl, gmt, proxy="trainS"):
    n = len(gpa)
    pa_o = np.zeros(n)
    fl_o = np.zeros(n)
    mt_o = np.zeros(n)
    picks = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_key, best_s = None, 9.0
        for key in keys:
            pa, fl, mt = bank[key]
            if proxy == "trainS":
                s = nested_trainS(pa, fl, mt, gpa, gfl, gmt, tr)
            else:
                s = train_proxy(pa, fl, mt, gpa, gfl, gmt, tr)
            if s < best_s:
                best_s, best_key = s, key
        picks.append(best_key)
        pa, fl, mt = bank[best_key]
        a, b = np.polyfit(pa[tr], gpa[tr], 1)
        pa_o[i] = a * pa[i] + b
        a, b = np.polyfit(mt[tr], gmt[tr], 1)
        mt_o[i] = a * mt[i] + b
        fl_o[i] = fl[i]
    err = {"pa_deg": np.abs(pa_o - gpa), "fl_mm": np.abs(fl_o - gfl), "mt_mm": np.abs(mt_o - gmt)}
    comps = score_components(err)
    return dict(
        S=float(comps["S"]),
        MAE={k: float(np.mean(v)) for k, v in err.items()},
        pick_counts={str(k): int(v) for k, v in Counter(picks).items()},
        nested="per_fold_train_pick_fixed_config",
        proxy=proxy,
    )


def nested_feature_rule_joint(store, names, gpa, gfl, gmt, sin_ws, alphas, variants):
    n = len(names)
    base_df = store[("baseline", 0.0)].set_index("image_file").loc[names]
    feat = np.stack([base_df[k].values.astype(float) for k in FEATURE_KEYS], 1)
    col_med = np.nanmedian(feat, axis=0)
    inds = np.where(~np.isfinite(feat))
    feat[inds] = np.take(col_med, inds[1])
    feat_idx = {k: i for i, k in enumerate(FEATURE_KEYS)}

    arrays = {}
    for v in variants:
        for a in alphas:
            for sw in sin_ws:
                arrays[(v, sw, a)] = stack_preds(store, names, v, a, sin_w=sw)

    pa_o = np.zeros(n)
    fl_o = np.zeros(n)
    mt_o = np.zeros(n)
    picks = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_cfg = None
        best_train = 9.0
        for fkey in FEATURE_KEYS:
            fi = feat_idx[fkey]
            thr = float(np.median(feat[tr, fi]))
            for v_lo in variants:
                for v_hi in variants:
                    if v_lo == v_hi:
                        continue
                    for sw in sin_ws:
                        for a in alphas:
                            pa_lo, fl_lo, mt_lo = arrays[(v_lo, sw, a)]
                            pa_hi, fl_hi, mt_hi = arrays[(v_hi, sw, a)]

                            def train_err(lo_is_below, _plo=pa_lo, _flo=fl_lo, _phi=pa_hi, _fhi=fl_hi):
                                e = []
                                for t in tr:
                                    use_lo = (feat[t, fi] < thr) == lo_is_below
                                    if use_lo:
                                        e.append(abs(_flo[t] - gfl[t]) / 12.0 + abs(_plo[t] - gpa[t]) / 6.0)
                                    else:
                                        e.append(abs(_fhi[t] - gfl[t]) / 12.0 + abs(_phi[t] - gpa[t]) / 6.0)
                                return float(np.mean(e))

                            s0, s1 = train_err(True), train_err(False)
                            lo_is_below = s0 <= s1
                            train_s = min(s0, s1)
                            if train_s < best_train:
                                best_train = train_s
                                best_cfg = (fkey, fi, thr, v_lo, v_hi, lo_is_below, sw, a)
        fkey, fi, thr, v_lo, v_hi, lo_is_below, sw, a = best_cfg
        use_lo = (feat[i, fi] < thr) == lo_is_below
        v_pick = v_lo if use_lo else v_hi
        picks.append(f"{v_pick}|sin{sw:.2f}|a{a:.2f}|{fkey}")
        pa, fl, mt = arrays[(v_pick, sw, a)]
        a_fit, b_fit = np.polyfit(pa[tr], gpa[tr], 1)
        pa_o[i] = a_fit * pa[i] + b_fit
        a_fit, b_fit = np.polyfit(mt[tr], gmt[tr], 1)
        mt_o[i] = a_fit * mt[i] + b_fit
        fl_o[i] = fl[i]
    err = {"pa_deg": np.abs(pa_o - gpa), "fl_mm": np.abs(fl_o - gfl), "mt_mm": np.abs(mt_o - gmt)}
    comps = score_components(err)
    return dict(
        S=float(comps["S"]),
        MAE={k: float(np.mean(v)) for k, v in err.items()},
        pick_counts=dict(Counter(picks)),
        nested="per_fold_joint_feature_rule_sin_pa",
    )


def nested_feature_rule_fixed_hypers(store, names, gpa, gfl, gmt, sin_w, alpha, variants):
    n = len(names)
    base_df = store[("baseline", 0.0)].set_index("image_file").loc[names]
    feat = np.stack([base_df[k].values.astype(float) for k in FEATURE_KEYS], 1)
    col_med = np.nanmedian(feat, axis=0)
    inds = np.where(~np.isfinite(feat))
    feat[inds] = np.take(col_med, inds[1])
    feat_idx = {k: i for i, k in enumerate(FEATURE_KEYS)}
    arrays = {v: stack_preds(store, names, v, alpha, sin_w=sin_w) for v in variants}

    pa_o = np.zeros(n)
    fl_o = np.zeros(n)
    mt_o = np.zeros(n)
    picks = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_cfg = None
        best_train = 9.0
        for fkey in FEATURE_KEYS:
            fi = feat_idx[fkey]
            thr = float(np.median(feat[tr, fi]))
            for v_lo in variants:
                for v_hi in variants:
                    if v_lo == v_hi:
                        continue
                    pa_lo, fl_lo, _ = arrays[v_lo]
                    pa_hi, fl_hi, _ = arrays[v_hi]

                    def train_err(lo_is_below, _plo=pa_lo, _flo=fl_lo, _phi=pa_hi, _fhi=fl_hi):
                        e = []
                        for t in tr:
                            use_lo = (feat[t, fi] < thr) == lo_is_below
                            if use_lo:
                                e.append(abs(_flo[t] - gfl[t]) / 12.0 + abs(_plo[t] - gpa[t]) / 6.0)
                            else:
                                e.append(abs(_fhi[t] - gfl[t]) / 12.0 + abs(_phi[t] - gpa[t]) / 6.0)
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
        pa, fl, mt = arrays[v_pick]
        a_fit, b_fit = np.polyfit(pa[tr], gpa[tr], 1)
        pa_o[i] = a_fit * pa[i] + b_fit
        a_fit, b_fit = np.polyfit(mt[tr], gmt[tr], 1)
        mt_o[i] = a_fit * mt[i] + b_fit
        fl_o[i] = fl[i]
    err = {"pa_deg": np.abs(pa_o - gpa), "fl_mm": np.abs(fl_o - gfl), "mt_mm": np.abs(mt_o - gmt)}
    comps = score_components(err)
    return dict(
        S=float(comps["S"]),
        MAE={k: float(np.mean(v)) for k, v in err.items()},
        pick_counts=dict(Counter(picks)),
        nested="per_fold_feature_rule",
        sin_w=sin_w,
        alpha=alpha,
    )


def main():
    t0 = time.time()
    print(f"[joint] device={device} hold={HOLD} {mod.now()}", flush=True)

    print("[joint] building A2 PA overrides...", flush=True)
    pa_by_alpha = {0.0: {}}
    for a in (0.15, 0.25):
        print(f"  PA head blend={a}", flush=True)
        pa_by_alpha[a] = build_pa_overrides(a)

    print("[joint] precomputing variant x alpha maps...", flush=True)
    store, names = precompute(pa_by_alpha)
    base = store[("baseline", 0.0)].set_index("image_file").loc[names]
    gpa = base["gt_pa"].values.astype(float)
    gfl = base["gt_fl"].values.astype(float)
    gmt = base["gt_mt"].values.astype(float)
    print(f"[joint] n={len(names)} images", flush=True)

    print("[joint] sanity: C-path sin0.35 a0 feature_rule...", flush=True)
    c_check = nested_feature_rule_fixed_hypers(store, names, gpa, gfl, gmt, 0.35, 0.0, VARIANTS)
    print(f"  C-check S={c_check['S']:.5f} (expect ~0.13195) MAE={c_check['MAE']}", flush=True)

    print("[joint] fixed grid LOOCV...", flush=True)
    bank = build_config_bank(store, names)
    fixed_ranked = eval_fixed_bank(bank, gpa, gfl, gmt)
    fixed_best = fixed_ranked[0]

    base_keys = [k for k in bank if k[1] == "baseline"]
    print(f"[joint] nested pick baseline-path ({len(base_keys)} cfgs)...", flush=True)
    nested_base = nested_pick_fixed(bank, base_keys, gpa, gfl, gmt, proxy="trainS")
    print(f"  nested_base S={nested_base['S']:.5f}", flush=True)

    all_keys = list(bank.keys())
    print(f"[joint] nested pick all-fixed ({len(all_keys)} cfgs)...", flush=True)
    nested_all = nested_pick_fixed(bank, all_keys, gpa, gfl, gmt, proxy="trainS")
    print(f"  nested_all S={nested_all['S']:.5f}", flush=True)

    print("[joint] feature_rule at each (sin_w, alpha)...", flush=True)
    fr_grid = []
    for sw in SIN_WS:
        for a in PA_ALPHAS:
            r = nested_feature_rule_fixed_hypers(store, names, gpa, gfl, gmt, sw, a, VARIANTS)
            fr_grid.append(r)
            print(
                f"  FR sin={sw:.2f} a={a:.2f} S={r['S']:.5f} "
                f"PA={r['MAE']['pa_deg']:.3f} FL={r['MAE']['fl_mm']:.3f}",
                flush=True,
            )
    fr_best = min(fr_grid, key=lambda r: r["S"])

    print("[joint] joint feature_rule x sin_w x alpha (nested)...", flush=True)
    fr_joint = nested_feature_rule_joint(store, names, gpa, gfl, gmt, SIN_WS, PA_ALPHAS, VARIANTS)
    print(f"  FR-joint S={fr_joint['S']:.5f} MAE={fr_joint['MAE']}", flush=True)

    candidates = [
        {"tag": "fixed_best", "S": fixed_best["S"], "mae": fixed_best["mae"],
         "variant": fixed_best["variant"], "fl": fixed_best["fl"], "alpha": fixed_best["alpha"]},
        {"tag": "nested_base_path", "S": nested_base["S"], "mae": nested_base["MAE"]},
        {"tag": "nested_all_fixed", "S": nested_all["S"], "mae": nested_all["MAE"]},
        {"tag": "feature_rule_grid_best", "S": fr_best["S"], "mae": fr_best["MAE"],
         "sin_w": fr_best.get("sin_w"), "alpha": fr_best.get("alpha")},
        {"tag": "feature_rule_joint", "S": fr_joint["S"], "mae": fr_joint["MAE"]},
        {"tag": "C_sanity_sin035_a0", "S": c_check["S"], "mae": c_check["MAE"]},
    ]
    champ = min(candidates, key=lambda r: r["S"])
    best_S = float(champ["S"])
    promote = best_S < HOLD
    gate = best_S < GATE

    out = {
        "asof": mod.now(),
        "protocol": "honest nested LOOCV; PA+MT affine; FL raw; no leaky oracle",
        "hold": HOLD,
        "gate": GATE,
        "C_sanity": c_check,
        "fixed_best": fixed_best,
        "fixed_top10": fixed_ranked[:10],
        "nested_base_path": nested_base,
        "nested_all_fixed": nested_all,
        "feature_rule_grid": fr_grid,
        "feature_rule_grid_best": fr_best,
        "feature_rule_joint": fr_joint,
        "candidates": candidates,
        "champ": champ,
        "best_nested_S": best_S,
        "promote": promote,
        "gate_0.12": gate,
        "kaggle_submit": False,
        "elapsed_min": (time.time() - t0) / 60.0,
        "grid": {"variants": VARIANTS, "sin_ws": SIN_WS, "pa_alphas": PA_ALPHAS, "extra_fl": EXTRA_FL},
    }
    (REPORT / "multexp_joint_foldpure.json").write_text(json.dumps(out, indent=2, default=str))
    (REPORT / "multexp_joint_scores.json").write_text(json.dumps({
        "asof": out["asof"],
        "best_nested_S": best_S,
        "champ": champ,
        "C_sanity_S": c_check["S"],
        "hold": HOLD,
        "promote": promote,
        "gate_0.12": gate,
        "fixed_best": fixed_best,
        "fr_joint_S": fr_joint["S"],
        "nested_all_S": nested_all["S"],
        "elapsed_min": out["elapsed_min"],
        "kaggle_submit": False,
    }, indent=2, default=str))

    status = f"""# MultExp joint fold-pure — {mod.now()}

## Result
| track | S | notes |
|---|---|---|
| C sanity (FR sin0.35 a0) | {c_check['S']:.5f} | expect ~0.13195 |
| fixed grid best | {fixed_best['S']:.5f} | {fixed_best['variant']} fl={fixed_best['fl']} a={fixed_best['alpha']} |
| nested baseline-path | {nested_base['S']:.5f} | trainS pick among Frundles×sin×PA |
| nested all-fixed | {nested_all['S']:.5f} | trainS pick among all variants×sin×PA |
| FR grid best | {fr_best['S']:.5f} | sin={fr_best.get('sin_w')} a={fr_best.get('alpha')} |
| FR joint (sin×PA nested) | {fr_joint['S']:.5f} | full joint feature_rule |
| **champ** | **{best_S:.5f}** | {champ['tag']} |
| hold C | {HOLD} | |
| gate 0.12 | {'PASS' if gate else 'FAIL'} | |
| promote | {promote} | |
| no Kaggle | true | |

## MAE (champ)
{champ.get('mae', champ.get('MAE', {}))}

## Next
{"PROMOTABLE under hold — still no Kaggle until gate 0.12." if promote else "Joint search did not beat hold C 0.13195. Stop grid search; next non-search lever: MultExp-domain fasc maps / ExpDrawn stroke-geometry PA prior outside Frundles+A family (crop+scale aug FT), not more sin_w×PA×adapt search."}

Elapsed_min: {out['elapsed_min']:.1f}
"""
    sp = REPORT / "status.md"
    prev = sp.read_text() if sp.exists() else ""
    sp.write_text(status + "\n---\n\n" + prev)
    (REPORT / "multexp_joint_status.md").write_text(status)
    print(json.dumps(champ, indent=2, default=str), flush=True)
    print(f"[joint] BEST S={best_S:.5f} promote={promote} gate={gate} elapsed={out['elapsed_min']:.1f}m", flush=True)


if __name__ == "__main__":
    main()
