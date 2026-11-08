#!/usr/bin/env python3
"""MultExp fasc adapt A/C v2: milder blends + nested-safe feature rules.

Honest LOOCV only. Outer hyperparams (which feature/pair) chosen fold-pure.
Frozen recipe. No Kaggle.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.calibration.anisotropic_scale import mm_per_prob_pixel  # noqa: E402
from umud.geometry.host_faithful import (  # noqa: E402
    apo_curves,
    aponeuroses_rich,
    fascicle_segments,
    line_curve_intersection,
    pa_depth_fit,
    slope_deg,
    weighted_median,
)
from umud.geometry.texture_angle import texture_angle  # noqa: E402
from umud.metric.score import score_components  # noqa: E402

# reuse adapt helpers from v1 by import
sys.path.insert(0, str(ROOT / "scripts/chase018"))
from adapt_multexp_fasc import (  # noqa: E402
    FROZEN,
    PRIOR_BEST,
    adapt_fasc,
    apo_belly_mask,
    image_features,
    loocv_cal_pa_mt,
    predict_one,
    structure_ridge,
)

HK = timezone(timedelta(hours=8))

# Milder + previous useful set
VARIANTS = [
    "baseline",
    "belly_gate",
    "ridge_blend03",
    "morph_connect",
    "corridor_dilate",
    # new mild
    "ridge_soft02",
    "ridge_soft015",
    "close3_belly",
    "blend_base_ridge03",
    "blend_base_morph",
]


def adapt_fasc_v2(gray_full, apo, fasc, variant: str) -> np.ndarray:
    if variant in (
        "baseline",
        "belly_gate",
        "ridge_blend03",
        "morph_connect",
        "corridor_dilate",
        "belly_orient15",
        "belly_orient25",
        "ridge_blend05",
        "ridge_replace",
        "morph_prune",
        "frangi_like",
    ):
        return adapt_fasc(gray_full, apo, fasc, variant)

    h, w = apo.shape
    g = cv2.resize(gray_full, (w, h), interpolation=cv2.INTER_AREA)
    belly = apo_belly_mask(apo, 0.60, 2).astype(np.float32)
    ridge, ang, coher = structure_ridge(g, 1.5)
    base = fasc.astype(np.float32)

    if variant == "ridge_soft02":
        out = np.maximum(base, 0.20 * ridge * coher * belly)
        return np.clip(out, 0, 1).astype(np.float32)
    if variant == "ridge_soft015":
        out = np.maximum(base, 0.15 * ridge * coher * belly)
        return np.clip(out, 0, 1).astype(np.float32)
    if variant == "close3_belly":
        out = base * (0.3 + 0.7 * belly)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = (out > 0.15).astype(np.uint8)
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        blurred = cv2.GaussianBlur(out, (0, 0), 0.8)
        fill = (m2 > 0) & (m == 0)
        out = np.where(fill, np.maximum(out, blurred * 0.85), out)
        return np.clip(out, 0, 1).astype(np.float32)
    if variant == "blend_base_ridge03":
        a = adapt_fasc(gray_full, apo, fasc, "baseline")
        b = adapt_fasc(gray_full, apo, fasc, "ridge_blend03")
        return (0.7 * a + 0.3 * b).astype(np.float32)
    if variant == "blend_base_morph":
        a = adapt_fasc(gray_full, apo, fasc, "baseline")
        b = adapt_fasc(gray_full, apo, fasc, "morph_connect")
        return (0.7 * a + 0.3 * b).astype(np.float32)
    raise ValueError(variant)


def run_all(expert, expert_dir, cache_dir, kw):
    preds = {}
    summaries = []
    for v in VARIANTS:
        rows = []
        for _, er in expert.iterrows():
            name_img = er["image_file"]
            g = cv2.imread(str(expert_dir / name_img), cv2.IMREAD_GRAYSCALE)
            zpath = cache_dir / f"{name_img}.npz"
            if g is None or not zpath.exists():
                continue
            z = np.load(zpath)
            apo = z["apo"].astype(np.float32)
            fasc0 = z["fasc"].astype(np.float32)
            fasc = adapt_fasc_v2(g, apo, fasc0, v)
            px = float(er["Scale_pixel_per_cm"]) / 10.0
            pred = predict_one(g, apo, fasc, px, kw)
            feat = image_features(g, apo, fasc0, px)
            rows.append(
                dict(
                    image_file=name_img,
                    ok=pred["ok"],
                    pred_pa=pred["pred_pa"],
                    pred_fl=pred["pred_fl"],
                    pred_mt=pred["pred_mt"],
                    gt_pa=float(er["median_PA"]),
                    gt_fl=float(er["median_FL"]),
                    gt_mt=float(er["median_MT"]),
                    **feat,
                )
            )
        df = pd.DataFrame(rows)
        comps, mae, _, _ = loocv_cal_pa_mt(df)
        s = {
            "name": v,
            "n": int(len(df)),
            "ok": int(df.ok.sum()),
            "S_loocv_cal_pa_mt": float(comps["S"]),
            "MAE_loocv": mae,
            "beats_0.13294": float(comps["S"]) < PRIOR_BEST,
            "gate_0.12": float(comps["S"]) < 0.12,
            "gap_to_0.12": float(comps["S"]) - 0.12,
        }
        summaries.append(s)
        preds[v] = df
        print(
            f"{v}: S={s['S_loocv_cal_pa_mt']:.5f} pa={mae['pa_deg']:.3f} "
            f"fl={mae['fl_mm']:.3f} mt={mae['mt_mm']:.3f} beats={s['beats_0.13294']}",
            flush=True,
        )
    return summaries, preds


def nested_feature_rule(preds, feature_keys, variants):
    """Fully nested: for each test i, search rule on train only (inner LOOCV S)."""
    base = preds["baseline"]
    imgs = list(base.image_file.values)
    n = len(imgs)
    pa = {v: preds[v].set_index("image_file").loc[imgs, "pred_pa"].values for v in variants}
    fl = {v: preds[v].set_index("image_file").loc[imgs, "pred_fl"].values for v in variants}
    mt = {v: preds[v].set_index("image_file").loc[imgs, "pred_mt"].values for v in variants}
    gpa = base.set_index("image_file").loc[imgs, "gt_pa"].values
    gfl = base.set_index("image_file").loc[imgs, "gt_fl"].values
    gmt = base.set_index("image_file").loc[imgs, "gt_mt"].values
    feat = np.stack(
        [base.set_index("image_file").loc[imgs, k].values.astype(float) for k in feature_keys],
        axis=1,
    )
    col_med = np.nanmedian(feat, axis=0)
    inds = np.where(~np.isfinite(feat))
    feat[inds] = np.take(col_med, inds[1])
    feat_idx = {k: i for i, k in enumerate(feature_keys)}

    def inner_S(tr_idx, fkey, v_lo, v_hi, lo_is_below, thr):
        """LOOCV S on tr_idx only using the rule."""
        fi = feat_idx[fkey]
        errs = []
        for t in tr_idx:
            sub = [j for j in tr_idx if j != t]
            if len(sub) < 3:
                continue
            v = v_lo if ((feat[t, fi] < thr) == lo_is_below) else v_hi
            a, b = np.polyfit(pa[v][sub], gpa[sub], 1)
            pa_hat = a * pa[v][t] + b
            a, b = np.polyfit(mt[v][sub], gmt[sub], 1)
            mt_hat = a * mt[v][t] + b
            errs.append(
                [
                    abs(pa_hat - gpa[t]) / 6.0,
                    abs(fl[v][t] - gfl[t]) / 12.0,
                    abs(mt_hat - gmt[t]) / 3.0,
                ]
            )
        if not errs:
            return 9.0
        return float(np.mean(np.array(errs).mean(axis=0)))

    # Restrict pair search to top variants by global single-S to keep cost down
    # but selection of which pair is still nested per fold using train only.
    # Candidate pool = all variants (n~10, pairs~90, features~4 -> ok for n=35)

    pa_o = np.zeros(n)
    fl_o = np.zeros(n)
    mt_o = np.zeros(n)
    picked_meta = []

    # Precompute candidate rules list
    rules = []
    for fkey in ("fasc_mass", "orient_disp", "coher_mean", "apo_gap_px"):
        for v_lo in variants:
            for v_hi in variants:
                if v_lo == v_hi:
                    continue
                for lo_is_below in (True, False):
                    rules.append((fkey, v_lo, v_hi, lo_is_below))

    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best = None
        best_s = 9.0
        for fkey, v_lo, v_hi, lo_is_below in rules:
            fi = feat_idx[fkey]
            thr = float(np.median(feat[tr, fi]))
            s = inner_S(tr, fkey, v_lo, v_hi, lo_is_below, thr)
            if s < best_s:
                best_s = s
                best = (fkey, v_lo, v_hi, lo_is_below, thr)
        fkey, v_lo, v_hi, lo_is_below, thr = best
        fi = feat_idx[fkey]
        v_pick = v_lo if ((feat[i, fi] < thr) == lo_is_below) else v_hi
        picked_meta.append({"i": i, "img": imgs[i], "v": v_pick, "rule": best, "train_S": best_s})
        a, b = np.polyfit(pa[v_pick][tr], gpa[tr], 1)
        pa_o[i] = a * pa[v_pick][i] + b
        a, b = np.polyfit(mt[v_pick][tr], gmt[tr], 1)
        mt_o[i] = a * mt[v_pick][i] + b
        fl_o[i] = fl[v_pick][i]

    err = {
        "pa_deg": np.abs(pa_o - gpa),
        "fl_mm": np.abs(fl_o - gfl),
        "mt_mm": np.abs(mt_o - gmt),
    }
    comps = score_components(err)
    mae = {k: float(np.mean(v)) for k, v in err.items()}
    counts = {}
    for p in picked_meta:
        counts[p["v"]] = counts.get(p["v"], 0) + 1
    return {
        "S_loocv_cal_pa_mt": float(comps["S"]),
        "MAE_loocv": mae,
        "picked_counts": counts,
        "gate_0.12": float(comps["S"]) < 0.12,
        "beats_0.13294": float(comps["S"]) < PRIOR_BEST,
        "gap_to_0.12": float(comps["S"]) - 0.12,
        "n_rules_searched": len(rules),
        "sample_picks": picked_meta[:5],
    }


def nested_trainS_pick(preds, variants):
    base = preds["baseline"]
    imgs = list(base.image_file.values)
    n = len(imgs)
    pa = {v: preds[v].set_index("image_file").loc[imgs, "pred_pa"].values for v in variants}
    fl = {v: preds[v].set_index("image_file").loc[imgs, "pred_fl"].values for v in variants}
    mt = {v: preds[v].set_index("image_file").loc[imgs, "pred_mt"].values for v in variants}
    gpa = base.set_index("image_file").loc[imgs, "gt_pa"].values
    gfl = base.set_index("image_file").loc[imgs, "gt_fl"].values
    gmt = base.set_index("image_file").loc[imgs, "gt_mt"].values

    def train_S(v, tr):
        errs = []
        for t in tr:
            sub = [j for j in tr if j != t]
            if len(sub) < 3:
                continue
            a, b = np.polyfit(pa[v][sub], gpa[sub], 1)
            pa_hat = a * pa[v][t] + b
            a, b = np.polyfit(mt[v][sub], gmt[sub], 1)
            mt_hat = a * mt[v][t] + b
            errs.append(
                [
                    abs(pa_hat - gpa[t]) / 6.0,
                    abs(fl[v][t] - gfl[t]) / 12.0,
                    abs(mt_hat - gmt[t]) / 3.0,
                ]
            )
        return float(np.mean(np.array(errs).mean(axis=0))) if errs else 9.0

    pa_o = np.zeros(n)
    fl_o = np.zeros(n)
    mt_o = np.zeros(n)
    picked = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_v, best_s = None, 9.0
        for v in variants:
            s = train_S(v, tr)
            if s < best_s:
                best_s, best_v = s, v
        picked.append(best_v)
        a, b = np.polyfit(pa[best_v][tr], gpa[tr], 1)
        pa_o[i] = a * pa[best_v][i] + b
        a, b = np.polyfit(mt[best_v][tr], gmt[tr], 1)
        mt_o[i] = a * mt[best_v][i] + b
        fl_o[i] = fl[best_v][i]
    err = {
        "pa_deg": np.abs(pa_o - gpa),
        "fl_mm": np.abs(fl_o - gfl),
        "mt_mm": np.abs(mt_o - gmt),
    }
    comps = score_components(err)
    return {
        "S_loocv_cal_pa_mt": float(comps["S"]),
        "MAE_loocv": {k: float(np.mean(v)) for k, v in err.items()},
        "picked_counts": {v: picked.count(v) for v in variants},
        "gate_0.12": float(comps["S"]) < 0.12,
        "beats_0.13294": float(comps["S"]) < PRIOR_BEST,
    }


def oracle(preds, variants):
    base = preds["baseline"]
    imgs = list(base.image_file.values)
    n = len(imgs)
    gpa = base.set_index("image_file").loc[imgs, "gt_pa"].values
    gfl = base.set_index("image_file").loc[imgs, "gt_fl"].values
    gmt = base.set_index("image_file").loc[imgs, "gt_mt"].values
    oracle_pa = np.zeros(n)
    oracle_fl = np.zeros(n)
    oracle_mt = np.zeros(n)
    for i, im in enumerate(imgs):
        best_e, bp, bf, bm = 9.0, None, None, None
        for v in variants:
            r = preds[v].set_index("image_file").loc[im]
            e = abs(r.pred_fl - gfl[i]) / 12.0 + abs(r.pred_pa - gpa[i]) / 6.0
            if e < best_e:
                best_e = e
                bp, bf, bm = float(r.pred_pa), float(r.pred_fl), float(r.pred_mt)
        oracle_pa[i], oracle_fl[i], oracle_mt[i] = bp, bf, bm
    pa_o = np.zeros(n)
    mt_o = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        a, b = np.polyfit(oracle_pa[m], gpa[m], 1)
        pa_o[i] = a * oracle_pa[i] + b
        a, b = np.polyfit(oracle_mt[m], gmt[m], 1)
        mt_o[i] = a * oracle_mt[i] + b
    err = {
        "pa_deg": np.abs(pa_o - gpa),
        "fl_mm": np.abs(oracle_fl - gfl),
        "mt_mm": np.abs(mt_o - gmt),
    }
    comps = score_components(err)
    return {
        "S_loocv_cal_pa_mt": float(comps["S"]),
        "MAE_loocv": {k: float(np.mean(v)) for k, v in err.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--expert-csv",
        type=Path,
        default=ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv",
    )
    ap.add_argument(
        "--expert-dir",
        type=Path,
        default=ROOT
        / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "reports/phase3/chase018/cache/maps_multexp",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports/phase3/chase018",
    )
    args = ap.parse_args()

    expert = pd.read_csv(args.expert_csv)
    kw = dict(FROZEN)
    print("running variants...", flush=True)
    summaries, preds = run_all(expert, args.expert_dir, args.cache_dir, kw)
    ranked = sorted(summaries, key=lambda r: r["S_loocv_cal_pa_mt"])
    feat_keys = [
        "fasc_mass",
        "apo_mass",
        "belly_frac",
        "apo_gap_px",
        "coher_mean",
        "orient_disp",
        "scale",
    ]
    print("nested trainS pick...", flush=True)
    pick = nested_trainS_pick(preds, VARIANTS)
    print("trainS", pick, flush=True)
    print("nested feature rule (may take a bit)...", flush=True)
    rule = nested_feature_rule(preds, feat_keys, VARIANTS)
    print("rule", {k: rule[k] for k in rule if k != "sample_picks"}, flush=True)
    ora = oracle(preds, VARIANTS)
    print("oracle", ora, flush=True)

    # Also report the LEAKY (diagnostic) best fixed rule from v1-style for comparison
    # but do NOT promote it.
    leaky_best = None

    candidates = [
        {"name": r["name"], **{k: r[k] for k in ("S_loocv_cal_pa_mt", "MAE_loocv", "beats_0.13294", "gate_0.12", "gap_to_0.12")}}
        for r in ranked
    ]
    candidates.append({"name": "nested_trainS_pick", **pick, "gap_to_0.12": pick["S_loocv_cal_pa_mt"] - 0.12})
    candidates.append(
        {
            "name": "nested_feature_rule",
            "S_loocv_cal_pa_mt": rule["S_loocv_cal_pa_mt"],
            "MAE_loocv": rule["MAE_loocv"],
            "beats_0.13294": rule["beats_0.13294"],
            "gate_0.12": rule["gate_0.12"],
            "gap_to_0.12": rule["gap_to_0.12"],
            "picked_counts": rule["picked_counts"],
        }
    )
    best = min(candidates, key=lambda c: c["S_loocv_cal_pa_mt"])

    # NOTE on v1 leaky result
    v1_leaky = {
        "name": "v1_leaky_feature_rule_fasc_mass",
        "S_reported": 0.12962987263663162,
        "note": "NOT honest: outer rule pair chosen on full LOOCV; do not promote",
    }

    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "protocol": "honest LOOCV affine PA+MT; FL raw; frozen recipe; nested-safe C",
        "frozen_recipe": FROZEN,
        "prior_best_S": PRIOR_BEST,
        "variants_ranked": ranked,
        "nested_trainS_pick": pick,
        "nested_feature_rule": {k: rule[k] for k in rule if k != "sample_picks"},
        "oracle_among_A": ora,
        "v1_leaky_diagnostic": v1_leaky,
        "best_honest": best,
        "gate_0.12": bool(best["S_loocv_cal_pa_mt"] < 0.12),
        "promote": bool(best["S_loocv_cal_pa_mt"] < PRIOR_BEST),
        "kaggle_submit": False,
        "lab_fasc_ckpt": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "multexp_fasc_adapt_v2_scores.json"
    out_json.write_text(json.dumps(payload, indent=2))
    # update main adapt scores to include honesty note
    main_json = args.out_dir / "multexp_fasc_adapt_scores.json"
    if main_json.exists():
        prev = json.loads(main_json.read_text())
        prev["honesty_note"] = (
            "v1 feature_rule S=0.12963 was LEAKY (outer rule search on full LOOCV). "
            "See multexp_fasc_adapt_v2_scores.json for nested-safe C."
        )
        prev["v2_path"] = str(out_json)
        prev["v2_best_honest"] = best
        main_json.write_text(json.dumps(prev, indent=2))

    status = []
    status.append(f"# MultExp fasc adaptation A/C v2 — {payload['asof']}")
    status.append("")
    status.append(f"- Prior best S: **{PRIOR_BEST:.5f}**")
    status.append(
        f"- Best honest: **{best['name']}** S=**{best['S_loocv_cal_pa_mt']:.5f}** "
        f"PA {best['MAE_loocv']['pa_deg']:.3f} / FL {best['MAE_loocv']['fl_mm']:.3f} / "
        f"MT {best['MAE_loocv']['mt_mm']:.3f}"
    )
    status.append(
        f"- Beats prior: {best.get('beats_0.13294')} | gate 0.12: {best.get('gate_0.12')} | "
        f"promote: {payload['promote']}"
    )
    status.append(f"- Nested feature_rule S={rule['S_loocv_cal_pa_mt']:.5f} counts={rule['picked_counts']}")
    status.append(f"- Nested trainS_pick S={pick['S_loocv_cal_pa_mt']:.5f}")
    status.append(f"- Oracle among A: S={ora['S_loocv_cal_pa_mt']:.5f}")
    status.append(
        "- v1 leaky feature_rule 0.12963 is **not** promoted (outer search leak)."
    )
    status.append("")
    status.append("## Ranked single variants")
    for r in ranked:
        status.append(
            f"- {r['S_loocv_cal_pa_mt']:.5f} `{r['name']}` "
            f"pa={r['MAE_loocv']['pa_deg']:.3f} fl={r['MAE_loocv']['fl_mm']:.3f} "
            f"mt={r['MAE_loocv']['mt_mm']:.3f}"
        )
    status.append("")
    if payload["promote"]:
        status.append("Honest promotion candidate found.")
    else:
        status.append(
            f"No honest promotion. Gap to 0.12 ≈ {best['S_loocv_cal_pa_mt']-0.12:.5f}. "
            "A/C post-filters plateau; oracle ~0.10 shows headroom but selectors fail to transfer. "
            "Next: short MultExp-domain fasc fine-tune (B) with geometry-aware loss, nested-safe."
        )
    (args.out_dir / "multexp_fasc_adapt_status.md").write_text("\n".join(status) + "\n")
    print("wrote", out_json, flush=True)
    print(
        f"BEST_HONEST {best['name']} S={best['S_loocv_cal_pa_mt']:.5f} "
        f"promote={payload['promote']} gate={payload['gate_0.12']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
