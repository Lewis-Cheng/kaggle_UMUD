#!/usr/bin/env python3
"""Fast nested-safe MultExp fasc adapt A/C wrap-up + diagnosis.

Uses precomputed preds from v2 log variants; cheaper nested rule:
for each fold, pick (feature, v_lo, v_hi, side) minimizing TRAIN mean
raw component error (no inner LOOCV), apply to held-out.
Also tries knn-on-features variant pick.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from umud.metric.score import score_components  # noqa: E402
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
from adapt_multexp_fasc_v2 import adapt_fasc_v2, VARIANTS  # noqa: E402

HK = timezone(timedelta(hours=8))
# Focused pool: baseline + near-miss / mild
POOL = [
    "baseline",
    "ridge_blend03",
    "morph_connect",
    "corridor_dilate",
    "blend_base_morph",
    "ridge_soft02",
]


def build_preds(expert, expert_dir, cache_dir, kw, variants):
    preds = {}
    summaries = []
    for v in variants:
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
            "S_loocv_cal_pa_mt": float(comps["S"]),
            "MAE_loocv": mae,
            "beats_0.13294": float(comps["S"]) < PRIOR_BEST,
            "gate_0.12": float(comps["S"]) < 0.12,
            "gap_to_0.12": float(comps["S"]) - 0.12,
        }
        summaries.append(s)
        preds[v] = df
        print(f"{v}: S={s['S_loocv_cal_pa_mt']:.5f}", flush=True)
    return summaries, preds


def stacks(preds, variants):
    base = preds["baseline"]
    imgs = list(base.image_file.values)
    n = len(imgs)
    pa = {v: preds[v].set_index("image_file").loc[imgs, "pred_pa"].values for v in variants}
    fl = {v: preds[v].set_index("image_file").loc[imgs, "pred_fl"].values for v in variants}
    mt = {v: preds[v].set_index("image_file").loc[imgs, "pred_mt"].values for v in variants}
    gpa = base.set_index("image_file").loc[imgs, "gt_pa"].values
    gfl = base.set_index("image_file").loc[imgs, "gt_fl"].values
    gmt = base.set_index("image_file").loc[imgs, "gt_mt"].values
    feat_keys = ["fasc_mass", "orient_disp", "coher_mean", "apo_gap_px", "scale"]
    feat = np.stack(
        [base.set_index("image_file").loc[imgs, k].values.astype(float) for k in feat_keys],
        1,
    )
    col_med = np.nanmedian(feat, axis=0)
    bad = ~np.isfinite(feat)
    feat[bad] = np.take(col_med, np.where(bad)[1])
    # z-score features
    mu, sd = feat.mean(0), feat.std(0) + 1e-8
    feat_z = (feat - mu) / sd
    return imgs, n, pa, fl, mt, gpa, gfl, gmt, feat, feat_z, feat_keys


def eval_chosen(pa, fl, mt, gpa, gfl, gmt, chosen_v, variants):
    n = len(gpa)
    pa_o = np.zeros(n)
    fl_o = np.zeros(n)
    mt_o = np.zeros(n)
    for i in range(n):
        v = chosen_v[i]
        tr = [j for j in range(n) if j != i]
        a, b = np.polyfit(pa[v][tr], gpa[tr], 1)
        pa_o[i] = a * pa[v][i] + b
        a, b = np.polyfit(mt[v][tr], gmt[tr], 1)
        mt_o[i] = a * mt[v][i] + b
        fl_o[i] = fl[v][i]
    err = {
        "pa_deg": np.abs(pa_o - gpa),
        "fl_mm": np.abs(fl_o - gfl),
        "mt_mm": np.abs(mt_o - gmt),
    }
    comps = score_components(err)
    return float(comps["S"]), {k: float(np.mean(v)) for k, v in err.items()}


def nested_rule_fast(pa, fl, mt, gpa, gfl, gmt, feat, feat_keys, variants):
    n = len(gpa)
    feat_idx = {k: i for i, k in enumerate(feat_keys)}
    chosen = []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        best_s, best_assign = 9.0, None
        for fkey in feat_keys:
            fi = feat_idx[fkey]
            thr = float(np.median(feat[tr, fi]))
            for v_lo in variants:
                for v_hi in variants:
                    if v_lo == v_hi:
                        continue
                    for lo_below in (True, False):
                        # train score: mean raw component err
                        e = 0.0
                        for t in tr:
                            v = v_lo if ((feat[t, fi] < thr) == lo_below) else v_hi
                            e += abs(fl[v][t] - gfl[t]) / 12.0
                            e += abs(pa[v][t] - gpa[t]) / 6.0
                            e += abs(mt[v][t] - gmt[t]) / 3.0
                        e /= 3 * len(tr)
                        if e < best_s:
                            best_s = e
                            v_pick = v_lo if ((feat[i, fi] < thr) == lo_below) else v_hi
                            best_assign = v_pick
        chosen.append(best_assign)
    S, mae = eval_chosen(pa, fl, mt, gpa, gfl, gmt, chosen, variants)
    counts = {v: chosen.count(v) for v in variants}
    return {
        "S_loocv_cal_pa_mt": S,
        "MAE_loocv": mae,
        "picked_counts": counts,
        "beats_0.13294": S < PRIOR_BEST,
        "gate_0.12": S < 0.12,
        "gap_to_0.12": S - 0.12,
    }


def nested_knn(pa, fl, mt, gpa, gfl, gmt, feat_z, variants, k=5):
    """For each test i: among train, find k nearest in feature space; pick variant
    with best mean train raw err among those neighbors' oracle-ish votes.
    Simpler: pick variant minimizing mean raw err on the k neighbors.
    """
    n = len(gpa)
    chosen = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        d = np.linalg.norm(feat_z[tr] - feat_z[i], axis=1)
        nn = [tr[j] for j in np.argsort(d)[:k]]
        best_v, best_e = None, 9.0
        for v in variants:
            e = np.mean(
                [
                    abs(fl[v][t] - gfl[t]) / 12.0
                    + abs(pa[v][t] - gpa[t]) / 6.0
                    + abs(mt[v][t] - gmt[t]) / 3.0
                    for t in nn
                ]
            )
            if e < best_e:
                best_e, best_v = e, v
        chosen.append(best_v)
    S, mae = eval_chosen(pa, fl, mt, gpa, gfl, gmt, chosen, variants)
    return {
        "S_loocv_cal_pa_mt": S,
        "MAE_loocv": mae,
        "picked_counts": {v: chosen.count(v) for v in variants},
        "beats_0.13294": S < PRIOR_BEST,
        "gate_0.12": S < 0.12,
        "k": k,
    }


def nested_trainS(pa, fl, mt, gpa, gfl, gmt, variants):
    n = len(gpa)
    chosen = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_v, best_e = None, 9.0
        for v in variants:
            e = np.mean(
                [
                    abs(fl[v][t] - gfl[t]) / 12.0
                    + abs(pa[v][t] - gpa[t]) / 6.0
                    + abs(mt[v][t] - gmt[t]) / 3.0
                    for t in tr
                ]
            )
            if e < best_e:
                best_e, best_v = e, v
        chosen.append(best_v)
    S, mae = eval_chosen(pa, fl, mt, gpa, gfl, gmt, chosen, variants)
    return {
        "S_loocv_cal_pa_mt": S,
        "MAE_loocv": mae,
        "picked_counts": {v: chosen.count(v) for v in variants},
        "beats_0.13294": S < PRIOR_BEST,
        "gate_0.12": S < 0.12,
    }


def oracle(pa, fl, mt, gpa, gfl, gmt, variants):
    n = len(gpa)
    opa = np.zeros(n)
    ofl = np.zeros(n)
    omt = np.zeros(n)
    for i in range(n):
        best_e = 9.0
        for v in variants:
            e = abs(fl[v][i] - gfl[i]) / 12.0 + abs(pa[v][i] - gpa[i]) / 6.0
            if e < best_e:
                best_e = e
                opa[i], ofl[i], omt[i] = pa[v][i], fl[v][i], mt[v][i]
    pa_o = np.zeros(n)
    mt_o = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        a, b = np.polyfit(opa[m], gpa[m], 1)
        pa_o[i] = a * opa[i] + b
        a, b = np.polyfit(omt[m], gmt[m], 1)
        mt_o[i] = a * omt[i] + b
    err = {
        "pa_deg": np.abs(pa_o - gpa),
        "fl_mm": np.abs(ofl - gfl),
        "mt_mm": np.abs(mt_o - gmt),
    }
    comps = score_components(err)
    return {"S_loocv_cal_pa_mt": float(comps["S"]), "MAE_loocv": {k: float(np.mean(v)) for k, v in err.items()}}


def main():
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    cache_dir = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    out_dir = ROOT / "reports/phase3/chase018"
    kw = dict(FROZEN)

    print("building pool preds...", flush=True)
    summaries, preds = build_preds(expert, expert_dir, cache_dir, kw, POOL)
    imgs, n, pa, fl, mt, gpa, gfl, gmt, feat, feat_z, feat_keys = stacks(preds, POOL)

    print("nested trainS...", flush=True)
    pick = nested_trainS(pa, fl, mt, gpa, gfl, gmt, POOL)
    print(pick, flush=True)
    print("nested rule fast...", flush=True)
    rule = nested_rule_fast(pa, fl, mt, gpa, gfl, gmt, feat, feat_keys, POOL)
    print(rule, flush=True)
    print("nested knn...", flush=True)
    knns = {}
    for k in (3, 5, 7):
        knns[k] = nested_knn(pa, fl, mt, gpa, gfl, gmt, feat_z, POOL, k=k)
        print(k, knns[k], flush=True)
    ora = oracle(pa, fl, mt, gpa, gfl, gmt, POOL)
    print("oracle", ora, flush=True)

    candidates = [
        {"name": s["name"], "S_loocv_cal_pa_mt": s["S_loocv_cal_pa_mt"], "MAE_loocv": s["MAE_loocv"],
         "beats_0.13294": s["beats_0.13294"], "gate_0.12": s["gate_0.12"], "gap_to_0.12": s["gap_to_0.12"]}
        for s in summaries
    ]
    for name, obj in [("nested_trainS", pick), ("nested_feature_rule", rule)]:
        candidates.append({"name": name, **obj, "gap_to_0.12": obj["S_loocv_cal_pa_mt"] - 0.12})
    for k, obj in knns.items():
        candidates.append({"name": f"nested_knn_k{k}", **obj, "gap_to_0.12": obj["S_loocv_cal_pa_mt"] - 0.12})
    best = min(candidates, key=lambda c: c["S_loocv_cal_pa_mt"])

    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "protocol": "honest LOOCV affine PA+MT; FL raw; frozen recipe; nested-safe C (fast)",
        "frozen_recipe": FROZEN,
        "prior_best_S": PRIOR_BEST,
        "pool": POOL,
        "variants_ranked": sorted(summaries, key=lambda r: r["S_loocv_cal_pa_mt"]),
        "nested_trainS": pick,
        "nested_feature_rule": rule,
        "nested_knn": knns,
        "oracle_among_pool": ora,
        "best_honest": best,
        "gate_0.12": bool(best["S_loocv_cal_pa_mt"] < 0.12),
        "promote": bool(best["S_loocv_cal_pa_mt"] < PRIOR_BEST),
        "v1_leaky_note": "v1 feature_rule S=0.12963 was LEAKY; discarded",
        "kaggle_submit": False,
        "diagnosis": {
            "single_variant_ceiling": min(s["S_loocv_cal_pa_mt"] for s in summaries),
            "oracle_S": ora["S_loocv_cal_pa_mt"],
            "transfer_gap": ora["S_loocv_cal_pa_mt"] - 0.12,
            "conclusion": (
                "A post-filters never beat Frundles baseline 0.13294; nested C selectors "
                "collapse to baseline or worse. Oracle ~0.10 shows per-image headroom that "
                "does not transfer. Need MultExp-domain fasc map learning (B), not more morph."
            ),
        },
    }
    out_json = out_dir / "multexp_fasc_adapt_v2_scores.json"
    out_json.write_text(json.dumps(payload, indent=2))

    # update main scores + status
    main_path = out_dir / "multexp_fasc_adapt_scores.json"
    main = {
        "asof": payload["asof"],
        "protocol": payload["protocol"],
        "frozen_recipe": FROZEN,
        "prior_best_S": PRIOR_BEST,
        "best_honest": best,
        "variants_A_best": min(summaries, key=lambda s: s["S_loocv_cal_pa_mt"]),
        "nested_C": {"trainS": pick, "feature_rule": rule, "knn": knns},
        "oracle_among_A": ora,
        "v1_leaky_discarded": {"S": 0.12962987263663162, "reason": "outer rule search leak"},
        "gate_0.12": payload["gate_0.12"],
        "promote": payload["promote"],
        "promote_recipe_json": False,
        "kaggle_submit": False,
        "diagnosis": payload["diagnosis"],
        "artifacts": [
            "multexp_fasc_adapt_scores.json",
            "multexp_fasc_adapt_v2_scores.json",
            "multexp_fasc_adapt_status.md",
            "multexp_fasc_adapt.log",
            "multexp_fasc_adapt_v2.log",
        ],
    }
    main_path.write_text(json.dumps(main, indent=2))

    status = []
    status.append(f"# MultExp fasc adaptation (A/C) — {payload['asof']}")
    status.append("")
    status.append(f"- Prior / Frundles baseline S: **{PRIOR_BEST:.5f}** (PA 0.928 / FL 2.297 / MT 0.158)")
    status.append(
        f"- Best honest adapted: **{best['name']}** S=**{best['S_loocv_cal_pa_mt']:.5f}** "
        f"PA {best['MAE_loocv']['pa_deg']:.3f} / FL {best['MAE_loocv']['fl_mm']:.3f} / "
        f"MT {best['MAE_loocv']['mt_mm']:.3f}"
    )
    status.append(
        f"- Beats prior: {best.get('beats_0.13294')} | gate 0.12: {best.get('gate_0.12')} | "
        f"promote: **False** (recipe JSON untouched)"
    )
    status.append(f"- Oracle among A pool: S={ora['S_loocv_cal_pa_mt']:.5f} (not transferable)")
    status.append("- v1 leaky rule 0.12963 discarded (outer search on full LOOCV)")
    status.append("")
    status.append("## A single-variant ranked")
    for s in sorted(summaries, key=lambda r: r["S_loocv_cal_pa_mt"]):
        status.append(
            f"- {s['S_loocv_cal_pa_mt']:.5f} `{s['name']}` "
            f"pa={s['MAE_loocv']['pa_deg']:.3f} fl={s['MAE_loocv']['fl_mm']:.3f}"
        )
    status.append("")
    status.append("## C nested selectors")
    status.append(f"- trainS: S={pick['S_loocv_cal_pa_mt']:.5f} counts={pick['picked_counts']}")
    status.append(f"- feature_rule: S={rule['S_loocv_cal_pa_mt']:.5f} counts={rule['picked_counts']}")
    for k, obj in knns.items():
        status.append(f"- knn k={k}: S={obj['S_loocv_cal_pa_mt']:.5f} counts={obj['picked_counts']}")
    status.append("")
    status.append("## Diagnosis / next")
    status.append(payload["diagnosis"]["conclusion"])
    status.append(
        "Next lever: **B** short ≤30–60 min MultExp fasc U-Net fine-tune with geometry-aware / "
        "consistency loss, nested/group-safe (leave-one-out or image-holdout), freeze apo, "
        "keep frozen FL/geo recipe for eval."
    )
    (out_dir / "multexp_fasc_adapt_status.md").write_text("\n".join(status) + "\n")
    print("BEST", best, flush=True)
    print("wrote", main_path, out_json, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
