#!/usr/bin/env python3
"""Update promoted recipe for honest fasc-adapt selector; optional short MultExp self-train B."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

HK = timezone(timedelta(hours=8))
ROOT = Path("/home/chocheng9/kaggle_UMUD")
OUT = ROOT / "reports/phase3/chase018"

PRIOR = 0.13293555901039555
BEST = {
    "name": "nested_feature_rule_fasc_adapt",
    "S_loocv_cal_pa_mt": 0.13195144580335014,
    "MAE_loocv": {
        "pa_deg": 0.9124039825246483,
        "fl_mm": 2.292390615561086,
        "mt_mm": 0.15826250681487788,
    },
    "kw_frozen": {
        "fasc_thr": 0.35,
        "apo_thr": 0.6,
        "tex_w": 0.4,
        "fl_factor": 0.9,
        "fl_sin_w": 0.35,
        "agree_thr": None,
        "pa_mode": "hough",
    },
    "fasc_adapt": {
        "type": "nested_feature_rule",
        "pool": [
            "baseline",
            "ridge_blend03",
            "morph_connect",
            "corridor_dilate",
            "blend_base_morph",
            "ridge_soft02",
        ],
        "picked_counts": {
            "baseline": 16,
            "ridge_blend03": 1,
            "morph_connect": 0,
            "corridor_dilate": 0,
            "blend_base_morph": 18,
            "ridge_soft02": 0,
        },
        "note": "Per fold: choose feature-threshold rule on train raw err; apply to held-out. Maps = Frundles + A post-filters.",
    },
}

asof = datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong")
recipe = {
    "asof": asof,
    "name": BEST["name"],
    "S_loocv_cal_pa_mt": BEST["S_loocv_cal_pa_mt"],
    "MAE_loocv": BEST["MAE_loocv"],
    "kw": BEST["kw_frozen"],
    "fasc_adapt": BEST["fasc_adapt"],
    "cache_dir": "reports/phase3/chase018/cache/maps_multexp",
    "maps": "Frundles apo+fasc + MultExp-domain A post-filters selected fold-pure (C)",
    "prior_fixed_recipe_S": PRIOR,
    "prior_fixed_recipe_name": "hough_tex0.4_thr035_apo60_ff0.9_sin0.35_agNone",
    "beats_prior_fixed": True,
    "gate_0.12": False,
    "gap_to_0.12": BEST["S_loocv_cal_pa_mt"] - 0.12,
    "kaggle_submit": False,
    "lab_fasc_ckpt": False,
    "protocol": "honest LOOCV affine PA+MT only; FL raw; nested-safe fasc adapt selector",
}
(OUT / "math_map_promoted_recipe.json").write_text(json.dumps(recipe, indent=2))

# also keep fixed-recipe snapshot
fixed = {
    "asof": asof,
    "name": "hough_tex0.4_thr035_apo60_ff0.9_sin0.35_agNone",
    "S_loocv_cal_pa_mt": PRIOR,
    "MAE_loocv": {
        "pa_deg": 0.9279384631482271,
        "fl_mm": 2.2967497297340818,
        "mt_mm": 0.15826250681487788,
    },
    "kw": BEST["kw_frozen"],
    "note": "Fixed Frundles-map recipe retained as comparator; superseded as MultExp floor by nested_feature_rule_fasc_adapt",
}
(OUT / "math_map_fixed_recipe_frundles.json").write_text(json.dumps(fixed, indent=2))

status = f"""# MultExp fasc adaptation (A/C) — {asof}

- Prior fixed Frundles recipe S: **{PRIOR:.5f}** (PA 0.928 / FL 2.297 / MT 0.158)
- Best honest adapted: **{BEST['name']}** S=**{BEST['S_loocv_cal_pa_mt']:.5f}** PA {BEST['MAE_loocv']['pa_deg']:.3f} / FL {BEST['MAE_loocv']['fl_mm']:.3f} / MT {BEST['MAE_loocv']['mt_mm']:.3f}
- Beats prior: **True** | gate 0.12: **False** (gap {BEST['S_loocv_cal_pa_mt']-0.12:.5f}) | recipe JSON **updated**
- v1 leaky rule 0.12963 discarded (outer search leak)
- Oracle among A pool ≈ 0.1216 (barely above gate; not transferable)

## What was tried (A/C)
- A: apo-belly gate, orient filter, ridge blend/replace, morph connect/prune, frangi-like, corridor dilate, soft ridge, base blends
- Single-variant A: **none** beat 0.13294 (best = baseline)
- C nested-safe: trainS→baseline; feature_rule→**0.13195**; knn k=5/7 → 0.13246 / 0.13230

## Diagnosis
A morph/ridge post-filters alone do not beat Frundles; fold-pure C yields a small honest gain (~0.001). Pool oracle ~0.122 means even perfect A-pool transfer barely clears 0.12 — remaining gap needs better MultExp fasc maps (B), not more morph.

## Next lever
Short MultExp-domain fasc U-Net self-train (B): Frundles init, soft teacher = apo-bounded ridge/Frundles blend on MultExp images only (no MultExp GT → no LOOCV leak), ≤30–60 min on cuda:0, then frozen-recipe MultExp eval.
"""
(OUT / "multexp_fasc_adapt_status.md").write_text(status)
print("updated recipe + status", asof)
