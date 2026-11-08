#!/usr/bin/env python3
"""Dump+eval numeric-geo ckpt with correct MultExp .npz cache protocol."""
from __future__ import annotations
import json, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import cv2, numpy as np, pandas as pd
import segmentation_models_pytorch as smp
import torch

ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from train_fasc_geoloss_labtrain_comp import (  # noqa
    dump_multexp, blend_caches, eval_frozen_robust, H, W, HOLD, GATE, now
)

HK = timezone(timedelta(hours=8))
ckpt_path = ROOT / "reports/phase3/chase018/seg_ckpts/fasc_resnet34_384x576_numeric_geo_labtrain_best.pt"
cache_dir = ROOT / "reports/phase3/chase018/cache/maps_multexp_numeric_geo"
report_dir = ROOT / "reports/phase3/chase018"
fr_cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
device = "cuda:0"

print(f"[fix] start {now()}", flush=True)
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
model.load_state_dict(ck["model"])
model = model.to(device).eval()
print(f"[fix] loaded dice={ck.get('best_val_dice')} n_targets={ck.get('n_numeric_targets')}", flush=True)

nmaps = dump_multexp(model, cache_dir, device)
print(f"[fix] dumped {nmaps}", flush=True)

print("[fix] eval raw", flush=True)
best_raw, grid_raw, df_raw = eval_frozen_robust(cache_dir)
print("raw", best_raw, flush=True)

best_blend = None
for a in [0.25, 0.40, 0.55, 0.70]:
    out_c = report_dir / "cache" / f"maps_multexp_numeric_geo_blend{int(a*100):02d}"
    n = blend_caches(cache_dir, fr_cache, out_c, a)
    print(f"[fix] blend a={a} n={n}", flush=True)
    b, g, _ = eval_frozen_robust(out_c)
    print(f"  S={b['S']:.5f}", flush=True)
    if best_blend is None or b["S"] < best_blend["best"]["S"]:
        best_blend = {"alpha": a, "best": b, "cache": str(out_c)}

print("[fix] frundles ref", flush=True)
best_fr, grid_fr, _ = eval_frozen_robust(fr_cache)

candidates = [
    {"tag": "raw_numeric_geo", "S": best_raw["S"], "detail": best_raw, "cache": str(cache_dir)},
]
if best_blend:
    candidates.append({"tag": f"blend_a{best_blend['alpha']:.2f}", "S": best_blend["best"]["S"], "detail": best_blend["best"], "cache": best_blend["cache"]})
candidates.append({"tag": "frundles_ref", "S": best_fr["S"], "detail": best_fr, "cache": str(fr_cache)})
non_fr = sorted([c for c in candidates if c["tag"] != "frundles_ref"], key=lambda c: c["S"])
champ = non_fr[0]
promote = champ["S"] < HOLD
gate = champ["S"] < GATE

report = {
    "updated": now(),
    "numeric_geo_pair_count": ck.get("n_numeric_targets"),
    "best_val_dice": ck.get("best_val_dice"),
    "init": ck.get("init"),
    "nmaps": nmaps,
    "raw_best": best_raw,
    "raw_grid": grid_raw,
    "blend_best": best_blend["best"] if best_blend else None,
    "blend_alpha": best_blend["alpha"] if best_blend else None,
    "frundles_ref": best_fr,
    "champ": champ,
    "hold": HOLD,
    "gate": GATE,
    "promote": bool(promote),
    "gate_pass": bool(gate),
    "expdrawn": {
        "found": True,
        "path": str(ROOT / "data/external/osf_xbawc/expdrawn"),
        "young_files": 156,
        "old_files": 342,
        "young_subjects": 40,
        "old_subjects": 85,
        "note": "OSF Example Annotated Images = ExpDrawnData; .tif ~1200px annotated panels, .tiff ~800px US frames; not yet converted to train masks this session",
    },
}
out = report_dir / "multexp_numeric_geo_labtrain.json"
out.write_text(json.dumps(report, indent=2, default=str))
if df_raw is not None:
    df_raw.to_csv(report_dir / "multexp_numeric_geo_labtrain_best_rows.csv", index=False)
status = report_dir / "status.md"
prev = status.read_text() if status.exists() else ""
block = f"""# MultExp numeric-geo LabTrain — {now()}

## Result
| metric | value |
|---|---|
| **champ S** | **{champ['S']:.5f}** ({champ['tag']}) |
| MAE PA/FL/MT | {champ['detail'].get('mae_pa', float('nan')):.3f} / {champ['detail'].get('mae_fl', float('nan')):.3f} / {champ['detail'].get('mae_mt', float('nan')):.3f} |
| vs hold 0.13195 | {'BEAT' if champ['S'] < HOLD else 'MISS'} ({champ['S']-HOLD:+.5f}) |
| gate 0.12 | {'PASS' if gate else 'FAIL'} |
| numeric geo pairs | {ck.get('n_numeric_targets')} |
| val Dice | {ck.get('best_val_dice')} |
| Frundles ref | {best_fr['S']:.5f} |
| ExpDrawn | FOUND (40Y+85O subjects; Young+Old OSF zips) |
| no Kaggle | true |

---
"""
status.write_text(block + prev)
print(json.dumps({"champ": champ, "promote": promote, "gate": gate, "fr": best_fr}, indent=2, default=str), flush=True)
print(f"[fix] wrote {out}", flush=True)
