#!/usr/bin/env python3
"""Resume after A FT: eval aporelpa cache + blends + PA head + B FL geo + combo."""
from __future__ import annotations
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "aporel", ROOT / "scripts/chase018/multexp_aporelpa_flgeo.py"
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

from umud.seg.frundles_smp import load_ckpt, predict_prob  # noqa: E402

REPORT = ROOT / "reports/phase3/chase018"
HOLD = mod.HOLD
GATE = mod.GATE
FRUNDLE_BASE = mod.FRUNDLE_BASE
H, W = mod.H, mod.W
device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"


def main():
    t0 = time.time()
    fr_cache = REPORT / "cache" / "maps_multexp"
    cache_raw = REPORT / "cache" / "maps_multexp_aporelpa"
    ckpt = REPORT / "seg_ckpts" / "fasc_resnet34_384x576_aporelpa_best.pt"
    print(f"[resume] device={device} cache_exists={cache_raw.exists()} ckpt={ckpt.exists()}", flush=True)

    results = []
    best = None
    # A eval on existing maps
    if not cache_raw.exists() or len(list(cache_raw.glob("*.npz"))) < 30:
        if not ckpt.exists():
            raise SystemExit("missing ckpt and cache")
        print("[resume] re-dumping MultExp from ckpt", flush=True)
        model, _ = load_ckpt(ckpt, device)
        mod.dump_multexp_fasc(model, cache_raw, device)

    for thr in (0.30, 0.35, 0.40):
        kw = dict(mod.FROZEN)
        kw["fasc_thr"] = thr
        S, mae, df = mod.eval_cache(cache_raw, fl_protocol="baseline", kw=kw)
        rec = {"tag": "aporelpa_raw", "thr": thr, "S": S, "mae": mae}
        results.append(rec)
        print(f"[A] raw thr={thr} S={S:.5f} PA={mae['pa_deg']:.3f} FL={mae['fl_mm']:.3f}", flush=True)
        if best is None or S < best["S"]:
            best = {**rec, "df": df}

    for a in (0.15, 0.25, 0.40):
        outc = REPORT / "cache" / f"maps_multexp_aporelpa_blend{int(a*100):02d}"
        mod.blend_caches(cache_raw, fr_cache, outc, a)
        S, mae, df = mod.eval_cache(outc, fl_protocol="baseline")
        rec = {"tag": f"blend_a{a:.2f}", "thr": 0.35, "S": S, "mae": mae}
        results.append(rec)
        print(f"[A] blend a={a:.2f} S={S:.5f} PA={mae['pa_deg']:.3f} FL={mae['fl_mm']:.3f}", flush=True)
        if best is None or S < best["S"]:
            best = {**rec, "df": df, "cache": str(outc)}

    S0, mae0, df0 = mod.eval_cache(fr_cache, fl_protocol="baseline")
    print(f"[A] Frundles ref S={S0:.5f} PA={mae0['pa_deg']:.3f}", flush=True)

    # PA head from saved targets
    targets = json.loads((REPORT / "multexp_aporelpa_targets.json").read_text())
    rows_meta = targets["rows"]
    img_dir = ROOT / "data/external/osf_xbawc/expdrawn_images"
    # rebuild train features with Frundles fasc
    fasc_model_fr, _ = load_ckpt(ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt", device)
    head = mod.PAHead().to(device)
    Xtr, Ytr = [], []
    for r in rows_meta:
        ip = img_dir / f"{r['stem']}.png"
        if not ip.exists():
            continue
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        fasc = predict_prob(fasc_model_fr, (H, W), g, device)
        feats = mod.fasc_feats(fasc, r["apo_slope"], r["pa"], r["pa"])
        Xtr.append(feats)
        Ytr.append(r["pa"])
    Xtr_t = torch.tensor(np.stack(Xtr), device=device)
    Ytr_t = torch.tensor(np.asarray(Ytr, dtype=np.float32), device=device)
    opt_h = torch.optim.Adam(head.parameters(), lr=1e-3)
    head.train()
    for _ in range(400):
        pred = head(Xtr_t)
        loss = F.smooth_l1_loss(pred, Ytr_t)
        opt_h.zero_grad()
        loss.backward()
        opt_h.step()
    head.eval()

    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    pa_overrides = {}
    with torch.no_grad():
        for _, er in expert.iterrows():
            name = er["image_file"]
            zp = fr_cache / f"{name}.npz"
            g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
            if g is None or not zp.exists():
                continue
            z = np.load(zp)
            pred0 = mod.predict_bundle(
                g, z["apo"].astype(np.float32), z["fasc"].astype(np.float32), float(er["Scale_pixel_per_cm"]) / 10.0
            )
            feats = mod.fasc_feats(z["fasc"].astype(np.float32), 0.0, pred0["pa_h"], pred0["pa_tex"])
            pa_hat = float(head(torch.tensor(feats[None], device=device)).cpu().numpy()[0])
            pa_overrides[name] = 0.5 * pred0["pred_pa"] + 0.5 * float(np.clip(pa_hat, 5, 40))
    S_h, mae_h, df_h = mod.eval_cache(fr_cache, fl_protocol="baseline", pa_overrides=pa_overrides)
    results.append({"tag": "pa_head_blend50", "thr": 0.35, "S": S_h, "mae": mae_h})
    print(f"[A] PA-head blend S={S_h:.5f} PA={mae_h['pa_deg']:.3f}", flush=True)
    if S_h < best["S"]:
        best = {"tag": "pa_head_blend50", "thr": 0.35, "S": S_h, "mae": mae_h, "df": df_h}

    A_out = {
        "asof": mod.now(),
        "experiment": "A_aporel_soft_pa",
        "n_pa_targets": targets["n"],
        "mean_target_pa": targets["mean_pa"],
        "val_dice_best": None,
        "ckpt": str(ckpt),
        "frundles_ref": {"S": S0, "mae": mae0},
        "results": [{k: v for k, v in r.items() if k != "df"} for r in results],
        "best": {k: v for k, v in best.items() if k != "df"},
        "pa_improved_vs_frundles": bool(best["mae"]["pa_deg"] < mae0["pa_deg"] - 1e-6),
        "beats_hold": bool(best["S"] < HOLD),
        "gate_0.12": bool(best["S"] < GATE),
        "elapsed_min": (time.time() - t0) / 60.0,
        "kaggle_submit": False,
        "resumed": True,
    }
    (REPORT / "multexp_aporelpa_flgeo_A.json").write_text(json.dumps(A_out, indent=2, default=str))
    if "df" in best:
        best["df"].to_csv(REPORT / "multexp_aporelpa_flgeo_A_best_rows.csv", index=False)
    print(f"[A] BEST S={best['S']:.5f} tag={best['tag']} pa_improved={A_out['pa_improved_vs_frundles']}", flush=True)

    # B
    class Args:
        pass
    args = Args()
    B_out = mod.run_B(args, REPORT, pa_overrides=None)

    combo_out = {"note": "skipped — A did not improve PA vs Frundles"}
    if A_out["pa_improved_vs_frundles"]:
        print("[combo] A improved PA — re-running B with A PA", flush=True)
        pa_ov = None
        if best.get("tag") == "pa_head_blend50":
            pa_ov = pa_overrides
        elif "df" in best:
            pa_ov = {r.image_file: float(r.pred_pa) for r in best["df"].itertuples()}
        if pa_ov:
            combo_out = mod.run_B(args, REPORT, pa_overrides=pa_ov)

    A_S = A_out["best"]["S"]
    B_S = B_out["best"]["S"]
    C_S = combo_out.get("best", {}).get("S") if isinstance(combo_out, dict) else None
    scores = [s for s in (A_S, B_S, C_S) if isinstance(s, (int, float))]
    champ = min(scores) if scores else None
    status = f"""# MultExp apo-rel PA + FL geo — {mod.now()}

## Result
| track | S | notes |
|---|---|---|
| A apo-rel soft PA | {A_S:.5f} | {A_out['best']} |
| B FL geometry | {B_S:.5f} | {B_out['best']} |
| combo | {C_S} | {combo_out.get('best', combo_out)} |
| **champ** | **{champ}** | |
| hold | {HOLD} | |
| Frundles baseline | {FRUNDLE_BASE} | |
| gate 0.12 | {'PASS' if (champ is not None and champ < GATE) else 'FAIL'} | |
| promote vs hold | {bool(champ is not None and champ < HOLD)} | |
| no Kaggle | true | |

## Diagnosis
A: ExpDrawn stroke→deep-apo soft PA FT (+ PA head) on Frundles init; MultExp held out. Resume after thinning fix.
B: Frundles MultExp maps only; tip/deep/cond/sin FL protocols + fold-pure.

## Next
{"Promoted — prepare LB CSV; still no Kaggle until gate." if (champ is not None and champ < HOLD) else "Still above hold. Next: MultExp-safe synthetic apo-rel PA pairs, or insertion-aware FL with expert-like deep snap priors beyond Frundles mass."}

Artifacts: multexp_aporelpa_flgeo_A.json / _B.json / _summary.json
Elapsed_all_min: {(time.time()-t0)/60.0:.1f}
"""
    status_path = REPORT / "status.md"
    prev = status_path.read_text() if status_path.exists() else ""
    status_path.write_text(status + "\n---\n\n" + prev)
    summary = {
        "asof": mod.now(),
        "A_S": A_S,
        "B_S": B_S,
        "combo_S": C_S,
        "champ_S": champ,
        "hold": HOLD,
        "gate_0.12": bool(champ is not None and champ < GATE),
        "promote": bool(champ is not None and champ < HOLD),
        "kaggle_submit": False,
        "A_best": A_out["best"],
        "B_best": B_out["best"],
        "combo_best": combo_out.get("best"),
        "pa_improved": A_out["pa_improved_vs_frundles"],
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    (REPORT / "multexp_aporelpa_flgeo_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
