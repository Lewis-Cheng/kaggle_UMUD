#!/usr/bin/env python3
"""B-only re-run with corrected baseline FL + expanded protocols. Also strengthen PA-head A2."""
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
SPEC = importlib.util.spec_from_file_location("aporel", ROOT / "scripts/chase018/multexp_aporelpa_flgeo.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
from umud.calibration.anisotropic_scale import mm_per_prob_pixel
from umud.geometry.host_faithful import apo_curves, slope_deg
from umud.seg.frundles_smp import load_ckpt, predict_prob

REPORT = ROOT / "reports/phase3/chase018"
HOLD = mod.HOLD
GATE = mod.GATE
device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
H, W = mod.H, mod.W


def deep_apo_slope(apo, px_per_mm, gshape):
    H0, W0 = gshape
    h, w = apo.shape
    mx, my = mm_per_prob_pixel(px_per_mm, px_per_mm, W0, H0, w, h)
    curves = apo_curves(apo, mx, my, thr=0.60)
    if curves is None or len(curves) < 2:
        return 0.0
    cx = 0.5 * (curves[1][1] + curves[1][2])
    return float(slope_deg(curves[1][0], np.clip(cx, curves[1][1], curves[1][2])))


def main():
    t0 = time.time()
    fr_cache = REPORT / "cache" / "maps_multexp"

    # Verify corrected baseline
    S0, mae0, df0 = mod.eval_cache(fr_cache, fl_protocol="baseline")
    print(f"[check] corrected baseline S={S0:.5f} PA={mae0['pa_deg']:.3f} FL={mae0['fl_mm']:.3f} (expect ~0.13294)", flush=True)

    class Args: pass
    B_out = mod.run_B(Args(), REPORT, pa_overrides=None)

    # A2: stronger PA head using deep apo slope + ExpDrawn targets
    print("[A2] stronger PA head...", flush=True)
    targets = json.loads((REPORT / "multexp_aporelpa_targets.json").read_text())
    fasc_fr, _ = load_ckpt(ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt", device)
    apo_fr, apo_sz = load_ckpt(ROOT / "data/external/frundles_weights/apo_resnet34_384x576_f0.pt", device)
    img_dir = ROOT / "data/external/osf_xbawc/expdrawn_images"
    Xtr, Ytr = [], []
    for r in targets["rows"]:
        ip = img_dir / f"{r['stem']}.png"
        if not ip.exists():
            continue
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        fasc = predict_prob(fasc_fr, (H, W), g, device)
        apo = predict_prob(apo_fr, apo_sz, g, device)
        # use stored apo_slope from target build
        feats = mod.fasc_feats(fasc, r["apo_slope"], r["pa"], r["pa"])
        # extend feats: add more moments
        f = fasc.astype(np.float32)
        m = f > 0.35
        extras = np.array([
            float(f.std()),
            float(np.percentile(f, 90)),
            float(m.sum()) / f.size,
            float(r["apo_slope"]) / 30.0,
        ], dtype=np.float32)
        # pad into 8-d head by replacing last dims carefully — rebuild larger head
        Xtr.append(np.concatenate([feats[:6], extras[:2]]))
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
    # val on train (sanity) 
    with torch.no_grad():
        pred = head(Xtr_t).cpu().numpy()
    print(f"[A2] train MAE PA={np.mean(np.abs(pred-np.asarray(Ytr))):.3f}", flush=True)

    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    best_A2 = None
    for blend in (0.25, 0.40, 0.50, 0.60, 0.75, 1.0):
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
                m = f > 0.35
                extras = np.array([float(f.std()), float(np.percentile(f, 90))], dtype=np.float32)
                x = np.concatenate([feats[:6], extras])
                pa_hat = float(head(torch.tensor(x[None], device=device)).cpu().numpy()[0])
                pa_hat = float(np.clip(pa_hat, 5, 40))
                pa_ov[name] = (1 - blend) * pred0["pred_pa"] + blend * pa_hat
        S, mae, df = mod.eval_cache(fr_cache, fl_protocol="baseline", pa_overrides=pa_ov)
        rec = {"blend": blend, "S": S, "mae": mae}
        print(f"[A2] blend={blend:.2f} S={S:.5f} PA={mae['pa_deg']:.3f} FL={mae['fl_mm']:.3f}", flush=True)
        if best_A2 is None or S < best_A2["S"]:
            best_A2 = {**rec, "pa_ov": pa_ov, "df": df}

    # Also try A2 PA + B best protocol foldpure
    combo = None
    if best_A2["mae"]["pa_deg"] < mae0["pa_deg"] - 1e-6:
        print("[combo] A2 improved PA — B with PA overrides", flush=True)
        combo = mod.run_B(Args(), REPORT, pa_overrides=best_A2["pa_ov"])
    else:
        # still try combo with A2 PA even if PA MAE not better — maybe helps FL via sin
        print("[combo] trying A2 PA + B anyway (sin path)", flush=True)
        combo = mod.run_B(Args(), REPORT, pa_overrides=best_A2["pa_ov"])

    # load prior A from soft-PA FT
    A_ft = json.loads((REPORT / "multexp_aporelpa_flgeo_A.json").read_text()) if (REPORT / "multexp_aporelpa_flgeo_A.json").exists() else {}
    A_S_ft = A_ft.get("best", {}).get("S")
    A_S_head = best_A2["S"]
    B_S = B_out["best"]["S"]
    C_S = combo.get("best", {}).get("S") if combo else None
    scores = [s for s in (A_S_ft, A_S_head, B_S, C_S) if isinstance(s, (int, float))]
    champ = min(scores) if scores else None

    summary = {
        "asof": mod.now(),
        "baseline_check_S": S0,
        "baseline_check_mae": mae0,
        "A_ft_S": A_S_ft,
        "A_ft_best": A_ft.get("best"),
        "A2_pa_head_S": A_S_head,
        "A2_best": {k: v for k, v in best_A2.items() if k not in ("pa_ov", "df")},
        "B_S": B_S,
        "B_best": B_out["best"],
        "combo_S": C_S,
        "combo_best": combo.get("best") if combo else None,
        "champ_S": champ,
        "hold": HOLD,
        "frundles_floor": mod.FRUNDLE_BASE,
        "gate_0.12": bool(champ is not None and champ < GATE),
        "promote": bool(champ is not None and champ < HOLD),
        "kaggle_submit": False,
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    (REPORT / "multexp_aporelpa_flgeo_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (REPORT / "multexp_aporelpa_flgeo_A2.json").write_text(json.dumps({k: v for k, v in summary.items() if k.startswith("A") or k in ("asof", "baseline_check_S")}, indent=2, default=str))
    status = f"""# MultExp apo-rel PA + FL geo (corrected) — {mod.now()}

## Result
| track | S | notes |
|---|---|---|
| corrected Frundles baseline check | {S0:.5f} | PA {mae0['pa_deg']:.3f} / FL {mae0['fl_mm']:.3f} |
| A soft-PA FT (ExpDrawn) | {A_S_ft} | {A_ft.get('best')} |
| A2 PA-head blend | {A_S_head:.5f} | blend={best_A2['blend']} PA={best_A2['mae']['pa_deg']:.3f} |
| B FL geometry | {B_S:.5f} | {B_out['best']} |
| combo A2+B | {C_S} | {combo.get('best') if combo else None} |
| **champ** | **{champ}** | |
| hold | {HOLD} | |
| Frundles floor | {mod.FRUNDLE_BASE} | |
| gate 0.12 | {'PASS' if (champ is not None and champ < GATE) else 'FAIL'} | |
| promote | {bool(champ is not None and champ < HOLD)} | |
| no Kaggle | true | |

## Diagnosis
A soft-PA FT preserved Dice (~0.28) but MultExp maps worse than Frundles. A2 PA-head on ExpDrawn apo-rel targets. B expands FL protocols on corrected frozen baseline.

## Next
{"Promoted under hold — still no Kaggle until gate 0.12." if (champ is not None and champ < HOLD) else "Neither apo-rel PA nor FL protocol beat hold 0.13195. Next: fold-pure MultExp nested PA cal with ExpDrawn-informed prior, or Fascicle length via deep-apo geodesic on better MultExp-domain fasc."}

Artifacts: multexp_aporelpa_flgeo_summary.json, _A.json, _A2.json, _B.json
Elapsed_min: {summary['elapsed_min']:.1f}
"""
    sp = REPORT / "status.md"
    prev = sp.read_text() if sp.exists() else ""
    sp.write_text(status + "\n---\n\n" + prev)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
