#!/usr/bin/env python3
"""Build test CSV for honest MultExp feature_rule_joint (chase018).

Fits the joint feature-rule on ALL MultExp (transferable full-fit of the
nested recipe that scored S=0.13449), then applies it to official test
images using cached Frundles maps. No segmentation training.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location("joint", ROOT / "scripts/chase018/multexp_joint_foldpure.py")
joint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(joint)

mod = joint.mod
adapt = joint.adapt

REPORT = ROOT / "reports/phase3/chase018"
HK = timezone(timedelta(hours=8))
RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
FEATURE_KEYS = joint.FEATURE_KEYS
VARIANTS = joint.VARIANTS
SIN_WS = joint.SIN_WS
PA_ALPHAS = joint.PA_ALPHAS


def clip_row(r: dict) -> dict:
    for k, (lo, hi) in RANGES.items():
        r[k] = float(np.clip(r[k], lo, hi))
    return r


def fit_feature_rule(store, names, gpa, gfl, gmt):
    n = len(names)
    base_df = store[("baseline", 0.0)].set_index("image_file").loc[names]
    feat = np.stack([base_df[k].values.astype(float) for k in FEATURE_KEYS], 1)
    col_med = np.nanmedian(feat, axis=0)
    inds = np.where(~np.isfinite(feat))
    feat[inds] = np.take(col_med, inds[1])
    feat_idx = {k: i for i, k in enumerate(FEATURE_KEYS)}

    arrays = {}
    for v in VARIANTS:
        for a in PA_ALPHAS:
            for sw in SIN_WS:
                arrays[(v, sw, a)] = joint.stack_preds(store, names, v, a, sin_w=sw)

    tr = list(range(n))
    best_cfg = None
    best_train = 9.0
    for fkey in FEATURE_KEYS:
        fi = feat_idx[fkey]
        thr = float(np.median(feat[tr, fi]))
        for v_lo in VARIANTS:
            for v_hi in VARIANTS:
                if v_lo == v_hi:
                    continue
                for sw in SIN_WS:
                    for a in PA_ALPHAS:
                        pa_lo, fl_lo, _ = arrays[(v_lo, sw, a)]
                        pa_hi, fl_hi, _ = arrays[(v_hi, sw, a)]

                        def train_err(lo_is_below, _plo=pa_lo, _flo=fl_lo, _phi=pa_hi, _fhi=fl_hi, _fi=fi, _thr=thr):
                            e = []
                            for t in tr:
                                use_lo = (feat[t, _fi] < _thr) == lo_is_below
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
                            best_cfg = dict(
                                fkey=fkey, fi=fi, thr=thr, v_lo=v_lo, v_hi=v_hi,
                                lo_is_below=bool(lo_is_below), sin_w=float(sw), alpha=float(a),
                                train_proxy=float(train_s),
                            )

    picks = []
    pa = np.zeros(n)
    fl = np.zeros(n)
    mt = np.zeros(n)
    for i in range(n):
        use_lo = (feat[i, best_cfg["fi"]] < best_cfg["thr"]) == best_cfg["lo_is_below"]
        v_pick = best_cfg["v_lo"] if use_lo else best_cfg["v_hi"]
        picks.append(v_pick)
        pai, fli, mti = arrays[(v_pick, best_cfg["sin_w"], best_cfg["alpha"])]
        pa[i], fl[i], mt[i] = pai[i], fli[i], mti[i]

    pa_a, pa_b = np.polyfit(pa, gpa, 1)
    mt_a, mt_b = np.polyfit(mt, gmt, 1)
    # nested-style LOOCV affine on the full-fit assignments (sanity, slightly leaky vs 0.13449)
    S, mae, _ = joint.loocv_score_arrays(pa, fl, mt, gpa, gfl, gmt)
    return best_cfg, dict(pa_a=float(pa_a), pa_b=float(pa_b), mt_a=float(mt_a), mt_b=float(mt_b)), {
        "fullfit_loocv_S": float(S),
        "fullfit_loocv_MAE": mae,
        "pick_counts": {k: int(v) for k, v in __import__("collections").Counter(picks).items()},
    }


def px_from_aniso(sx, sy):
    sx, sy = float(sx), float(sy)
    val = float(np.sqrt(abs(sx * sy)))
    if not np.isfinite(val) or val <= 0:
        val = float(np.nanmean([sx, sy]))
    return val


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    joint.device = device
    out_csv = REPORT / "submission_joint_foldpure.csv"
    out_json = REPORT / "submission_joint_foldpure.json"
    ids_path = ROOT / "reports/phase3/official_test_file_names.txt"
    test_dir = ROOT / "data/raw/umud/test_images_v2/test_set_v2"
    scale_csv = REPORT / "scale_aniso.csv"
    test_cache = REPORT / "cache" / "maps_test"

    print(f"device={device} start {datetime.now(HK).isoformat()}", flush=True)
    skip_path = REPORT / "submission_joint_foldpure_rule.json"
    if skip_path.exists() or "--skip-fit" in sys.argv:
        src = skip_path
        blob = json.loads(src.read_text())
        cfg, cal, sanity = blob["rule"], blob["calibration"], blob.get("sanity", {})
        print("loaded cached rule from", src, json.dumps(cfg), flush=True)
    else:
        print("building PA overrides...", flush=True)
        pa_overrides_by_alpha = {0.0: {}}
        for a in PA_ALPHAS:
            if a <= 0:
                continue
            print(f"  PA head blend={a}", flush=True)
            pa_overrides_by_alpha[a] = joint.build_pa_overrides(a)

        print("precompute MultExp store...", flush=True)
        store, names = joint.precompute(pa_overrides_by_alpha)
        base = store[("baseline", 0.0)].set_index("image_file").loc[names]
        gpa = base["gt_pa"].values.astype(float)
        gfl = base["gt_fl"].values.astype(float)
        gmt = base["gt_mt"].values.astype(float)

        print("fit feature rule on all MultExp...", flush=True)
        cfg, cal, sanity = fit_feature_rule(store, names, gpa, gfl, gmt)
        print("rule", json.dumps(cfg, sort_keys=True), flush=True)
        print("cal", cal, "sanity", sanity, flush=True)
        skip_path.write_text(json.dumps({"rule": cfg, "calibration": cal, "sanity": sanity}, indent=2) + "\n")
        print("wrote", skip_path, flush=True)

    # PA head for test (reuse same training recipe; predict on test maps)
    pa_ov_test = {}
    if cfg["alpha"] > 0:
        print("predict PA-head overrides on test...", flush=True)
        # rebuild head the same way as build_pa_overrides, then score test
        # Call the inner training by invoking build_pa_overrides' head path via a copy.
        # We train once more at the chosen blend and also compute test overrides.
        targets = json.loads((REPORT / "multexp_aporelpa_targets.json").read_text())
        fasc_fr, _ = joint.load_ckpt(ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt", device)
        img_dir = ROOT / "data/external/osf_xbawc/expdrawn_images"
        Xtr, Ytr = [], []
        for r in targets["rows"]:
            ip = img_dir / f"{r['stem']}.png"
            if not ip.exists():
                continue
            g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
            fasc = joint.predict_prob(fasc_fr, (mod.H, mod.W), g, device)
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
            loss = torch.nn.functional.smooth_l1_loss(pred, Ytr_t)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
    else:
        head = None
        fasc_fr = None

    raw = [ln.strip() for ln in ids_path.read_text().splitlines() if ln.strip() and not ln.endswith(".csv")]
    basenames = [Path(ln).name for ln in raw]
    scale = pd.read_csv(scale_csv).set_index("image_id")

    rows = []
    pick_counts = {}
    missing_maps = []
    with torch.no_grad():
        for i, name in enumerate(basenames):
            if i % 25 == 0 or i + 1 == len(basenames):
                print(f"test {i+1}/{len(basenames)} {name}", flush=True)
            p = test_dir / name
            zp = test_cache / f"{name}.npz"
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None or not zp.exists():
                missing_maps.append(name)
                rows.append({"image_id": name, "pa_deg": 15.0, "fl_mm": 70.0, "mt_mm": 20.0, "ok": False, "variant": "missing"})
                continue
            z = np.load(zp)
            apo = z["apo"].astype(np.float32)
            fasc0 = z["fasc"].astype(np.float32)
            if name not in scale.index:
                raise SystemExit(f"missing scale for {name}")
            sx = float(scale.loc[name, "sx"])
            sy = float(scale.loc[name, "sy"])
            px = px_from_aniso(sx, sy)

            fasc_base = joint.adapt_ext(g, apo, fasc0, "baseline")
            feats = adapt.image_features(g, apo, fasc_base, px)
            feat_val = float(feats.get(cfg["fkey"], np.nan))
            if not np.isfinite(feat_val):
                feat_val = float(cfg["thr"])
            use_lo = (feat_val < cfg["thr"]) == cfg["lo_is_below"]
            v_pick = cfg["v_lo"] if use_lo else cfg["v_hi"]
            pick_counts[v_pick] = pick_counts.get(v_pick, 0) + 1
            fasc = joint.adapt_ext(g, apo, fasc0, v_pick)

            po = None
            if head is not None:
                pred0 = mod.predict_bundle(g, apo, fasc0, px)
                slope = joint.deep_apo_slope(apo, px, g.shape)
                pa_h = float(pred0.get("pa_h", pred0.get("pred_pa", 15.0)))
                pa_tex = float(pred0.get("pa_tex", pred0.get("pred_pa", pa_h)))
                if not np.isfinite(pa_tex):
                    pa_tex = pa_h
                fts = mod.fasc_feats(fasc0, slope, pa_h, pa_tex)
                extras = np.array([float(fasc0.std()), float(np.percentile(fasc0, 90))], dtype=np.float32)
                x = np.concatenate([fts[:6], extras])
                pa_hat = float(np.clip(head(torch.tensor(x[None], device=device)).cpu().numpy()[0], 5, 40))
                po = (1 - cfg["alpha"]) * float(pred0["pred_pa"]) + cfg["alpha"] * pa_hat

            pred = mod.predict_bundle(g, apo, fasc, px, fl_protocol="baseline", pa_override=po)
            key = f"sin_w{cfg['sin_w']:.2f}"
            fl = float(pred.get("fl_cands", {}).get(key, pred["pred_fl"]))
            pa = float(pred["pred_pa"])
            mt = float(pred["pred_mt"])
            pa_cal = cal["pa_a"] * pa + cal["pa_b"]
            mt_cal = cal["mt_a"] * mt + cal["mt_b"]
            rr = clip_row({"pa_deg": pa_cal, "fl_mm": fl, "mt_mm": mt_cal})
            rows.append({
                "image_id": name,
                "pa_deg": rr["pa_deg"],
                "fl_mm": rr["fl_mm"],
                "mt_mm": rr["mt_mm"],
                "ok": bool(pred.get("ok", True)),
                "variant": v_pick,
                "raw_pa": pa,
                "raw_fl": fl,
                "raw_mt": mt,
                "feat_val": feat_val,
            })

    df = pd.DataFrame(rows)
    sub = df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
    order = {b: i for i, b in enumerate(basenames)}
    sub["_o"] = sub.image_id.map(order)
    sub = sub.sort_values("_o").drop(columns="_o")
    sub.to_csv(out_csv, index=False, lineterminator="\n")
    df.to_csv(out_csv.with_name(out_csv.stem + "_diag.csv"), index=False)

    meta = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S Asia/Hong_Kong"),
        "recipe": "feature_rule_joint_fullfit_transfer",
        "honest_nested_S": 0.1344865957876554,
        "honest_tag": "fr_joint",
        "rule": cfg,
        "calibration": cal,
        "sanity": sanity,
        "test_pick_counts": pick_counts,
        "n_rows": int(len(sub)),
        "missing_maps": missing_maps,
        "out_csv": str(out_csv),
        "note": "Gate S<0.12 overridden for this single user-requested submit. Do not use leaky 0.13172.",
    }
    out_json.write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", out_csv, "rows", len(sub), flush=True)
    print(sub.describe().round(3), flush=True)
    print("missing_maps", missing_maps, flush=True)
    print("test picks", pick_counts, flush=True)
    print("meta", out_json, flush=True)


if __name__ == "__main__":
    main()
