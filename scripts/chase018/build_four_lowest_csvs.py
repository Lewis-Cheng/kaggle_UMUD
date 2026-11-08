#!/usr/bin/env python3
"""Build 4 test CSVs for the next-lowest honest MultExp recipes (no new seg train).

Recipes (exclude already-submitted joint_foldpure 0.13449):
  1. aporel_combo_a2b     reported 0.13240  (PA-head blend 0.25 + FL sin_w 0.30)
  2. fl_sin_w030          reported 0.13241  (Frundles maps, FL sin_w 0.30)
  3. pa_head_a2_b025      reported 0.13291  (PA-head blend 0.25, FL sin_w 0.35)
  4. frundles_fl_post     reported 0.13294  (frozen hough_tex0.4_thr035_apo60_ff0.9_sin0.35)
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
import torch.nn.functional as F

ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location("joint", ROOT / "scripts/chase018/multexp_joint_foldpure.py")
joint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(joint)

mod = joint.mod
from umud.metric.score import score_components

REPORT = ROOT / "reports/phase3/chase018"
HK = timezone(timedelta(hours=8))
RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
RECIPES = [
    {"tag": "aporel_combo_a2b", "alpha": 0.25, "sin_w": 0.30, "reported_S": 0.13240068410255748},
    {"tag": "fl_sin_w030", "alpha": 0.0, "sin_w": 0.30, "reported_S": 0.13241475990880067},
    {"tag": "pa_head_a2_b025", "alpha": 0.25, "sin_w": 0.35, "reported_S": 0.13290682825773772},
    {"tag": "frundles_fl_post", "alpha": 0.0, "sin_w": 0.35, "reported_S": 0.13293555901039555},
]


def clip_row(r: dict) -> dict:
    for k, (lo, hi) in RANGES.items():
        r[k] = float(np.clip(r[k], lo, hi))
    return r


def px_from_aniso(sx, sy):
    sx, sy = float(sx), float(sy)
    val = float(np.sqrt(abs(sx * sy)))
    if not np.isfinite(val) or val <= 0:
        val = float(np.nanmean([sx, sy]))
    return val


def loocv_and_fullfit(pa, fl, mt, gpa, gfl, gmt):
    S, mae, _ = joint.loocv_score_arrays(pa, fl, mt, gpa, gfl, gmt)
    pa_a, pa_b = [float(x) for x in np.polyfit(pa, gpa, 1)]
    mt_a, mt_b = [float(x) for x in np.polyfit(mt, gmt, 1)]
    return float(S), mae, dict(pa_a=pa_a, pa_b=pa_b, mt_a=mt_a, mt_b=mt_b)


def train_pa_head(device: str):
    torch.manual_seed(0)
    np.random.seed(0)
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
        loss = F.smooth_l1_loss(pred, Ytr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    print(f"PA head trained on {len(Xtr)} ExpDrawn rows", flush=True)
    return head, fasc_fr


def pa_hat_from_maps(head, device, g, apo, fasc, px):
    pred0 = mod.predict_bundle(g, apo, fasc, px)
    slope = joint.deep_apo_slope(apo, px, g.shape)
    pa_h = float(pred0.get("pa_h", pred0.get("pred_pa", 15.0)))
    pa_tex = float(pred0.get("pa_tex", pred0.get("pred_pa", pa_h)))
    if not np.isfinite(pa_tex):
        pa_tex = pa_h
    fts = mod.fasc_feats(fasc, slope, pa_h, pa_tex)
    extras = np.array([float(fasc.std()), float(np.percentile(fasc, 90))], dtype=np.float32)
    x = np.concatenate([fts[:6], extras])
    with torch.no_grad():
        hat = float(np.clip(head(torch.tensor(x[None], device=device)).cpu().numpy()[0], 5, 40))
    return float(pred0["pred_pa"]), hat, pred0


def predict_maps(g, apo, fasc, px, alpha, sin_w, head, device):
    po = None
    if alpha > 0 and head is not None:
        base_pa, hat, _ = pa_hat_from_maps(head, device, g, apo, fasc, px)
        po = (1.0 - alpha) * base_pa + alpha * hat
    pred = mod.predict_bundle(g, apo, fasc, px, fl_protocol="baseline", pa_override=po)
    key = f"sin_w{sin_w:.2f}"
    fl = float(pred.get("fl_cands", {}).get(key, pred["pred_fl"]))
    if abs(sin_w - 0.35) < 1e-9 and (not np.isfinite(fl)):
        fl = float(pred.get("fl_cands", {}).get("baseline", pred["pred_fl"]))
    return float(pred["pred_pa"]), fl, float(pred["pred_mt"]), bool(pred.get("ok", True))


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    joint.device = device
    print(f"device={device} start {datetime.now(HK).isoformat()}", flush=True)

    ids_path = ROOT / "reports/phase3/official_test_file_names.txt"
    test_dir = ROOT / "data/raw/umud/test_images_v2/test_set_v2"
    scale_csv = REPORT / "scale_aniso.csv"
    test_cache = REPORT / "cache" / "maps_test"
    mex_cache = REPORT / "cache" / "maps_multexp"
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"

    need_head = any(r["alpha"] > 0 for r in RECIPES)
    head = train_pa_head(device)[0] if need_head else None

    # MultExp raw preds + honest LOOCV + full-fit calibration
    mex_rows = {r["tag"]: [] for r in RECIPES}
    for i, er in expert.iterrows():
        name = er["image_file"]
        zp = mex_cache / f"{name}.npz"
        g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
        if g is None or not zp.exists():
            continue
        z = np.load(zp)
        apo = z["apo"].astype(np.float32)
        fasc = z["fasc"].astype(np.float32)
        px = float(er["Scale_pixel_per_cm"]) / 10.0
        gpa, gfl, gmt = float(er["median_PA"]), float(er["median_FL"]), float(er["median_MT"])
        for rec in RECIPES:
            pa, fl, mt, ok = predict_maps(g, apo, fasc, px, rec["alpha"], rec["sin_w"], head, device)
            mex_rows[rec["tag"]].append(dict(name=name, pa=pa, fl=fl, mt=mt, ok=ok, gpa=gpa, gfl=gfl, gmt=gmt))
        if (i + 1) % 10 == 0 or (i + 1) == len(expert):
            print(f"multexp {i+1}/{len(expert)}", flush=True)

    calibrated = {}
    summary = []
    for rec in RECIPES:
        df = pd.DataFrame(mex_rows[rec["tag"]])
        S, mae, cal = loocv_and_fullfit(
            df.pa.values.astype(float), df.fl.values.astype(float), df.mt.values.astype(float),
            df.gpa.values.astype(float), df.gfl.values.astype(float), df.gmt.values.astype(float),
        )
        calibrated[rec["tag"]] = cal
        rec["verified_S"] = S
        rec["verified_mae"] = mae
        rec["calibration"] = cal
        rec["n_multexp"] = int(len(df))
        print(f"VERIFY {rec['tag']} S={S:.5f} (reported {rec['reported_S']:.5f}) MAE={mae} cal={cal}", flush=True)
        summary.append(rec)

    raw = [ln.strip() for ln in ids_path.read_text().splitlines() if ln.strip() and not ln.endswith(".csv")]
    basenames = [Path(ln).name for ln in raw]
    scale = pd.read_csv(scale_csv).set_index("image_id")

    test_pred = {r["tag"]: [] for r in RECIPES}
    missing = []
    for i, name in enumerate(basenames):
        if i % 25 == 0 or i + 1 == len(basenames):
            print(f"test {i+1}/{len(basenames)} {name}", flush=True)
        p = test_dir / name
        zp = test_cache / f"{name}.npz"
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None or not zp.exists() or name not in scale.index:
            missing.append(name)
            for rec in RECIPES:
                test_pred[rec["tag"]].append({"image_id": name, "pa_deg": 15.0, "fl_mm": 70.0, "mt_mm": 20.0, "ok": False})
            continue
        z = np.load(zp)
        apo = z["apo"].astype(np.float32)
        fasc = z["fasc"].astype(np.float32)
        sx = float(scale.loc[name, "sx"])
        sy = float(scale.loc[name, "sy"])
        px = px_from_aniso(sx, sy)
        for rec in RECIPES:
            pa, fl, mt, ok = predict_maps(g, apo, fasc, px, rec["alpha"], rec["sin_w"], head, device)
            cal = calibrated[rec["tag"]]
            pa_cal = cal["pa_a"] * pa + cal["pa_b"]
            mt_cal = cal["mt_a"] * mt + cal["mt_b"]
            rr = clip_row({"pa_deg": pa_cal, "fl_mm": fl, "mt_mm": mt_cal})
            test_pred[rec["tag"]].append({"image_id": name, **rr, "ok": ok, "raw_pa": pa, "raw_fl": fl, "raw_mt": mt})

    out_meta = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S Asia/Hong_Kong"),
        "note": "Next 4 lowest honest MultExp recipes after excluding submitted joint_foldpure. No leaky 0.13172.",
        "missing_maps": missing,
        "recipes": [],
    }
    paths = []
    for rec in RECIPES:
        df = pd.DataFrame(test_pred[rec["tag"]])
        sub = df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
        order = {b: i for i, b in enumerate(basenames)}
        sub["_o"] = sub.image_id.map(order)
        sub = sub.sort_values("_o").drop(columns="_o")
        out_csv = REPORT / f"submission_{rec['tag']}.csv"
        sub.to_csv(out_csv, index=False, lineterminator="\n")
        df.to_csv(REPORT / f"submission_{rec['tag']}_diag.csv", index=False)
        rec_out = {
            "tag": rec["tag"],
            "path": str(out_csv),
            "reported_S": rec["reported_S"],
            "verified_S": rec["verified_S"],
            "verified_mae": rec["verified_mae"],
            "calibration": rec["calibration"],
            "n_rows": int(len(sub)),
            "alpha": rec["alpha"],
            "sin_w": rec["sin_w"],
        }
        out_meta["recipes"].append(rec_out)
        paths.append(out_csv)
        print(f"wrote {out_csv} rows={len(sub)}", flush=True)
        print(sub.describe().round(3), flush=True)

    meta_path = REPORT / "submission_four_lowest_meta.json"
    meta_path.write_text(json.dumps(out_meta, indent=2, default=str) + "\n")
    print("meta", meta_path, flush=True)
    print("missing", missing, flush=True)
    print("DONE", datetime.now(HK).isoformat(), flush=True)


if __name__ == "__main__":
    main()
