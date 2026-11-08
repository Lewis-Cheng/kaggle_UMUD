#!/usr/bin/env python3
"""Outside-Frundles MultExp fasc maps: classical Frangi/ridge (+ optional blends).

Inventory-driven. Keeps Frundles apo cache. Writes maps_multexp_<tag>,
honest LOOCV with frozen geo recipe, small fasc_thr grid for classical scale.
No Kaggle. No attention. No Frundles+A selector grids. No lab scratch dump.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "src").exists():
    ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from umud.geometry.timtrack_fl import _frangi_vesselness  # noqa: E402
from umud.metric.score import score_components  # noqa: E402

# Reuse frozen recipe + predict from adapt_multexp_fasc
from adapt_multexp_fasc import (  # noqa: E402
    FROZEN,
    apo_belly_mask,
    loocv_cal_pa_mt,
    predict_one,
    structure_ridge,
)

HK = timezone(timedelta(hours=8))
HOLD = 0.13195144580335014
GATE = 0.12


def now_hk() -> str:
    return datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z")


def normalize_p99(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1.0, 99.5])
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def gabor_energy(gray: np.ndarray, freqs=(0.08, 0.12, 0.18), n_theta: int = 8) -> np.ndarray:
    g = gray.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    g = g - cv2.GaussianBlur(g, (0, 0), 3.0)
    acc = np.zeros_like(g)
    for f in freqs:
        for i in range(n_theta):
            theta = i * np.pi / n_theta
            ksize = 21
            kern = cv2.getGaborKernel((ksize, ksize), sigma=3.0, theta=theta, lambd=1.0 / f, gamma=0.5, psi=0)
            resp = cv2.filter2D(g, cv2.CV_32F, kern)
            acc = np.maximum(acc, np.abs(resp))
    return normalize_p99(acc)


def classical_fasc(
    gray: np.ndarray,
    apo: np.ndarray,
    mode: str,
    apo_thr: float = 0.60,
) -> np.ndarray:
    """Full classical fasc probability in apo belly; outside belly → 0."""
    h, w = apo.shape
    if gray.shape[:2] != (h, w):
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    belly = apo_belly_mask(apo, apo_thr, margin_px=2).astype(np.float32)
    ves = _frangi_vesselness(gray, sigmas=(1.0, 1.5, 2.5, 4.0))
    ridge, _ang, coher = structure_ridge(gray, sigma=1.5)
    ridge = normalize_p99(ridge * np.clip(coher, 0, 1))
    if mode == "frangi":
        fasc = ves
    elif mode == "ridge":
        fasc = ridge
    elif mode == "frangi_ridge":
        fasc = normalize_p99(0.55 * ves + 0.45 * ridge)
    elif mode == "gabor":
        fasc = gabor_energy(gray)
    elif mode == "frangi_gabor":
        fasc = normalize_p99(0.6 * ves + 0.4 * gabor_energy(gray))
    elif mode == "all_blend":
        gab = gabor_energy(gray)
        fasc = normalize_p99(0.4 * ves + 0.35 * ridge + 0.25 * gab)
    else:
        raise ValueError(mode)
    fasc = (fasc * belly).astype(np.float32)
    # mild contrast stretch inside belly so thr~0.35 is usable
    m = belly > 0.5
    if m.sum() > 50:
        vals = fasc[m]
        lo, hi = np.percentile(vals, [20, 99])
        if hi > lo + 1e-6:
            fasc[m] = np.clip((vals - lo) / (hi - lo), 0, 1)
    return fasc


def dump_maps(expert, expert_dir, apo_cache: Path, out_dir: Path, mode: str, force: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    masses = []
    for _, er in expert.iterrows():
        name = er["image_file"]
        out = out_dir / f"{name}.npz"
        zsrc = apo_cache / f"{name}.npz"
        if out.exists() and not force:
            n_skip += 1
            continue
        if not zsrc.exists():
            print(f"WARN missing apo cache {zsrc}", flush=True)
            continue
        g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"WARN missing image {name}", flush=True)
            continue
        z = np.load(zsrc)
        apo = z["apo"].astype(np.float32)
        fasc = classical_fasc(g, apo, mode)
        np.savez_compressed(out, apo=apo.astype(np.float16), fasc=fasc.astype(np.float16))
        masses.append(float((fasc > 0.35).mean()))
        n_ok += 1
        if n_ok % 5 == 0 or n_ok == 1:
            print(f"  [{mode}] wrote {n_ok} {name} fasc_mass@0.35={masses[-1]:.4f}", flush=True)
    print(
        f"[dump {mode}] wrote={n_ok} skip={n_skip} mean_mass@0.35="
        f"{(np.mean(masses) if masses else float('nan')):.4f} dir={out_dir}",
        flush=True,
    )
    return {"wrote": n_ok, "skip": n_skip, "mean_mass_035": float(np.mean(masses)) if masses else None}


def dump_blend_frundles_classical(
    expert, expert_dir, apo_fasc_cache: Path, classical_cache: Path, out_dir: Path, alpha: float, force: bool = False
):
    """Blend Frundles fasc with classical fasc under Frundles apo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for _, er in expert.iterrows():
        name = er["image_file"]
        out = out_dir / f"{name}.npz"
        if out.exists() and not force:
            continue
        a = apo_fasc_cache / f"{name}.npz"
        c = classical_cache / f"{name}.npz"
        if not a.exists() or not c.exists():
            continue
        za, zc = np.load(a), np.load(c)
        apo = za["apo"].astype(np.float32)
        fasc = np.clip(
            (1 - alpha) * za["fasc"].astype(np.float32) + alpha * zc["fasc"].astype(np.float32),
            0,
            1,
        )
        np.savez_compressed(out, apo=apo.astype(np.float16), fasc=fasc.astype(np.float16))
        n_ok += 1
    print(f"[blend alpha={alpha}] wrote={n_ok} dir={out_dir}", flush=True)
    return n_ok


def run_frozen_on_cache(expert, expert_dir, cache_dir: Path, thr_grid, tag: str):
    results = []
    best = None
    best_df = None
    for thr in thr_grid:
        kw = dict(FROZEN)
        kw["fasc_thr"] = float(thr)
        kw["name"] = f"{FROZEN['name']}_thr{thr:.2f}"
        rows = []
        for _, er in expert.iterrows():
            name = er["image_file"]
            g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
            zp = cache_dir / f"{name}.npz"
            if g is None or not zp.exists():
                continue
            z = np.load(zp)
            pred = predict_one(
                g,
                z["apo"].astype(np.float32),
                z["fasc"].astype(np.float32),
                float(er["Scale_pixel_per_cm"]) / 10.0,
                kw,
            )
            rows.append(
                dict(
                    image_file=name,
                    ok=pred["ok"],
                    pred_pa=pred["pred_pa"],
                    pred_fl=pred["pred_fl"],
                    pred_mt=pred["pred_mt"],
                    gt_pa=float(er["median_PA"]),
                    gt_fl=float(er["median_FL"]),
                    gt_mt=float(er["median_MT"]),
                )
            )
        df = pd.DataFrame(rows)
        if len(df) < 5:
            results.append({"tag": tag, "thr": thr, "error": "too few", "n": len(df)})
            continue
        comps, mae, *_ = loocv_cal_pa_mt(df)
        summary = {
            "tag": tag,
            "thr": float(thr),
            "n": int(len(df)),
            "ok": int(df.ok.sum()),
            "S": float(comps["S"]),
            "S1": float(comps["S1"]),
            "MAE": mae,
            "gate_0.12": float(comps["S"]) < GATE,
            "beats_hold": float(comps["S"]) < HOLD,
            "gap_to_hold": float(comps["S"]) - HOLD,
            "gap_to_0.12": float(comps["S"]) - GATE,
            "kw": {k: v for k, v in kw.items() if k != "name"},
        }
        results.append(summary)
        print(
            f"  [{tag}] thr={thr:.2f} S={summary['S']:.5f} "
            f"PA={mae['pa_deg']:.3f} FL={mae['fl_mm']:.3f} MT={mae['mt_mm']:.3f} "
            f"beats_hold={summary['beats_hold']}",
            flush=True,
        )
        if best is None or summary["S"] < best["S"]:
            best = summary
            best_df = df
    return results, best, best_df


def inventory(root: Path) -> dict:
    wt = list((root / "data/external").rglob("*.pt")) + list((root / "data/external").rglob("*.pth"))
    lab = list((root / "reports/phase3/seg_ckpts").glob("*.pt")) if (root / "reports/phase3/seg_ckpts").exists() else []
    chase = list((root / "reports/phase3/chase018/seg_ckpts").glob("*.pt")) if (root / "reports/phase3/chase018/seg_ckpts").exists() else []
    maps = sorted(
        p.name
        for p in (root / "reports/phase3/chase018/cache").glob("maps_multexp*")
        if p.is_dir()
    )
    return {
        "external_weights": [str(p.relative_to(root)) for p in wt],
        "phase3_lab_fasc_ckpts_n": len([p for p in lab if "fasc" in p.name]),
        "phase3_lab_fasc_top": [
            "seg_fasc_v5_best.pt (Dice~0.285 MSTinyUNet)",
            "seg_fasc_v10_best.pt (Dice~0.284)",
            "seg_fasc_v6b_best.pt (Dice~0.283)",
            "NOTE: all Dice≪Frundles/scratch~0.41; treated as weak outside-family DL",
        ],
        "chase018_ckpts_n": len(chase),
        "chase018_ckpts_note": "Frundles FT / scratch / nested — EXHAUSTED, not re-dumped",
        "map_caches": maps,
        "public_non_frundles_on_disk": False,
        "documented_weight_urls": ["kaggle dataset watehell/umud-r34-unet-weights (Frundles only)"],
        "multexp_pixel_fasc_gt": False,
        "multexp_gt_note": "expert_consensus PA/FL/MT only; no MultExp fasc masks → skip nested MultExp fasc head",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-dump", action="store_true")
    args = ap.parse_args()
    root = args.root
    expert_csv = root / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv"
    expert_dir = root / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    apo_cache = root / "reports/phase3/chase018/cache/maps_multexp"
    out_dir = root / "reports/phase3/chase018"
    expert = pd.read_csv(expert_csv)

    inv = inventory(root)
    print(json.dumps({"inventory": inv, "ts": now_hk()}, indent=2), flush=True)

    # Classical modes (small set)
    modes = ["frangi", "ridge", "frangi_ridge", "gabor", "all_blend"]
    thr_grid = [0.20, 0.28, 0.35, 0.45, 0.55, 0.65]
    all_results = []
    best_overall = None
    dump_stats = {}

    for mode in modes:
        tag = f"classical_{mode}"
        cache = out_dir / "cache" / f"maps_multexp_{tag}"
        if not args.skip_dump:
            dump_stats[tag] = dump_maps(expert, expert_dir, apo_cache, cache, mode, force=args.force)
        print(f"=== eval {tag} ===", flush=True)
        results, best, best_df = run_frozen_on_cache(expert, expert_dir, cache, thr_grid, tag)
        all_results.extend(results)
        if best is not None:
            if best_overall is None or best["S"] < best_overall["S"]:
                best_overall = dict(best)
                best_overall["cache"] = str(cache)
            # save best rows for this tag
            if best_df is not None:
                best_df.to_csv(out_dir / f"multexp_nonfrundles_fasc_{tag}_best_rows.csv", index=False)

    # Blends: Frundles fasc + best classical so far (or frangi_ridge)
    blend_src_tag = "classical_frangi_ridge"
    blend_src = out_dir / "cache" / f"maps_multexp_{blend_src_tag}"
    for alpha in (0.3, 0.5, 0.7):
        tag = f"blend_frundles_frangi_ridge_a{alpha:.1f}"
        cache = out_dir / "cache" / f"maps_multexp_{tag}"
        if not args.skip_dump:
            dump_blend_frundles_classical(expert, expert_dir, apo_cache, blend_src, cache, alpha, force=args.force)
        print(f"=== eval {tag} ===", flush=True)
        results, best, best_df = run_frozen_on_cache(expert, expert_dir, cache, thr_grid, tag)
        all_results.extend(results)
        if best is not None:
            if best_overall is None or best["S"] < best_overall["S"]:
                best_overall = dict(best)
                best_overall["cache"] = str(cache)
            if best_df is not None:
                best_df.to_csv(out_dir / f"multexp_nonfrundles_fasc_{tag}_best_rows.csv", index=False)

    # Baseline Frundles maps with frozen recipe (sanity / thr fixed)
    print("=== eval frundles_baseline frozen ===", flush=True)
    base_results, base_best, _ = run_frozen_on_cache(
        expert, expert_dir, apo_cache, [FROZEN["fasc_thr"]], "frundles_baseline"
    )
    all_results.extend(base_results)

    promoted = None
    if best_overall is not None and best_overall["S"] < HOLD:
        promoted = best_overall

    payload = {
        "ts_hk": now_hk(),
        "hold_S": HOLD,
        "gate": GATE,
        "frozen_recipe": FROZEN,
        "inventory": inv,
        "dump_stats": dump_stats,
        "sources_tried": modes + [f"blend_frundles_frangi_ridge_a{a}" for a in (0.3, 0.5, 0.7)] + ["frundles_baseline"],
        "all_results": all_results,
        "best_outside_family": best_overall,
        "frundles_baseline_best": base_best,
        "promoted": promoted,
        "gate_pass": bool(best_overall and best_overall["S"] < GATE),
        "beats_hold": bool(best_overall and best_overall["S"] < HOLD),
        "skipped": {
            "public_mawra_ambros_dltrack_xiaolei_vibecoder": "not on disk; only Frundles URL documented in-repo — no download",
            "lab_scratch_best_scratch60_ft": "exhausted / forbidden re-dump",
            "multexp_nested_fasc_gt_masks": "no MultExp pixel fasc GT",
            "phase3_v5_v12_dump": "Dice~0.28≪0.41; deferred unless classical fails inventory need — optional next",
        },
        "next_lever": None,
    }
    if best_overall is None or best_overall["S"] >= HOLD:
        payload["next_lever"] = (
            "Classical + blend did not beat 0.13195. Next: (1) dump phase3 MSTinyUNet v5/v12 MultExp maps "
            "despite low Dice as true outside-family DL, (2) apo-bounded multi-orientation Frangi with dark-ridge "
            "bias + fasc skeleton thinning as fasc channel, (3) seek documented Mawra/open fasc weight URL in refs "
            "before any new download."
        )
    else:
        payload["next_lever"] = "Outside-family map beat hold — freeze recipe, consider gate path only if S<0.12."

    out_json = out_dir / "multexp_nonfrundles_fasc_classical.json"
    out_json.write_text(json.dumps(payload, indent=2))
    # compact status
    status = out_dir / "status.md"
    lines = [
        f"# MultExp chase018 status\n",
        f"Updated: {now_hk()}\n\n",
        f"## Outside-family fasc (this turn)\n",
        f"- Hold S: **{HOLD:.5f}**\n",
        f"- Best outside-family: **{best_overall['S']:.5f}** ({best_overall.get('tag')}, thr={best_overall.get('thr')}) "
        f"PA={best_overall['MAE']['pa_deg']:.3f} FL={best_overall['MAE']['fl_mm']:.3f} MT={best_overall['MAE']['mt_mm']:.3f}\n"
        if best_overall
        else "- Best outside-family: none\n",
        f"- Frundles baseline frozen: **{base_best['S']:.5f}**\n" if base_best else "",
        f"- Gate 0.12: **{'PASS' if payload['gate_pass'] else 'FAIL'}**; beats hold: **{payload['beats_hold']}**; no Kaggle\n",
        f"- Inventory: public non-Frundles weights on disk=NO; MultExp fasc pixel GT=NO; phase3 lab fasc ckpts exist (Dice~0.28)\n",
        f"- Artifacts: `{out_json.name}`\n",
        f"- Next: {payload['next_lever']}\n",
    ]
    # prepend to status without wiping history too hard — write dedicated status file too
    (out_dir / "multexp_nonfrundles_fasc_status.md").write_text("".join(lines))
    # update main status.md with short prepend
    old = status.read_text() if status.exists() else ""
    status.write_text("".join(lines) + "\n---\n\n" + old)
    print(json.dumps({"best": best_overall, "baseline": base_best, "gate": payload["gate_pass"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
