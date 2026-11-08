#!/usr/bin/env python3
"""Deeper probe: hflip right-tick parse + tick-column search + 1080 left ticks."""
from __future__ import annotations
import csv, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path.home() / "kaggle_UMUD"
sys.path.insert(0, str(ROOT / "src"))
from umud.data.layout import discover_layout, list_images

_RIGHT_TICK_FOV = {
    7: (142, 91, 1058), 8: (163, 91, 1037), 9: (211, 91, 989),
    10: (249, 91, 951), 11: (282, 91, 918), 12: (308, 91, 892),
    13: (331, 91, 869), 14: (349, 91, 851), 15: (142, 91, 1058),
}

def parse_right_ticks_at_col(a, col, label):
    h, w = a.shape[:2]
    colv = a[:, col].mean(axis=-1).astype(np.float64)
    if float(colv.max()) < 50:
        return None
    first_tick = int(np.argmax(colv > 50))
    if first_tick + 25 >= h:
        return None
    second_major = first_tick + 20 + int(np.argmax(colv[first_tick + 20:] > 50))
    last_tick = h - 1 - int(np.argmax(colv[::-1] > 50))
    gap = max(second_major - first_tick, 1)
    n_ticks_f = (last_tick - first_tick) / gap
    n = int(round(n_ticks_f))
    if 7 <= n <= 14:
        px_per_cm = (last_tick - first_tick) / n * 2.0
        q = 5.0
    elif n == 15:
        px_per_cm = (last_tick - first_tick) / 3.0
        q = 2.0
    else:
        return dict(ok=False, n=n, n_f=n_ticks_f, gap=gap, first=first_tick, last=last_tick, label=label, col=col)
    return dict(ok=True, n=n, n_f=n_ticks_f, gap=gap, first=first_tick, last=last_tick,
                px_per_cm=px_per_cm, q=q, label=label, col=col)

rows = list(csv.DictReader((ROOT / "reports/phase3/ambrosm_scale_rows.csv").open()))
u = [r for r in rows if r["family"] == "rgb_800x1200_other"]
lay = discover_layout(ROOT / "data/raw/umud")
stem_map = {}
for pool, folder in {"fasc": lay.fasc_imgs, "apo": lay.apo_imgs, "test": lay.test_imgs}.items():
    for p in list_images(folder):
        stem_map[(pool, p.name)] = p

# Classify each unknown 800 into Rm / other and try parse
results = Counter()
n_hist = Counter()
px_vals = []
fail_n = Counter()
col_search_success = Counter()

for i, r in enumerate(u):
    p = stem_map.get((r["pool"], r["name"]))
    if p is None:
        results["missing"] += 1
        continue
    a = np.asarray(Image.open(p))
    if a.ndim != 3 or a.shape[:2] != (800, 1200):
        results["bad"] += 1
        continue
    aa = a.astype(np.uint8) if a.dtype == np.uint8 else (
        (a.astype(np.float32) * (255.0 / max(float(a.max()), 1))).astype(np.uint8)
    )
    rt_m = float(aa[87, 43:53].mean())
    # try mirrored right-tick col=49 (and nearby)
    parsed = None
    for col in (49, 50, 48, 51, 47, 52, 45, 55, 40, 60):
        cand = parse_right_ticks_at_col(aa, col, "hflip_right")
        if cand and cand.get("ok"):
            parsed = cand
            break
    if parsed:
        results["hflip_right_ok"] += 1
        n_hist[parsed["n"]] += 1
        px_vals.append(parsed["px_per_cm"])
        col_search_success[parsed["col"]] += 1
        continue
    # try normal right col in case fingerprint soft-missed
    for col in (1150, 1149, 1151, 1148, 1152):
        cand = parse_right_ticks_at_col(aa, col, "right")
        if cand and cand.get("ok"):
            parsed = cand
            break
    if parsed:
        results["right_soft_ok"] += 1
        n_hist[("R", parsed["n"])] += 1
        px_vals.append(parsed["px_per_cm"])
        continue
    # try left-tick fixed span after hflip: mirrored left fingerprint region
    # original left: a[42,67:74]; hflip -> a[42,1126:1133]
    if (aa[42, 1126:1133, 0] > 115).all() or float(aa[42, 1126:1133, 0].mean()) > 100:
        results["hflip_left_fp"] += 1
        continue
    # bottom ticks?
    bot = aa[-5, :, 0] if aa.ndim == 3 else aa[-5, :]
    if float(bot.max()) > 150:
        results["bottom_bright"] += 1
        continue
    # record fail reason
    if rt_m > 100:
        results["rm_fp_but_parse_fail"] += 1
        # keep one fail detail
        if results["rm_fp_but_parse_fail"] <= 5:
            cand = parse_right_ticks_at_col(aa, 49, "dbg")
            print("Rm fail", r["pool"], r["name"], cand)
    else:
        results["no_parse"] += 1

print("results", dict(results))
print("n_hist", dict(n_hist))
print("cols", dict(col_search_success))
if px_vals:
    print("px_per_cm median", float(np.median(px_vals)), "mean", float(np.mean(px_vals)), "n", len(px_vals))

# Probe 1080x1640 left ticks more carefully
print("\n=== 1080x1640 left-column tick analysis ===")
items = [r for r in rows if r["family"] == "other_1080x1640_RGB"]
gaps = []
for r in items[:40]:
    p = stem_map[(r["pool"], r["name"])]
    a = np.asarray(Image.open(p))
    # find brightest thin column in left 80 px
    best_c, best_score = None, -1
    for c in range(0, 80):
        colv = a[:, c].mean(axis=-1).astype(np.float64)
        score = float(np.percentile(colv, 95))
        if score > best_score:
            best_score, best_c = score, c
    colv = a[:, best_c].mean(axis=-1).astype(np.float64)
    thr = max(50.0, float(np.percentile(colv, 80)))
    b = (colv > thr).astype(np.int8)
    idx = np.where((b[1:] == 1) & (b[:-1] == 0))[0] + 1
    keep = []
    for x in idx:
        if not keep or x - keep[-1] >= 8:
            keep.append(int(x))
    keep = np.asarray(keep)
    med = float(np.median(np.diff(keep))) if len(keep) >= 4 else float("nan")
    gaps.append(med)
    if len(gaps) <= 8:
        print(f" {r['name']} best_c={best_c} score={best_score:.0f} n_edges={len(keep)} med_gap={med:.1f} first={keep[0] if len(keep) else None} last={keep[-1] if len(keep) else None}")

print("1080 med_gap over samples:", float(np.nanmedian(gaps)), "finite", int(np.isfinite(gaps).sum()))

# 500x760
print("\n=== 500x760 edge tick analysis ===")
items = [r for r in rows if r["family"] == "other_500x760_RGB"]
for r in items[:8]:
    p = stem_map[(r["pool"], r["name"])]
    a = np.asarray(Image.open(p))
    for side, cols in [("L", range(0, 40)), ("R", range(a.shape[1]-40, a.shape[1]))]:
        best_c, best_score = None, -1
        for c in cols:
            colv = a[:, c].mean(axis=-1).astype(np.float64)
            score = float(np.percentile(colv, 95))
            if score > best_score:
                best_score, best_c = score, c
        colv = a[:, best_c].mean(axis=-1).astype(np.float64)
        thr = max(40.0, float(np.percentile(colv, 85)))
        b = (colv > thr).astype(np.int8)
        idx = np.where((b[1:] == 1) & (b[:-1] == 0))[0] + 1
        keep = []
        for x in idx:
            if not keep or x - keep[-1] >= 6:
                keep.append(int(x))
        med = float(np.median(np.diff(keep))) if len(keep) >= 3 else float("nan")
        print(f" {r['name']} {side} c={best_c} p95={best_score:.0f} n={len(keep)} gap={med:.1f}")
