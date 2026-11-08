#!/usr/bin/env python3
from __future__ import annotations
import csv, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path.home() / "kaggle_UMUD"
sys.path.insert(0, str(ROOT / "src"))
from umud.data.layout import discover_layout, list_images

rows = list(csv.DictReader((ROOT / "reports/phase3/ambrosm_scale_rows.csv").open()))
u = [r for r in rows if r["family"] == "rgb_800x1200_other"]
print("n 800 other", len(u), "pools", Counter(r["pool"] for r in u))

lay = discover_layout(ROOT / "data/raw/umud")
pool_dirs = {"test": lay.test_imgs, "fasc": lay.fasc_imgs, "apo": lay.apo_imgs}
stem_map = {}
for pool, folder in pool_dirs.items():
    files = list_images(folder)
    print(pool, len(files), folder)
    for p in files:
        stem_map[(pool, p.name)] = p

counters = Counter()
samples = []
for r in u[:200]:
    p = stem_map.get((r["pool"], r["name"]))
    if p is None:
        counters["missing"] += 1
        continue
    a = np.asarray(Image.open(p))
    if a.ndim != 3 or tuple(a.shape[:2]) != (800, 1200):
        counters["shape_" + str(a.shape) + "_" + str(a.dtype)] += 1
        continue
    aa = a
    if a.dtype != np.uint8:
        counters["dtype_" + str(a.dtype)] += 1
        if int(a.max()) > 255:
            aa = (a.astype(np.float32) * (255.0 / float(a.max()))).astype(np.uint8)
        else:
            aa = a.astype(np.uint8)
    rt = float(aa[87, 1147:1157].mean())
    lt = float(aa[42, 67:74, 0].mean())
    rt_m = float(aa[87, 43:53].mean())
    lt_m = float(aa[42, 1126:1133, 0].mean())
    # AmbrosM exact-ish
    right_hit = (aa[87, 1147:1157] == 175).all() or rt > 150
    left_hit = (aa[42, 67:74, 0] > 115).all()
    right_hit_m = (aa[87, 43:53] == 175).all() or rt_m > 150
    left_hit_m = (aa[42, 1126:1133, 0] > 115).all()
    col49 = aa[:, 49].mean(axis=-1)
    col50 = aa[:, 50].mean(axis=-1)
    col1150 = aa[:, 1150].mean(axis=-1)
    tag = []
    if right_hit or rt > 100: tag.append("right")
    if left_hit or lt > 100: tag.append("left")
    if right_hit_m or rt_m > 100: tag.append("right_hflip")
    if left_hit_m or lt_m > 100: tag.append("left_hflip")
    if not tag:
        if max(float(col49.max()), float(col50.max())) > 100:
            tag.append("left_col_bright")
        if float(col1150.max()) > 100:
            tag.append("right_col_bright")
    key = "+".join(tag) if tag else "none"
    counters[key] += 1
    if len(samples) < 20:
        samples.append((r["pool"], r["name"], key, round(rt,1), round(lt,1), round(rt_m,1), round(lt_m,1), round(float(col49.max()),1), round(float(col1150.max()),1), int(a.min()), int(a.max()), str(a.dtype)))

print("counters", dict(counters))
print("samples pool name key rt lt rt_m lt_m col49max col1150max amin amax dtype")
for s in samples:
    print(" ", s)

# Full sweep counts for all 1972 (lighter): only fingerprint tags
print("\nFULL fingerprint sweep all", len(u))
full = Counter()
for r in u:
    p = stem_map.get((r["pool"], r["name"]))
    if p is None:
        full["missing"] += 1
        continue
    a = np.asarray(Image.open(p))
    if a.ndim != 3 or tuple(a.shape[:2]) != (800, 1200):
        full["bad_shape"] += 1
        continue
    aa = a.astype(np.uint8) if a.dtype != np.uint8 and int(a.max()) <= 255 else a
    if aa.dtype != np.uint8:
        aa = (a.astype(np.float32) * (255.0 / max(float(a.max()), 1.0))).astype(np.uint8)
    rt = float(aa[87, 1147:1157].mean())
    lt = float(aa[42, 67:74, 0].mean())
    rt_m = float(aa[87, 43:53].mean())
    lt_m = float(aa[42, 1126:1133, 0].mean())
    tag = []
    if rt > 100: tag.append("R")
    if lt > 100: tag.append("L")
    if rt_m > 100: tag.append("Rm")
    if lt_m > 100: tag.append("Lm")
    # looser column peaks
    if not tag:
        if float(aa[:, 49].mean(axis=-1).max()) > 120 or float(aa[:, 50].mean(axis=-1).max()) > 120:
            tag.append("Lcol")
        if float(aa[:, 1150].mean(axis=-1).max()) > 120:
            tag.append("Rcol")
    full["+".join(tag) if tag else "none"] += 1
print("full", dict(full))

# other families
for fam in ["other_1080x1640_RGB","other_500x760_RGB","other_652x800_RGB","other_600x800_RGB","other_556x660_RGB","other_688x1234_RGB"]:
    items = [r for r in rows if r["family"] == fam]
    print("\n===", fam, "n=", len(items), "===")
    for r in items[:3]:
        p = stem_map.get((r["pool"], r["name"]))
        if p is None:
            print(" missing", r["pool"], r["name"]); continue
        a = np.asarray(Image.open(p))
        print(" ", r["pool"], r["name"], "shape", a.shape, a.dtype, "minmax", int(a.min()), int(a.max()))
        if a.dtype != np.uint8 and int(a.max()) > 255:
            g = a.astype(np.float32)
            if g.ndim == 3: g = g.mean(axis=2)
            g = g * (255.0 / float(g.max() + 1e-6))
        else:
            g = a.mean(axis=2).astype(np.float64) if a.ndim == 3 else a.astype(np.float64)
        for side, band, ax in [("L", g[:, :40], 1), ("R", g[:, -40:], 1), ("B", g[-40:, :], 0), ("T", g[:40, :], 0)]:
            prof = band.mean(axis=ax)
            x = prof - prof.mean()
            lag = None
            if float(x.std()) > 1:
                c = np.correlate(x, x, mode="full")
                mid = len(c) // 2
                seg = c[mid + 10: mid + min(300, len(x) // 3)]
                if len(seg):
                    lag = 10 + int(np.argmax(seg))
            print(f"    {side}: max={prof.max():.1f} p95={np.percentile(prof,95):.1f} argmax={int(np.argmax(prof))} lag={lag}")
