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
lay = discover_layout(ROOT / "data/raw/umud")
stem_map = {}
for pool, folder in {"fasc": lay.fasc_imgs, "apo": lay.apo_imgs}.items():
    for p in list_images(folder):
        stem_map[(pool, p.name)] = p

def parse_right(a, col):
    h = a.shape[0]
    colv = a[:, col].mean(axis=-1).astype(np.float64)
    first = int(np.argmax(colv > 50))
    if first + 25 >= h: return None
    second = first + 20 + int(np.argmax(colv[first+20:] > 50))
    last = h - 1 - int(np.argmax(colv[::-1] > 50))
    gap = max(second - first, 1)
    n = int(round((last - first) / gap))
    if 7 <= n <= 14:
        return dict(ok=True, n=n, px=(last-first)/n*2, q=5.0, first=first, last=last, gap=gap)
    if n == 15:
        return dict(ok=True, n=n, px=(last-first)/3.0, q=2.0, first=first, last=last, gap=gap)
    return dict(ok=False, n=n, first=first, last=last, gap=gap)

# Bottom-bright subset of 800 other
u = [r for r in rows if r["family"]=="rgb_800x1200_other"]
print("=== bottom / residual 800x1200_other ===")
bottom_n = 0
residual = []
for r in u:
    p = stem_map.get((r["pool"], r["name"]))
    if not p: continue
    a = np.asarray(Image.open(p))
    # skip if hflip right works
    if parse_right(a, 49) and parse_right(a, 49).get("ok"):
        continue
    bot = a[-8:, :, :].mean(axis=2)
    bottom_n += 1
    # bottom profile
    prof = bot.mean(axis=0)
    # rising edges on bottom
    thr = max(40.0, float(np.percentile(prof, 85)))
    b = (prof > thr).astype(np.int8)
    idx = np.where((b[1:]==1)&(b[:-1]==0))[0]+1
    keep=[]
    for x in idx:
        if not keep or x-keep[-1]>=8: keep.append(int(x))
    med = float(np.median(np.diff(keep))) if len(keep)>=3 else float("nan")
    # also try left fixed-span after mild checks
    left_m = float(a[42, 1126:1133, 0].mean())
    left = float(a[42, 67:74, 0].mean())
    residual.append((r["name"], r["pool"], float(bot.max()), len(keep), med, left, left_m, parse_right(a,49)))
print("residual count", len(residual))
for x in residual[:15]:
    print(" ", x)
# gap hist for bottom
gaps = [x[4] for x in residual if np.isfinite(x[4])]
print("bottom med_gaps sample", gaps[:20], "median", float(np.median(gaps)) if gaps else None)

# 1080: inspect left strip as image region - maybe vertical ruler with small ticks
print("\n=== 1080 left strip detail ===")
items = [r for r in rows if r["family"]=="other_1080x1640_RGB"]
for r in items[:3]:
    a = np.asarray(Image.open(stem_map[(r["pool"], r["name"])]))
    strip = a[:, :80, :].mean(axis=2)
    # row-wise mean of strip
    row = strip.mean(axis=1)
    # find bright rows
    thr = max(50, float(np.percentile(row, 90)))
    bright_rows = np.where(row > thr)[0]
    print(r["name"], "strip mean", float(strip.mean()), "row p90", float(np.percentile(row,90)), "n_bright_rows", len(bright_rows), "first/last", bright_rows[:5], bright_rows[-5:] if len(bright_rows) else None)
    # column profile inside strip at mid-height band
    for y0,y1 in [(0,100),(100,400),(400,800),(800,1080)]:
        band = strip[y0:y1].mean(axis=0)
        print(f"  y{y0}-{y1} colmax={band.max():.0f}@{int(band.argmax())} colmean={band.mean():.0f}")
    # try detect periodic ticks along a thin column with derivative
    for c in range(0, 60):
        colv = a[:, c].mean(axis=-1).astype(np.float64)
        d = np.diff(colv)
        peaks = np.where((d[:-1] > 20) & (d[1:] < -5))[0]  # crude spike
        if len(peaks) >= 5:
            med = float(np.median(np.diff(peaks)))
            print(f"  spike col={c} n={len(peaks)} med={med:.1f} first={peaks[0]} last={peaks[-1]}")
            break
    else:
        print("  no spike col found")

# Compare: known right-tick image vs hflip success px distribution vs measured right family
print("\n=== sanity: known right_ticks col1150 vs hflip col49 px ===")
known = [r for r in rows if r["family"]=="rgb_800x1200_right_ticks" and r["provenance"]=="measured"][:5]
for r in known:
    p = None
    for pool in ("test","fasc","apo"):
        if (pool, r["name"]) in stem_map:
            p = stem_map[(pool, r["name"])]; break
    # stem_map only fasc/apo above - rebuild test
print("rebuild test map")
for p in list_images(lay.test_imgs):
    stem_map[("test", p.name)] = p
for r in known:
    pool = r["pool"]
    p = stem_map.get((pool, r["name"]))
    if not p: 
        print("miss", r); continue
    a = np.asarray(Image.open(p))
    pr = parse_right(a, 1150)
    print(r["pool"], r["name"], "csv_px_cm", r["px_per_cm"], "parse1150", pr)

# 500x760: check consistency of L gap 58 across all
print("\n=== 500x760 L/R gap distribution ===")
Lgaps=[]; Rgaps=[]
for r in [r for r in rows if r["family"]=="other_500x760_RGB"]:
    a = np.asarray(Image.open(stem_map[(r["pool"], r["name"])]))
    for side, cols, bucket in [("L", range(0,40), Lgaps), ("R", range(a.shape[1]-40, a.shape[1]), Rgaps)]:
        best_c, best_score = None, -1
        for c in cols:
            colv = a[:, c].mean(axis=-1).astype(np.float64)
            score = float(np.percentile(colv, 95))
            if score > best_score:
                best_score, best_c = score, c
        colv = a[:, best_c].mean(axis=-1).astype(np.float64)
        thr = max(40.0, float(np.percentile(colv, 85)))
        b = (colv > thr).astype(np.int8)
        idx = np.where((b[1:]==1)&(b[:-1]==0))[0]+1
        keep=[]
        for x in idx:
            if not keep or x-keep[-1]>=6: keep.append(int(x))
        if len(keep)>=3:
            bucket.append(float(np.median(np.diff(keep))))
print("L gaps n", len(Lgaps), "median", float(np.median(Lgaps)) if Lgaps else None, "std", float(np.std(Lgaps)) if Lgaps else None)
print("R gaps n", len(Rgaps), "median", float(np.median(Rgaps)) if Rgaps else None, "std", float(np.std(Rgaps)) if Rgaps else None)
print("L unique-ish", Counter(int(round(g)) for g in Lgaps).most_common(8))
print("R unique-ish", Counter(int(round(g)) for g in Rgaps).most_common(8))
