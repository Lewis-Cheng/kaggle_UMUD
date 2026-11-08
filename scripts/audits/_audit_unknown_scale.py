#!/usr/bin/env python3
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path.home() / "kaggle_UMUD"
ROWS = ROOT / "reports/phase3/ambrosm_scale_rows.csv"
DATA = ROOT / "data/raw/umud"

def find_img(name: str, pool: str):
    for p in [DATA/pool/name, DATA/f"{pool}_img"/name, DATA/"images"/pool/name, DATA/pool/"images"/name]:
        if p.exists():
            return p
    hits = list(DATA.rglob(name))
    return hits[0] if hits else None

rows = list(csv.DictReader(ROWS.open()))
unk = [r for r in rows if r.get("provenance") == "unknown"]
print("total", len(rows), "unknown", len(unk))
print("unk by pool", dict(Counter(r["pool"] for r in unk)))
for fam, n in Counter(r["family"] for r in unk).most_common(20):
    print(f"  {fam}: {n}")
print("layout", DATA, DATA.exists(), [p.name for p in DATA.iterdir()][:30] if DATA.exists() else None)

fam800 = [r for r in unk if r["family"] == "rgb_800x1200_other"]
counters = Counter()
n_check = min(120, len(fam800))
sample_notes = []
for i, r in enumerate(fam800[:n_check]):
    p = find_img(r["name"], r["pool"])
    if p is None:
        counters["missing"] += 1
        continue
    a = np.asarray(Image.open(p))
    if a.ndim != 3 or a.shape[:2] != (800, 1200):
        counters[f"bad_shape"] += 1
        continue
    rt = float(a[87, 1147:1157].mean())
    lt = float(a[42, 67:74, 0].mean())
    rt_m = float(a[87, 43:53].mean())
    lt_m = float(a[42, 1126:1133, 0].mean())
    right_exact_m = bool((a[87, 43:53] == 175).all())
    left_exact_m = bool((a[42, 1126:1133, 0] > 115).all())
    # also try slightly different mirror columns sometimes used
    rt_m2 = float(a[87, 40:55].mean())
    tag = []
    if rt > 100: tag.append("right")
    if lt > 100: tag.append("left")
    if rt_m > 100 or right_exact_m: tag.append("right_hflip")
    if lt_m > 100 or left_exact_m: tag.append("left_hflip")
    if not tag and rt_m2 > 100: tag.append("right_hflip_loose")
    key = "+".join(tag) if tag else "none"
    counters[key] += 1
    if len(sample_notes) < 12:
        col49 = a[:, 49].mean(axis=-1)
        col50 = a[:, 50].mean(axis=-1)
        col1150 = a[:, 1150].mean(axis=-1)
        sample_notes.append(f"{r['pool']} {r['name']} key={key} rt={rt:.0f} lt={lt:.0f} rt_m={rt_m:.0f} lt_m={lt_m:.0f} col49max={col49.max():.0f} col50max={col50.max():.0f} col1150max={col1150.max():.0f}")

print(f"\nfingerprint tags first {n_check}:", dict(counters))
for s in sample_notes:
    print(" ", s)

# deeper: for images tagged none, scan left/right columns for tick-like brightness
none_rows = []
for r in fam800[:n_check]:
    p = find_img(r["name"], r["pool"])
    if p is None: continue
    a = np.asarray(Image.open(p))
    if a.ndim != 3 or a.shape[:2] != (800,1200): continue
    rt_m = float(a[87, 43:53].mean())
    lt_m = float(a[42, 1126:1133, 0].mean())
    rt = float(a[87, 1147:1157].mean())
    lt = float(a[42, 67:74, 0].mean())
    if max(rt, lt, rt_m, lt_m) <= 100:
        none_rows.append((r, a, p))

print(f"\nnone-fingerprint among checked: {len(none_rows)}")
for r, a, p in none_rows[:8]:
    gray = a.mean(axis=2)
    best = []
    for name, band in [("L", gray[:,:30]), ("R", gray[:,-30:]), ("B", gray[-30:,:]), ("T", gray[:30,:])]:
        prof = band.mean(axis=1 if name in "LR" else 0)
        best.append((name, float(prof.max()), float(np.percentile(prof,95)), int(np.argmax(prof))))
    best.sort(key=lambda x: -x[2])
    print(f" {r['pool']} {r['name']} sides={best[:3]}")
    # column-wise max near left edge
    left_cols = [float(a[:, c].mean(axis=-1).max()) for c in range(0, 80)]
    right_cols = [float(a[:, c].mean(axis=-1).max()) for c in range(1120, 1200)]
    print(f"  left_col_maxpeak={max(left_cols):.0f}@c={int(np.argmax(left_cols))} right_col_maxpeak={max(right_cols):.0f}@c={1120+int(np.argmax(right_cols))}")

for fam in ["other_1080x1640_RGB","other_500x760_RGB","other_652x800_RGB","other_600x800_RGB","other_556x660_RGB","other_688x1234_RGB"]:
    items = [r for r in unk if r["family"]==fam][:5]
    print(f"\n=== {fam} ===")
    for r in items:
        p = find_img(r["name"], r["pool"])
        if p is None:
            print(" missing", r["name"]); continue
        a = np.asarray(Image.open(p))
        gray = a.mean(axis=2) if a.ndim==3 else a.astype(np.float64)
        for side, axis, band in [("L",1, gray[:,:24]), ("R",1, gray[:,-24:]), ("B",0, gray[-24:,:]), ("T",0, gray[:24,:])]:
            prof = band.mean(axis=axis)
            x = prof - prof.mean()
            lag = None
            if x.std() > 1:
                c = np.correlate(x, x, mode="full")
                mid = len(c)//2
                seg = c[mid+12: mid+min(220, len(x)//3)]
                if len(seg):
                    lag = 12 + int(np.argmax(seg))
            if side == "L":
                print(f" {r['pool']} {r['name']} shape={a.shape}", end="")
            print(f" {side}:max={prof.max():.0f}/p95={np.percentile(prof,95):.0f}/lag={lag}", end="")
        print()
