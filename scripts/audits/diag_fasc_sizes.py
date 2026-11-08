#!/usr/bin/env python3
"""Compare actual fasc image sizes vs AmbrosM scale CSV FOV."""
from __future__ import annotations
import csv, json, random
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

ROOT = Path.home() / "kaggle_UMUD"
layout_fasc = ROOT / "data/raw/umud"
# find fasc imgs
cands = list(ROOT.glob("data/raw/umud/**/fasc*"))
print("fasc dirs", cands[:20])

# load scale
rows = {}
with open(ROOT / "reports/phase3/ambrosm_scale_v2_rows.csv") as f:
    for r in csv.DictReader(f):
        if r.get("pool") == "fasc":
            rows[r["stem"]] = r

# discover images via layout
import sys
sys.path.insert(0, str(ROOT / "src"))
from umud.data.layout import discover_layout, list_images
layout = discover_layout(ROOT / "data/raw/umud")
imgs = list(list_images(layout.fasc_imgs))
masks = {p.stem: p for p in list_images(layout.fasc_masks)}
print("n_imgs", len(imgs), "n_masks", len(masks), "n_scale_fasc", len(rows))

size_c = Counter()
mismatch = 0
fov_fit_raw = 0
fov_fit_scaled = 0
gt_drop_raw = []
gt_drop_scaled = []
samples = imgs[:]
random.seed(0)
random.shuffle(samples)
samples = samples[:80]

for ip in samples:
    im = Image.open(ip)
    w, h = im.size
    size_c[(h, w)] += 1
    r = rows.get(ip.stem)
    if not r:
        continue
    ch, cw = int(float(r["height"])), int(float(r["width"]))
    if (h, w) != (ch, cw):
        mismatch += 1
    try:
        l,t,rr,b = int(float(r["fov_l"])), int(float(r["fov_t"])), int(float(r["fov_r"])), int(float(r["fov_b"]))
    except (TypeError, ValueError):
        continue
    mp = masks.get(ip.stem)
    if mp is None:
        continue
    m = np.asarray(Image.open(mp).convert("L")) > 0
    # raw FOV
    if rr <= w and b <= h and l >= 0 and t >= 0:
        fov_fit_raw += 1
        crop = m[t:b, l:rr]
        gt_drop_raw.append(1 - crop.sum() / max(1, m.sum()))
    # scaled FOV from CSV recorded size to actual
    sx, sy = w / max(1, cw), h / max(1, ch)
    ls, ts, rs, bs = int(l*sx), int(t*sy), int(rr*sx), int(b*sy)
    ls, ts = max(0, ls), max(0, ts)
    rs, bs = min(w, rs), min(h, bs)
    if rs > ls + 8 and bs > ts + 8:
        fov_fit_scaled += 1
        crop = m[ts:bs, ls:rs]
        gt_drop_scaled.append(1 - crop.sum() / max(1, m.sum()))

print("actual sizes", size_c.most_common(10))
print("csv_size_mismatch_in_sample", mismatch, "/", len(samples))
print("fov_fit_raw", fov_fit_raw, "mean_gt_drop", float(np.mean(gt_drop_raw)) if gt_drop_raw else None)
print("fov_fit_scaled", fov_fit_scaled, "mean_gt_drop", float(np.mean(gt_drop_scaled)) if gt_drop_scaled else None)
print("gt_drop_scaled p50", float(np.median(gt_drop_scaled)) if gt_drop_scaled else None)
print("gt_drop_raw p50", float(np.median(gt_drop_raw)) if gt_drop_raw else None)
