#!/usr/bin/env python3
"""Check whether AmbrosM FOV crop makes fasc image match mask size."""
from __future__ import annotations
import csv
from pathlib import Path
from collections import Counter
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from umud.data.layout import discover_layout, list_images
from umud.baselines.io import load_split_assignments

layout = discover_layout(Path("data/raw/umud"))
imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
masks = {p.stem: p for p in list_images(layout.fasc_masks)}
rows = load_split_assignments(Path("reports/phase2/split_assignments.jsonl"))
val = [str(r["image_id"]) for r in rows if r.get("task") == "fascicle" and r.get("split") == "val"]

fov = {}
with open("reports/phase3/ambrosm_scale_v2_rows.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("pool") != "fasc":
            continue
        try:
            fov[r["stem"]] = {
                "box": (int(float(r["fov_l"])), int(float(r["fov_t"])), int(float(r["fov_r"])), int(float(r["fov_b"]))),
                "h": int(float(r["height"])),
                "w": int(float(r["width"])),
                "family": r.get("family"),
            }
        except Exception:
            pass

exact = approx = fail = nofov = 0
shape_pairs = Counter()
examples = []
for s in val:
    if s not in imgs or s not in masks:
        continue
    im = Image.open(imgs[s])
    mk = Image.open(masks[s])
    wi, hi = im.size
    wm, hm = mk.size
    shape_pairs[(wi, hi, wm, hm)] += 1
    if (wi, hi) == (wm, hm):
        exact += 1
        continue
    meta = fov.get(s)
    if not meta:
        nofov += 1
        continue
    l, t, r, b = meta["box"]
    # try as-is
    cw, ch = r - l, b - t
    if abs(cw - wm) <= 2 and abs(ch - hm) <= 2:
        approx += 1
        if len(examples) < 5:
            examples.append({"stem": s, "img": (wi, hi), "mask": (wm, hm), "fov_wh": (cw, ch), "box": meta["box"], "family": meta["family"], "mode": "fov_as_is"})
        continue
    # scale FOV from CSV recorded size to actual
    sx, sy = wi / max(1, meta["w"]), hi / max(1, meta["h"])
    ls, ts, rs, bs = int(round(l * sx)), int(round(t * sy)), int(round(r * sx)), int(round(b * sy))
    cw2, ch2 = rs - ls, bs - ts
    if abs(cw2 - wm) <= 3 and abs(ch2 - hm) <= 3:
        approx += 1
        if len(examples) < 8:
            examples.append({"stem": s, "img": (wi, hi), "mask": (wm, hm), "fov_scaled_wh": (cw2, ch2), "box_scaled": (ls, ts, rs, bs), "family": meta["family"], "mode": "fov_scaled"})
    else:
        fail += 1
        if fail <= 5:
            examples.append({"stem": s, "img": (wi, hi), "mask": (wm, hm), "fov_wh": (cw, ch), "fov_scaled": (cw2, ch2), "csv_hw": (meta["h"], meta["w"]), "family": meta["family"], "mode": "FAIL"})

print("exact_same", exact, "fov_matches_mask", approx, "nofov", nofov, "fail", fail)
print("top_shape_pairs", shape_pairs.most_common(12))
for e in examples:
    print(e)

# Also try: mask is content; image is content+HUD — does center crop or proportional resize of mask onto image work?
# Check aspect ratios
print("aspect_img_vs_mask for mismatches:")
for (wi, hi, wm, hm), n in shape_pairs.most_common(8):
    if (wi, hi) != (wm, hm):
        print(f"  img {wi}x{hi} mask {wm}x{hm} n={n} aspect_img={wi/hi:.4f} aspect_mask={wm/hm:.4f} scale_w={wi/wm:.4f} scale_h={hi/hm:.4f}")
