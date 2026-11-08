#!/usr/bin/env python3
from pathlib import Path
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
mismatch = same = 0
samples = []
for s in val:
    if s not in imgs or s not in masks:
        continue
    wi, hi = Image.open(imgs[s]).size
    wm, hm = Image.open(masks[s]).size
    if (wi, hi) != (wm, hm):
        mismatch += 1
        if len(samples) < 10:
            samples.append({"stem": s, "img": (wi, hi), "mask": (wm, hm), "ip": str(imgs[s]), "mp": str(masks[s])})
    else:
        same += 1
print("fasc_imgs", layout.fasc_imgs)
print("fasc_masks", layout.fasc_masks)
print("same", same, "mismatch", mismatch)
for x in samples:
    print(x)
