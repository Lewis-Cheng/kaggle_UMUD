#!/usr/bin/env python3
"""Find fasc image folders whose sizes match masks for shared stems."""
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from umud.data.layout import discover_layout, list_images
from umud.baselines.io import load_split_assignments

root = Path("data/raw/umud")
layout = discover_layout(root)
print("layout fasc_imgs", layout.fasc_imgs)
print("layout fasc_masks", layout.fasc_masks)

# list candidate dirs
cands = []
for p in root.rglob("*"):
    if not p.is_dir():
        continue
    name = p.name.lower()
    if "fasc" in name and ("img" in name or "image" in name or "mask" in name):
        cands.append(p)
print("candidate dirs:")
for c in sorted(cands):
    n = len(list(c.glob("*")))
    print(f"  {c} n≈{n}")

masks = {p.stem: p for p in list_images(layout.fasc_masks)}
rows = load_split_assignments(Path("reports/phase2/split_assignments.jsonl"))
val_stems = [str(r["image_id"]) for r in rows if r.get("task") == "fascicle" and r.get("split") == "val"][:40]

# For each candidate image dir, check match rate vs masks
for d in sorted(cands):
    if "mask" in d.name.lower():
        continue
    imgs = {}
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg"):
        for p in d.glob(ext):
            imgs[p.stem] = p
        for p in d.glob(ext.upper()):
            imgs[p.stem] = p
    if len(imgs) < 50:
        # maybe one more nesting
        for sub in d.iterdir() if d.is_dir() else []:
            if sub.is_dir():
                for ext in ("*.tif", "*.tiff", "*.png"):
                    for p in sub.glob(ext):
                        imgs[p.stem] = p
    if not imgs:
        continue
    same = mism = missing = 0
    for s in val_stems:
        if s not in masks:
            continue
        if s not in imgs:
            missing += 1
            continue
        wi, hi = Image.open(imgs[s]).size
        wm, hm = Image.open(masks[s]).size
        if (wi, hi) == (wm, hm):
            same += 1
        else:
            mism += 1
    print(f"DIR {d} n_imgs={len(imgs)} val40 same={same} mism={mism} missing={missing}")
