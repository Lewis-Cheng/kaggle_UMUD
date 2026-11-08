#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import json
import numpy as np
from PIL import Image

root = Path("/home/chocheng9/kaggle_UMUD/data/raw/umud")
for kind in ["apo_masks_v1/apo_masks_new_model_v1", "fasc_masks_v1/fasc_masks_new_model_v1"]:
    folder = root / kind
    paths = sorted(folder.glob("*.tif"))
    print("KIND", kind, "n", len(paths))
    for p in paths[:5]:
        a = np.array(Image.open(p))
        u = np.unique(a)
        print(p.name, a.shape, a.dtype, "unique", u[:20], "nnz", int((a > 0).sum()))

apo = {p.stem for p in (root / "apo_imgs_v1/apo_images_new_model_v1").glob("*") if p.is_file()}
fasc = {p.stem for p in (root / "fasc_imgs_v1/fasc_images_new_model_v1").glob("*") if p.is_file()}
print("overlap", len(apo & fasc), "apo_only", len(apo - fasc), "fasc_only", len(fasc - apo))

rows = []
for line in open("/home/chocheng9/kaggle_UMUD/reports/phase2/split_assignments.jsonl"):
    rows.append(json.loads(line))
print("split", Counter((r["task"], r["split"]) for r in rows))
