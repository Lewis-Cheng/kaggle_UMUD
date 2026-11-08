#!/usr/bin/env python3
"""Overlap LabTrain masks vs competition; optionally wait for images then launch train."""
from __future__ import annotations
import json, time, subprocess, sys
from pathlib import Path

ROOT = Path("/home/chocheng9/kaggle_UMUD")
OUT = ROOT / "data/external/osf_xbawc"

def names(d: Path):
    if not d.exists():
        return set()
    return {p.name for p in d.rglob("*") if p.suffix.lower() in {".tif", ".png", ".jpg"}}

lab_m = names(OUT / "labtrain_fasc_masks")
comp_i = names(ROOT / "data/raw/umud/fasc_imgs_v1")
comp_m = names(ROOT / "data/raw/umud/fasc_masks_v1")
print("lab_masks", len(lab_m), "comp_imgs", len(comp_i), "comp_masks", len(comp_m))
print("overlap lab_masks&comp_imgs", len(lab_m & comp_i))
print("overlap lab_masks&comp_masks", len(lab_m & comp_m))
print("sample lab", sorted(lab_m)[:8])

# wait for fasc_images.zip complete (~367MB)
expect = 367494790
z = OUT / "fasc_images.zip"
part = OUT / "fasc_images.zip.part"
print("waiting for fasc_images.zip ...")
for i in range(360):  # up to ~60 min
    if z.exists() and z.stat().st_size >= expect * 0.98:
        print("READY", z, z.stat().st_size)
        break
    sz = part.stat().st_size if part.exists() else (z.stat().st_size if z.exists() else 0)
    if i % 6 == 0:
        print(f"  t={i*10}s size={sz/1e6:.1f}MB / {expect/1e6:.1f}MB", flush=True)
    time.sleep(10)
else:
    print("TIMEOUT waiting for fasc_images")
    sys.exit(2)

# unzip
img_dir = OUT / "labtrain_fasc_images"
img_dir.mkdir(exist_ok=True)
if not (img_dir / ".ok").exists():
    subprocess.check_call(["unzip", "-o", "-q", str(z), "-d", str(img_dir)])
    (img_dir / ".ok").write_text("ok\n")
print("images", len(list(img_dir.rglob("*.tif"))))

# also wait briefly for apo images optional
# launch train
log = ROOT / "reports/phase3/chase018/train_eval_labtrain_fasc.log"
cmd = [
    str(ROOT / ".venv/bin/python"), "-u",
    str(ROOT / "scripts/chase018/train_eval_labtrain_fasc.py"),
    "--img-dir", str(img_dir),
    "--mask-dir", str(OUT / "labtrain_fasc_masks"),
    "--epochs", "25",
    "--batch", "8",
    "--device", "cuda:0",
]
print("LAUNCH", cmd, flush=True)
with open(log, "w") as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
print("PID", p.pid)
(ROOT / "reports/phase3/chase018/labtrain_watch.json").write_text(json.dumps({
    "lab_masks": len(lab_m),
    "comp_overlap_imgs": len(lab_m & comp_i),
    "comp_overlap_masks": len(lab_m & comp_m),
    "train_pid": p.pid,
    "img_dir": str(img_dir),
}, indent=2))
