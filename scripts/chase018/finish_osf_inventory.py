#!/usr/bin/env python3
"""Unzip remaining OSF/Zenodo assets; probe architecture folders; TF check."""
from __future__ import annotations
import json, zipfile, subprocess
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path("/home/chocheng9/kaggle_UMUD")
OUT = ROOT / "data/external/osf_xbawc"
OUT.mkdir(parents=True, exist_ok=True)

def unzip_ok(zname, dest_name):
    z = OUT / zname
    d = OUT / dest_name
    d.mkdir(exist_ok=True)
    if (d / ".ok").exists():
        print("SKIP unzip", zname)
        return
    if not z.exists():
        print("MISSING", z)
        return
    print("UNZIP", z, "->", d, "size", z.stat().st_size)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(d)
    (d / ".ok").write_text("ok\n")
    n = sum(1 for _ in d.rglob("*") if _.is_file())
    print("  files", n)

for z, d in [
    ("apo_images.zip", "labtrain_apo_images"),
    ("apo_masks.zip", "labtrain_apo_masks"),
    ("DLTrack_models.zip", "labtrain_DLTrack_models"),
]:
    try:
        unzip_ok(z, d)
    except Exception as e:
        print("FAIL", z, e)

# list h5
h5 = list((OUT / "labtrain_DLTrack_models").rglob("*.h5")) if (OUT / "labtrain_DLTrack_models").exists() else []
print("h5_models", len(h5))
for p in h5[:30]:
    print(" ", p.relative_to(OUT), p.stat().st_size)

# OSF architecture probe
UA = "umud/1.0"
arch = {}
for label, fid in [
    ("Muscle Architecture - Fascicles", "677812e4a7e8891cbdf19dbc"),
    ("Muscle Architecture - Aponeurosis", "677812f7c52c86f2f869805f"),
    ("Bechmark Models", "6778129ea7e8891cbdf19d5e"),
]:
    url = f"https://api.osf.io/v2/nodes/xbawc/files/osfstorage/{fid}/"
    try:
        req = Request(url, headers={"User-Agent": UA, "Accept": "application/vnd.api+json"})
        with urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        kids = [
            {
                "name": x["attributes"]["name"],
                "kind": x["attributes"]["kind"],
                "size": x["attributes"].get("size"),
            }
            for x in data.get("data", [])
        ]
        arch[label] = kids
        print("PROBE", label, kids)
    except Exception as e:
        arch[label] = {"error": str(e)}
        print("PROBE_FAIL", label, e)

import importlib.util as u
tf = u.find_spec("tensorflow")
keras = u.find_spec("keras")
inv = {
    "fasc_images": (OUT / "fasc_images.zip").stat().st_size if (OUT / "fasc_images.zip").exists() else None,
    "fasc_masks_n": len(list((OUT / "labtrain_fasc_masks").rglob("*.tif"))) if (OUT / "labtrain_fasc_masks").exists() else 0,
    "fasc_images_n": len(list((OUT / "labtrain_fasc_images").rglob("*.tif"))) if (OUT / "labtrain_fasc_images").exists() else 0,
    "apo_images_zip": (OUT / "apo_images.zip").stat().st_size if (OUT / "apo_images.zip").exists() else None,
    "apo_masks_n": len(list((OUT / "labtrain_apo_masks").rglob("*.tif"))) if (OUT / "labtrain_apo_masks").exists() else 0,
    "dltrack_zip": (OUT / "DLTrack_models.zip").stat().st_size if (OUT / "DLTrack_models.zip").exists() else None,
    "dltrack_h5_n": len(h5),
    "dltrack_h5": [str(p.relative_to(OUT)) for p in h5[:40]],
    "architecture_folders": arch,
    "tensorflow": bool(tf),
    "keras": bool(keras),
    "labtrain_comp_overlap_names": 0,
}
(OUT / "osf_xbawc_inventory.json").write_text(json.dumps(inv, indent=2))
print(json.dumps(inv, indent=2)[:3000])
