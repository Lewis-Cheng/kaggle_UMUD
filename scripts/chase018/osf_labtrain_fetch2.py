#!/usr/bin/env python3
"""Fast LabTrain fetch: Example Dataset Structure masks/images + Zenodo DLTrack models."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path("/home/chocheng9/kaggle_UMUD")
OUT = ROOT / "data/external/osf_xbawc"
OUT.mkdir(parents=True, exist_ok=True)
UA = "umud-chase018-labtrain/1.0"

# Direct osfstorage file IDs from inventory
FILES = [
    ("fasc_images.zip", "66ffe0aeeb7c27da94fcb442", 367494790),
    ("apo_images.zip", "66ffe0ad91affabc9632d84e", 305141391),
    ("fasc_masks.zip", "66ffe0c7f97b69ebae32da26", 7118062),
    ("apo_masks.zip", "66ffe0c8cf2abd8a8f32d671", 4447512),
    ("dictionary_DL_Track_US_2023.py", "677854d7906cebfce8e6dd30", 3976),
    ("Readme_example.md", "66ffe6677d7bc650e0fcb4ee", 2742),
]


def download(url: str, dest: Path, expect: int | None = None):
    if dest.exists() and expect and dest.stat().st_size >= expect * 0.98:
        print(f"SKIP {dest} size={dest.stat().st_size}", flush=True)
        return
    if dest.exists() and (expect is None) and dest.stat().st_size > 0:
        print(f"SKIP {dest} size={dest.stat().st_size}", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"GET {url}", flush=True)
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=1200) as r, open(tmp, "wb") as f:
        n = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
            if n % (20 << 20) < (1 << 20):
                print(f"  {dest.name}: {n/1e6:.1f} MB", flush=True)
    tmp.rename(dest)
    print(f"DONE {dest} {dest.stat().st_size}", flush=True)


def unzip(z: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".ok"
    if marker.exists():
        print(f"SKIP unzip {dest}", flush=True)
        return
    print(f"UNZIP {z} -> {dest}", flush=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    marker.write_text("ok\n")
    # list shallow
    kids = list(dest.iterdir())[:20]
    print("  kids:", [p.name for p in kids], flush=True)


def main():
    landed = []
    for name, fid, size in FILES:
        url = f"https://files.de-1.osf.io/v1/resources/xbawc/providers/osfstorage/{fid}?action=download&version=1&direct&display=download"
        dest = OUT / name
        try:
            download(url, dest, size)
            landed.append({"name": name, "size": dest.stat().st_size, "id": fid})
            if name.endswith(".zip"):
                unzip(dest, OUT / ("labtrain_" + name.replace(".zip", "")))
        except Exception as ex:
            print(f"FAIL {name}: {ex}", flush=True)
            # fallback osf.io/download by file id
            try:
                download(f"https://osf.io/download/{fid}/", dest, size)
                landed.append({"name": name, "size": dest.stat().st_size, "id": fid, "via": "fallback"})
                if name.endswith(".zip"):
                    unzip(dest, OUT / ("labtrain_" + name.replace(".zip", "")))
            except Exception as ex2:
                print(f"FAIL2 {name}: {ex2}", flush=True)

    # Zenodo DLTrack models (apo+fasc h5)
    zurl = "https://zenodo.org/records/7318089/files/DLTrack_models.zip?download=1"
    zdest = OUT / "DLTrack_models.zip"
    try:
        download(zurl, zdest, 480_000_000)
        unzip(zdest, OUT / "labtrain_DLTrack_models")
        landed.append({"name": "DLTrack_models.zip", "size": zdest.stat().st_size})
    except Exception as ex:
        print(f"FAIL zenodo: {ex}", flush=True)

    # Probe Muscle Architecture folders
    import json as _json
    from urllib.request import Request as _R, urlopen as _U

    for label, fid in [
        ("Muscle Architecture - Fascicles", "677812e4a7e8891cbdf19dbc"),
        ("Muscle Architecture - Aponeurosis", "677812f7c52c86f2f869805f"),
        ("ACSA Vastus Lateralis", "677812c6fd0ad5417ce6dc5b"),
        ("ACSA Rectus Femoris", "677812a6701e529795e6db24"),
    ]:
        url = f"https://api.osf.io/v2/nodes/xbawc/files/osfstorage/{fid}/"
        try:
            req = _R(url, headers={"User-Agent": UA, "Accept": "application/vnd.api+json"})
            with _U(req, timeout=60) as r:
                d = _json.loads(r.read().decode())
            kids = [(x["attributes"]["name"], x["attributes"]["kind"], x["attributes"].get("size"), x["id"]) for x in d.get("data", [])]
            print(f"PROBE {label}: {kids}", flush=True)
            (OUT / f"probe_{fid}.json").write_text(_json.dumps(d, indent=2))
        except Exception as ex:
            print(f"PROBE FAIL {label}: {ex}", flush=True)

    summary = {"landed": landed, "out": str(OUT)}
    (OUT / "labtrain_fetch2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
