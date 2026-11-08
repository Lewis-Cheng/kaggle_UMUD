#!/usr/bin/env python3
"""Inventory + download UMUD OSF xbawc Benchmark Models / LabTrain-like assets.

Nested-safe: MultExp is eval-only. No Kaggle.
"""
from __future__ import annotations

import json
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path("/home/chocheng9/kaggle_UMUD")
OUT = ROOT / "data/external/osf_xbawc"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.osf.io/v2"
UA = "umud-chase018-labtrain/1.0"


def get_json(url: str):
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/vnd.api+json"})
    with urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def list_folder(node: str, folder_id: str | None = None):
    if folder_id:
        url = f"{API}/nodes/{node}/files/osfstorage/{folder_id}/?page[size]=100"
    else:
        url = f"{API}/nodes/{node}/files/osfstorage/?page[size]=100"
    items = []
    while url:
        d = get_json(url)
        items.extend(d.get("data", []))
        url = (d.get("links") or {}).get("next")
    return items


def walk(node: str, folder_id: str | None = None, prefix: str = ""):
    out = []
    for x in list_folder(node, folder_id):
        a = x["attributes"]
        name = a["name"]
        kind = a["kind"]
        path = prefix + name + ("/" if kind == "folder" else "")
        entry = {
            "id": x["id"],
            "name": name,
            "kind": kind,
            "path": path,
            "size": a.get("size"),
            "download": (x.get("links") or {}).get("download"),
        }
        out.append(entry)
        if kind == "folder":
            out.extend(walk(node, x["id"], path))
    return out


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP exists {dest} ({dest.stat().st_size})", flush=True)
        return dest
    print(f"GET {url} -> {dest}", flush=True)
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        n = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
            if n % (50 << 20) == 0:
                print(f"  ... {n/1e6:.1f} MB", flush=True)
    print(f"DONE {dest} {dest.stat().st_size}", flush=True)
    return dest


def maybe_unzip(zpath: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".unzipped_ok"
    if marker.exists():
        print(f"SKIP unzip {dest}", flush=True)
        return
    print(f"UNZIP {zpath} -> {dest}", flush=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest)
    marker.write_text("ok\n")


def interesting(name: str, path: str) -> bool:
    s = (name + " " + path).lower()
    keys = [
        "labtrain",
        "lab_train",
        "architecture",
        "aponeurosis",
        "fasc",
        "muscle architecture",
        "drawn",
        "expdrawn",
        "train",
        "model",
        "deepacsa",
        "dl_track",
        "dltrack",
        "benchmark",
    ]
    return any(k in s for k in keys)


def main() -> int:
    print("Walking OSF xbawc ...", flush=True)
    tree = walk("xbawc")
    inv_path = OUT / "osf_xbawc_inventory.json"
    inv_path.write_text(json.dumps(tree, indent=2))
    print(f"inventory n={len(tree)} -> {inv_path}", flush=True)
    for e in tree:
        print(f"  [{e['kind']}] {e['path']} size={e['size']} id={e['id']}", flush=True)

    # Prefer files under Bechmark Models / Example Annotated / Expert (already have expert)
    targets = []
    for e in tree:
        if e["kind"] != "file":
            continue
        if not interesting(e["name"], e["path"]):
            continue
        # skip huge unrelated if any; keep zips/h5/pt
        if not re.search(r"\.(zip|7z|h5|pt|pth|onnx|npz|tar\.gz|tgz)$", e["name"], re.I):
            # also keep readmes
            if not re.search(r"(readme|license|csv|xlsx|json)$", e["name"], re.I):
                continue
        targets.append(e)

    # Always include everything under Bechmark Models
    for e in tree:
        if e["kind"] == "file" and e["path"].lower().startswith("bechmark models/"):
            if e not in targets:
                targets.append(e)

    print(f"download candidates={len(targets)}", flush=True)
    downloaded = []
    for e in targets:
        url = e.get("download")
        if not url:
            # construct from id
            url = f"https://osf.io/download/{e['id']}/"
        dest = OUT / e["path"].replace("/", "__")
        try:
            download(url, dest)
            downloaded.append({"path": e["path"], "local": str(dest), "size": dest.stat().st_size})
            if dest.suffix.lower() == ".zip":
                maybe_unzip(dest, OUT / ("unzipped_" + dest.stem))
        except Exception as ex:
            print(f"FAIL {e['path']}: {ex}", flush=True)

    # Also fetch Zenodo DL_Track models (fasc+apo h5) — outside Frundles
    zenodo_models = "https://zenodo.org/records/7318089/files/DLTrack_models.zip?download=1"
    zdest = OUT / "DLTrack_models.zip"
    try:
        download(zenodo_models, zdest)
        maybe_unzip(zdest, OUT / "unzipped_DLTrack_models")
        downloaded.append({"path": "zenodo:DLTrack_models.zip", "local": str(zdest), "size": zdest.stat().st_size})
    except Exception as ex:
        print(f"FAIL zenodo models: {ex}", flush=True)

    summary = {
        "inventory_n": len(tree),
        "downloaded": downloaded,
        "inventory_path": str(inv_path),
        "out": str(OUT),
    }
    (OUT / "fetch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
