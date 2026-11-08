#!/bin/bash
set -euo pipefail
ROOT=/home/chocheng9/kaggle_UMUD
OUT=$ROOT/data/external/osf_xbawc
LOG=$ROOT/reports/phase3/chase018/fetch_labtrain.sh.log
mkdir -p "$OUT"
cd "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date)"

dl() {
  local url="$1" dest="$2" expect="${3:-0}"
  if [[ -f "$dest" && "$expect" -gt 0 ]]; then
    sz=$(stat -c%s "$dest")
    if [[ "$sz" -ge $((expect * 95 / 100)) ]]; then
      echo "SKIP $dest ($sz)"
      return 0
    fi
  fi
  echo "GET $dest"
  curl -L --fail --retry 5 --retry-delay 2 -o "${dest}.part" "$url"
  mv "${dest}.part" "$dest"
  echo "DONE $dest $(stat -c%s "$dest")"
}

# OSF LabTrain (Example Dataset Structure) — quoted URLs
dl 'https://files.de-1.osf.io/v1/resources/xbawc/providers/osfstorage/66ffe0c7f97b69ebae32da26?action=download&version=1&direct&display=download' fasc_masks.zip 7118062
dl 'https://files.de-1.osf.io/v1/resources/xbawc/providers/osfstorage/66ffe0c8cf2abd8a8f32d671?action=download&version=1&direct&display=download' apo_masks.zip 4447512
dl 'https://files.de-1.osf.io/v1/resources/xbawc/providers/osfstorage/66ffe0aeeb7c27da94fcb442?action=download&version=1&direct&display=download' fasc_images.zip 367494790
dl 'https://files.de-1.osf.io/v1/resources/xbawc/providers/osfstorage/66ffe0ad91affabc9632d84e?action=download&version=1&direct&display=download' apo_images.zip 305141391

# Zenodo DL_Track pretrained models (apo+fasc h5)
dl 'https://zenodo.org/records/7318089/files/DLTrack_models.zip?download=1' DLTrack_models.zip 480000000

# unzip
for z in fasc_masks.zip apo_masks.zip fasc_images.zip apo_images.zip DLTrack_models.zip; do
  if [[ -f "$z" ]]; then
    d="labtrain_${z%.zip}"
    mkdir -p "$d"
    if [[ ! -f "$d/.ok" ]]; then
      echo "UNZIP $z -> $d"
      unzip -o -q "$z" -d "$d"
      touch "$d/.ok"
    fi
    echo "LIST $d"; find "$d" -maxdepth 3 -type d | head -40
    find "$d" -type f | head -20
  fi
done

# probe empty architecture folders via API (no &)
python3 - <<'PY'
import json
from urllib.request import Request, urlopen
UA="umud/1.0"
for label,fid in [
 ("Muscle Architecture - Fascicles","677812e4a7e8891cbdf19dbc"),
 ("Muscle Architecture - Aponeurosis","677812f7c52c86f2f869805f"),
]:
  url=f"https://api.osf.io/v2/nodes/xbawc/files/osfstorage/{fid}/"
  req=Request(url, headers={"User-Agent":UA,"Accept":"application/vnd.api+json"})
  with urlopen(req, timeout=60) as r:
    d=json.loads(r.read().decode())
  kids=[(x["attributes"]["name"], x["attributes"]["kind"], x["attributes"].get("size")) for x in d.get("data",[])]
  print("PROBE", label, kids)
PY

python3 - <<'PY'
import importlib.util as u
print("tensorflow", u.find_spec("tensorflow"))
print("keras", u.find_spec("keras"))
PY

ls -lah
echo "DONE $(date)"
