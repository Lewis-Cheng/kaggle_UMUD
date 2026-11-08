#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
TOKEN=$(cat ~/.kaggle/access_token | tr -d '\n\r ')
# Write kaggle.json using access_token as key; username probe
python3 - <<PY
import json, os
from pathlib import Path
token = Path.home().joinpath(".kaggle/access_token").read_text().strip()
# try common usernames
for user in ["chocheng9", "lewischeng", "paulritsche"]:
    pass
Path.home().joinpath(".kaggle/kaggle.json").write_text(json.dumps({"username":"chocheng9","key":token}))
os.chmod(Path.home()/".kaggle"/"kaggle.json", 0o600)
print("wrote kaggle.json with access_token key, username=chocheng9")
PY
# upgrade kaggle if needed for token file
pip install -U --user "kaggle>=1.6" -q 2>/dev/null || true
kaggle --version
# probe auth
kaggle config view 2>&1 | head -20
kaggle competitions list 2>&1 | head -5
echo '--- submissions ---'
kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>&1 | head -10