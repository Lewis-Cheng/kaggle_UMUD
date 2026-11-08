#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
export KAGGLE_API_TOKEN=$(tr -d "\n\r " < ~/.kaggle/access_token)
# Remove broken lewischeng json so token is used
mv ~/.kaggle/kaggle.json ~/.kaggle/kaggle.json.disabled 2>/dev/null || true
cd ~/kaggle_UMUD
OUT=reports/phase3/rl_track_b
/home/chocheng9/images_reduced/.venv/bin/python - <<'PY'
import os, sys
# Ensure user site kaggle is importable
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.10/site-packages"))
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
# Force token auth
os.environ["KAGGLE_API_TOKEN"] = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
# Some versions read CONFIG_NAME
try:
    api.authenticate()
except Exception as e:
    print("authenticate error:", e)
    # manual: set api.token if attribute exists
    tok = os.environ["KAGGLE_API_TOKEN"]
    if hasattr(api, "config_values"):
        api.config_values = {"username": "token", "key": tok}
    try:
        api.authenticate()
    except Exception as e2:
        print("authenticate2:", e2)
        # last resort: set headers
        api.api_client.set_default_header("Authorization", f"Bearer {tok}")

path = "reports/phase3/rl_track_b/submission_track_b_v5_rl.csv"
comp = "umud-challenge-muscle-architecture-in-ultrasound-data"
msg = "trackB_v5_rl_hybrid"
print("submitting", path)
try:
    result = api.competition_submit(path, msg, comp)
    print("RESULT:", result)
except Exception as e:
    print("SUBMIT_ERROR:", type(e), e)
    import traceback; traceback.print_exc()
PY