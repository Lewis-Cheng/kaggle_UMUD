#!/bin/bash
set -e
cd ~/kaggle_UMUD
OUT=reports/phase3/rl_track_b
TOKEN=$(tr -d "\n\r " < ~/.kaggle/access_token)
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
FILE=$OUT/submission_track_b_v5_rl.csv
FNAME=submission_track_b_v5_rl.csv
CLEN=$(stat -c%s "$FILE")
MTIME=$(stat -c%Y "$FILE")
MSG=trackB_v5_rl_hybrid

echo "step1 form fileName"
curl -sS -o /tmp/kg_url.json -w "HTTP %{http_code}\n" \
  -X POST -H "Authorization: Bearer $TOKEN" \
  -F "fileName=$FNAME" \
  "https://www.kaggle.com/api/v1/competitions/$COMP/submissions/url/$CLEN/$MTIME"
cat /tmp/kg_url.json; echo

python3 - <<'PY'
import json, os, subprocess, pathlib, time
TOKEN = pathlib.Path.home().joinpath(".kaggle/access_token").read_text().strip()
COMP = "umud-challenge-muscle-architecture-in-ultrasound-data"
FILE = "reports/phase3/rl_track_b/submission_track_b_v5_rl.csv"
MSG = "trackB_v5_rl_hybrid"
url = json.load(open("/tmp/kg_url.json"))
print("URL", url)
create = url["createUrl"]
token = url["token"]
# PUT file with content-type
cmd = [
    "curl","-sS","-o","/tmp/kg_up.bin","-w","%{http_code}",
    "-X","PUT",
    "-H","Content-Type: text/csv",
    "--data-binary",f"@{FILE}",
    create,
]
code = subprocess.check_output(cmd).decode().strip()
body = open("/tmp/kg_up.bin","rb").read()[:500]
print("upload http", code, body)
# submit form
cmd2 = [
    "curl","-sS","-o","/tmp/kg_sub.json","-w","%{http_code}",
    "-X","POST","-H",f"Authorization: Bearer {TOKEN}",
    "-F",f"blobFileTokens={token}",
    "-F",f"submissionDescription={MSG}",
    f"https://www.kaggle.com/api/v1/competitions/submissions/submit/{COMP}",
]
code2 = subprocess.check_output(cmd2).decode().strip()
print("submit http", code2, open("/tmp/kg_sub.json").read())
for i in range(8):
    time.sleep(12)
    listing = subprocess.check_output([
        "curl","-sS","-H",f"Authorization: Bearer {TOKEN}",
        f"https://www.kaggle.com/api/v1/competitions/submissions/list/{COMP}",
    ]).decode()
    open("reports/phase3/rl_track_b/kaggle_submissions_list.json","w").write(listing)
    data = json.loads(listing)
    top = data[0] if data else {}
    print(f"poll{i}", top.get("description"), top.get("fileName"), top.get("status") or top.get("publicScore"), top.get("errorDescription"), "score=", top.get("publicScore"))
    # break when scored or error cleared meaningfully
    if top.get("publicScore") not in (None, ""):
        break
    err = top.get("errorDescription") or ""
    if err and "Invalid submission file extension" not in err and top.get("description")==MSG:
        # still waiting or other error
        if "Pending" not in str(top) and top.get("publicScore")=="" and i>=2:
            # keep waiting for score
            pass
    if top.get("description")==MSG and not err and top.get("publicScore"):
        break
print("DONE")
print(json.dumps(data[0] if data else {}, indent=2)[:1200])
PY
TZ=Asia/Hong_Kong date '+%Y-%m-%d %H:%M:%S %Z'