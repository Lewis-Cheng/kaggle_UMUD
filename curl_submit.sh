#!/bin/bash
set -e
cd ~/kaggle_UMUD
OUT=reports/phase3/rl_track_b
TOKEN=$(tr -d "\n\r " < ~/.kaggle/access_token)
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
FILE=$OUT/submission_track_b_v5_rl.csv
FNAME=$(basename "$FILE")
CLEN=$(stat -c%s "$FILE")
MTIME=$(stat -c%Y "$FILE")
MSG=trackB_v5_rl_hybrid
AUTH="Authorization: Bearer $TOKEN"

echo "step1 url POST /competitions/$COMP/submissions/url/$CLEN/$MTIME"
curl -sS -o /tmp/kg_url.json -w "HTTP %{http_code}\n" \
  -X POST -H "$AUTH" -H "Content-Type: application/json" \
  -d "{\"fileName\":\"$FNAME\"}" \
  "https://www.kaggle.com/api/v1/competitions/$COMP/submissions/url/$CLEN/$MTIME"
echo "URL_BODY:"; cat /tmp/kg_url.json; echo

python3 - <<'PY'
import json, os, subprocess, sys
url = json.load(open("/tmp/kg_url.json"))
print("keys", url.keys() if isinstance(url, dict) else type(url))
print(url)
open("/tmp/kg_url_parsed.json","w").write(json.dumps(url))
PY

# continue in python for clarity
python3 - <<'PY'
import json, os, subprocess, pathlib, time
TOKEN = pathlib.Path.home().joinpath(".kaggle/access_token").read_text().strip()
COMP = "umud-challenge-muscle-architecture-in-ultrasound-data"
FILE = "reports/phase3/rl_track_b/submission_track_b_v5_rl.csv"
MSG = "trackB_v5_rl_hybrid"
url = json.load(open("/tmp/kg_url.json"))
if not isinstance(url, dict) or "createUrl" not in url and "token" not in str(url):
    # try query param style fileName
    import urllib.request
    clen = os.path.getsize(FILE)
    mtime = int(os.path.getmtime(FILE))
    fname = os.path.basename(FILE)
    req = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/competitions/{COMP}/submissions/url/{clen}/{mtime}?fileName={fname}",
        data=b"{}",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            url = json.loads(r.read().decode())
            print("retry url", url)
    except Exception as e:
        print("retry failed", e)
        if hasattr(e, "read"):
            print(e.read()[:500])
        raise SystemExit(1)

print("URL result:", url)
create = url.get("createUrl") or url.get("create_url")
token = url.get("token")
is_complete = "isComplete" in url
print("createUrl", create, "token", token, "isComplete", is_complete)

if is_complete:
    # old path
    parts = create.rstrip("/").split("/")
    guid, content_length, last_mod = parts[-3], parts[-2], parts[-1]
    print("old upload", guid, content_length, last_mod)
    cmd = [
        "curl","-sS","-o","/tmp/kg_up.json","-w","%{http_code}",
        "-X","POST","-H",f"Authorization: Bearer {TOKEN}",
        "-F",f"file=@{FILE}",
        f"https://www.kaggle.com/api/v1/competitions/submissions/upload/{guid}/{content_length}/{last_mod}",
    ]
    code = subprocess.check_output(cmd).decode().strip()
    print("upload http", code, open("/tmp/kg_up.json").read()[:400])
    upload_token = json.load(open("/tmp/kg_up.json"))["token"]
else:
    # new path: PUT/POST file to createUrl
    print("new upload to", create)
    cmd = [
        "curl","-sS","-o","/tmp/kg_up.bin","-w","%{http_code}",
        "-X","PUT","-H","Content-Type: application/octet-stream",
        "--data-binary",f"@{FILE}",
        create,
    ]
    code = subprocess.check_output(cmd).decode().strip()
    print("upload http", code, open("/tmp/kg_up.bin","rb").read()[:200])
    upload_token = token

print("submit with blob token", upload_token)
cmd = [
    "curl","-sS","-o","/tmp/kg_sub.json","-w","%{http_code}",
    "-X","POST","-H",f"Authorization: Bearer {TOKEN}",
    "-H","Content-Type: application/json",
    "-d",json.dumps({"blobFileTokens":[upload_token] if not isinstance(upload_token,list) else upload_token,
                     "submissionDescription": MSG}),
    f"https://www.kaggle.com/api/v1/competitions/submissions/submit/{COMP}",
]
# Also try form-encoded variant used by older client
code = subprocess.check_output(cmd).decode().strip()
print("submit json http", code, open("/tmp/kg_sub.json").read()[:500])

cmd2 = [
    "curl","-sS","-o","/tmp/kg_sub2.json","-w","%{http_code}",
    "-X","POST","-H",f"Authorization: Bearer {TOKEN}",
    "-F",f"blobFileTokens={upload_token}",
    "-F",f"submissionDescription={MSG}",
    f"https://www.kaggle.com/api/v1/competitions/submissions/submit/{COMP}",
]
code2 = subprocess.check_output(cmd2).decode().strip()
print("submit form http", code2, open("/tmp/kg_sub2.json").read()[:500])

time.sleep(15)
cmd3 = ["curl","-sS","-H",f"Authorization: Bearer {TOKEN}",
        f"https://www.kaggle.com/api/v1/competitions/submissions/list/{COMP}"]
print("list:", subprocess.check_output(cmd3).decode()[:800])
PY