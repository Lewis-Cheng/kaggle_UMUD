#!/bin/bash
set -e
cd ~/kaggle_UMUD
OUT=reports/phase3/rl_track_b
TOKEN=$(tr -d "\n\r " < ~/.kaggle/access_token)
# Parse lewischeng json without printing secrets
USER=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json.bak_lewischeng'))['username'])")
KEY=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json.bak_lewischeng'))['key'])")
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
FILE=$OUT/submission_track_b_v5_rl.csv
MSG=trackB_v5_rl_hybrid

echo "try1: Bearer access_token"
code=$(curl -sS -o /tmp/kg1.json -w "%{http_code}" \
  -X POST "https://www.kaggle.com/api/v1/competitions/submissions/submit/$COMP" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@$FILE" \
  -F "submissionDescription=$MSG" \
  -F "competitionName=$COMP")
echo "HTTP $code body=$(head -c 300 /tmp/kg1.json)"; echo

echo "try2: basic lewischeng:key"
code=$(curl -sS -o /tmp/kg2.json -w "%{http_code}" \
  -u "$USER:$KEY" \
  -X POST "https://www.kaggle.com/api/v1/competitions/submissions/submit/$COMP" \
  -F "submissionfile=@$FILE;filename=submission_track_b_v5_rl.csv" \
  -F "message=$MSG")
echo "HTTP $code body=$(head -c 400 /tmp/kg2.json)"; echo

echo "try3: api.kaggle.com Bearer"
code=$(curl -sS -o /tmp/kg3.json -w "%{http_code}" \
  -X POST "https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{}")
echo "HTTP $code body=$(head -c 300 /tmp/kg3.json)"; echo

echo "try4: list with Bearer"
code=$(curl -sS -o /tmp/kg4.json -w "%{http_code}" \
  -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/competitions/submissions/list/$COMP")
echo "HTTP $code body=$(head -c 400 /tmp/kg4.json)"; echo

echo "try5: list basic"
code=$(curl -sS -o /tmp/kg5.json -w "%{http_code}" \
  -u "$USER:$KEY" \
  "https://www.kaggle.com/api/v1/competitions/submissions/list/$COMP")
echo "HTTP $code body=$(head -c 400 /tmp/kg5.json)"; echo