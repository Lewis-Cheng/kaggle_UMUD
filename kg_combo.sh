#!/bin/bash
# restore lewischeng json from bak if present; try combo auth that previously listed
export PATH="$HOME/.local/bin:$PATH"
if [ -f ~/.kaggle/kaggle.json.bak_lewischeng ]; then
  cp ~/.kaggle/kaggle.json.bak_lewischeng ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token | tr -d "\n\r ")
echo "json=$(head -c 40 ~/.kaggle/kaggle.json)"
echo "token_prefix=$(echo $KAGGLE_API_TOKEN | head -c 8)"
kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>&1 | head -8
# try pip install newest kaggle into user
pip install -U --user kaggle==1.6.17 -q
python3 -m kaggle --version 2>&1 | head
KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN python3 -m kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>&1 | head -8