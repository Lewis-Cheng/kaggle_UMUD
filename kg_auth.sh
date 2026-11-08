#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd ~/kaggle_UMUD
# Try new token auth
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
# Some versions want the token without KGAT_ prefix quirks - try as-is
kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>&1 | head -15
echo EXIT:$?
# also try kaggle auth
kaggle --version 2>&1
python3 -c "import kaggle; print(kaggle.__file__)" 2>&1 | head