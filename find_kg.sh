#!/bin/bash
ls -la ~/.kaggle/ 2>/dev/null
ls -la ~/kaggle_UMUD/.kaggle 2>/dev/null
find /home/chocheng9 -maxdepth 3 -name 'kaggle.json' 2>/dev/null
# earlier we saw access_token
ls -la ~/.kaggle/access_token 2>/dev/null
head -c 40 ~/.kaggle/access_token 2>/dev/null; echo
env | grep -i kaggle || true
# check if kg or another CLI
which kg; type kaggle