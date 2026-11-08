#!/usr/bin/env bash
# Recreate lab virtualenvs from lockfiles (Linux x86_64 / Python 3.10+).
# Literal .venv / .venv_tf folders are NOT in git (OS/arch/CUDA lock-in, ~4GB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

py="${PYTHON:-python3}"
echo "Using: $($py -V) on $(uname -srm)"

if [[ ! -d .venv ]]; then
  $py -m venv .venv
fi
.venv/bin/pip install -U pip wheel
if [[ -f requirements-venv.lock.txt ]]; then
  .venv/bin/pip install -r requirements-venv.lock.txt
else
  .venv/bin/pip install -r requirements.txt
fi

if [[ ! -d .venv_tf ]]; then
  $py -m venv .venv_tf
fi
.venv_tf/bin/pip install -U pip wheel
if [[ -f requirements-venv_tf.lock.txt ]]; then
  .venv_tf/bin/pip install -r requirements-venv_tf.lock.txt
fi
echo "Done. Activate with: source .venv/bin/activate"
