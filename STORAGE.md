# Storage layout for kaggle_UMUD

The main repo holds code, configs, scripts, and reports metadata.

## Large artifacts (companion private repos)

| Content | Size (lab) | Repo |
|---------|------------|------|
| Checkpoints / weights (`*.pt`, `*.h5`, frundles, DLTrack) | ~8.2 GB LFS | https://github.com/Lewis-Cheng/kaggle_UMUD-weights |
| Full `data/` (raw + external) | ~16 GB | https://github.com/Lewis-Cheng/kaggle_UMUD-data |

```bash
git clone git@github.com:Lewis-Cheng/kaggle_UMUD.git
git clone git@github.com:Lewis-Cheng/kaggle_UMUD-weights.git
git clone git@github.com:Lewis-Cheng/kaggle_UMUD-data.git
# Merge trees as needed, e.g.:
#   rsync -a kaggle_UMUD-weights/reports/ kaggle_UMUD/reports/
#   rsync -a kaggle_UMUD-data/data/ kaggle_UMUD/data/
```

## Virtualenvs (not committed)

`.venv` (~1.9G) and `.venv_tf` (~2.2G) are Linux x86_64 / glibc lab builds.
Do not copy them to Windows/macOS. Recreate:

```bash
bash scripts/recreate_venvs.sh
```

Lockfiles: `requirements-venv.lock.txt`, `requirements-venv_tf.lock.txt`.

## Secrets (NEVER in git)

These live only on the lab home directory / local machines — not in any GitHub repo:

- `~/.kaggle/kaggle.json`
- `~/.kaggle/access_token`
- `~/.kaggle/kaggle.json.bak_lewischeng`
- `~/.kaggle/kaggle.json.disabled`

Repo scripts (`kg_auth.sh`, `curl_submit.sh`) read those paths at runtime.
Recommended: store credentials in a password manager / Cursor vault; keep `.gitignore` entries for `kaggle.json`, `.env`, `*.pem`, `secrets/`.

## Why not one mega-repo?

GitHub soft limits and LFS bandwidth make a single 28G repo painful to clone.
Companion private repos keep code clones fast while still hosting weights + data.
