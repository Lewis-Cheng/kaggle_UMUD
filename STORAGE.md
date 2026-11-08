# Storage layout for kaggle_UMUD

The main repo holds code, configs, scripts, and reports metadata.

## Large artifacts (companion private repos) — PUSHED

| Content | Size (lab) | Repo | Commit |
|---------|------------|------|--------|
| Checkpoints / weights (`*.pt`, `*.h5`, frundles, DLTrack) | ~8.2 GB LFS (56 objects) | https://github.com/Lewis-Cheng/kaggle_UMUD-weights | `13a01d9` |
| Full `data/` (raw + external) | ~15 GB LFS (16002 objects) | https://github.com/Lewis-Cheng/kaggle_UMUD-data | `3a524ec` |

```bash
git clone git@github.com:Lewis-Cheng/kaggle_UMUD.git
git clone git@github.com:Lewis-Cheng/kaggle_UMUD-weights.git
git clone git@github.com:Lewis-Cheng/kaggle_UMUD-data.git
# Merge trees as needed, e.g.:
#   rsync -a kaggle_UMUD-weights/reports/ kaggle_UMUD/reports/
#   rsync -a kaggle_UMUD-weights/data/ kaggle_UMUD/data/
#   rsync -a kaggle_UMUD-data/data/ kaggle_UMUD/data/
```

### Competition zip note (data repo)

GitHub LFS rejects single objects > 2 GiB. The original
`data/raw/umud-challenge-muscle-architecture-in-ultrasound-data.zip` (~2.6 GiB)
is stored as split parts under `data/raw/zip_parts/` with `reassemble.sh`.
The extracted tree at `data/raw/umud/` is already in the data repo.

## Virtualenvs (NOT committed as folders)

`.venv` (~1.9G) and `.venv_tf` (~2.2G) are Linux x86_64 / glibc lab builds
(Python 3.12 on the lab). Copying them to Windows/macOS will not work.
Recreate on a Linux x86_64 machine:

```bash
bash scripts/recreate_venvs.sh
```

Lockfiles in main repo: `requirements-venv.lock.txt`, `requirements-venv_tf.lock.txt`.

## Secrets (NEVER in git — lab home only)

These live only on the lab / local machines — not in any GitHub repo:

- `~/.kaggle/kaggle.json`
- `~/.kaggle/access_token`
- `~/.kaggle/kaggle.json.bak_lewischeng`
- `~/.kaggle/kaggle.json.disabled`

Repo scripts (`kg_auth.sh`, `curl_submit.sh`) read those paths at runtime.
Recommended: store credentials in a password manager / Cursor vault; keep
`.gitignore` entries for `kaggle.json`, `.env`, `*.pem`, `secrets/`.

## Why not one mega-repo?

GitHub soft limits and LFS object size (2 GiB max) make a single 28G repo
painful. Companion private repos keep code clones fast while hosting weights + data.
