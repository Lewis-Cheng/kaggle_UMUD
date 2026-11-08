# kaggle_UMUD

Lab working repo for the [UMUD Challenge](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data):
predict per test ultrasound image `pa_deg`, `fl_mm`, and `mt_mm`.

**Host:** `chocheng9@144.214.209.211` · path `/home/chocheng9/kaggle_UMUD` · **4× NVIDIA L40**  
**Primary status docs:** `reports/phase3/chase018/` (especially `status.md`, `multexp_joint_status.md`)

This README is the on-server entry point. Experiment detail lives under `reports/phase3/chase018/` — do not treat this file as a full experiment log.

---

## Current approach (chase018)

**Working stack (not the original attention sketch):**

1. **Frundles** public SMP ResNet34 U-Nets for **apo** + **fasc** probability maps (384×576).
2. **Classical / host-faithful geometry** on those maps (apo curves, Hough/texture PA, fascicle FL with `fl_sin_w` / factor, MT from apo separation).
3. **Calibration** on held-out expert MultExpData: LOOCV affine on **PA+MT only**; **FL raw**.
4. Optional MultExp-domain fasc **adaptation** / light PA heads — always with MultExp **held out of training**.

**MultExpData** (`data/external/umud_expert_arch_v0.1.0/`, n=35) is the honest offline gate (`S_expert_NOT_LB`). It must not leak into train folds used to pick recipes.

The early scaffold “anatomy-query attention” path is **blocked** until gates pass (see Constraints). Prefer 2D Frundles + geometry over 3D-first approaches.

---

## Latest verified status (2026-11-07 Asia/Hong_Kong)

Source of truth: `reports/phase3/chase018/multexp_joint_scores.json` + `multexp_joint_status.md` + Kaggle submit receipts below.

### Honest MultExp (offline gate)

| track | S | notes |
|---|---|---|
| **best honest nested** (`feature_rule_joint` / fold-pure) | **0.13449** | fasc-adapt × sin_w × PA-head, fold-pure |
| Frundles FL-post maps (baseline) | **~0.13294** | Frundles maps + FL post; selector-era floor |
| C honest FR sin0.35 α0 | 0.14418 | honest reproduction of “hold C” recipe |
| hold C **reported** | 0.13195 | **NOT a valid hold** — leaky outer rule-family pick |
| post-hoc fixed grid best | ~0.13172 | **NOT promotable** (selected on outer S) |
| Crop/scale + tiny-rot Frundles fasc FT | **0.15111** | MultExp held out; val Dice ↑ (~0.485) but MultExp PA/FL worse vs Frundles maps |
| gate S < 0.12 | **FAIL** | no promote under gate |

Promote criterion is **honest nested / fold-pure** only. Selector grids on Frundles×adapt×sin×PA are exhausted. LabTrain/ExpDrawn Dice FT with crop/scale + tiny rot **did not** beat Frundles MultExp maps.

### Kaggle public (verified)

| submission | ref | publicScore | notes |
|---|---|---|---|
| **texture_rl** | **58005465** | **0.29895** | **best public so far** |
| texture_pa | 58004990 | 0.31903 | Frundles apo+fasc + texture tex0.5 / fl_sin0.25 / fl×0.9 |
| joint_foldpure | 58058659 | 0.44238 | honest MultExp best CSV; did not transfer |
| aporel combo | 58062956 | 0.43577 | MultExp-lowest batch (~00:08–00:10 HKT) |
| FL sin_w0.30 | 58062965 | 0.43793 | same batch |
| PA-head A2 | 58062983 | 0.43615 | same batch |
| Frundles FL-post | 58062992 | 0.43694 | same batch |

**Finding:** MultExp-lowest geometry recipes (~honest S 0.13) land ~**0.43–0.44** on the hidden LB. **MultExp ranking alone is not a safe LB proxy.** Daily submit quota for that day was burned; none of the MultExp-lowest submits beat `texture_rl`.

---

## Score definition

Pinned formulas in `src/umud/metric/score.py` (organizer notebook `paulritsche/umud-score`):

```text
S ≈ S1 = mean( MAE_PA / 6 , MAE_FL / 12 , MAE_MT / 3 )
S  = S1 + 1e-6·S2 + 1e-9·S3   # S2 median-AE, S3 RMSE (same tolerances)
```

Offline MultExp label is **`S_expert_NOT_LB`** (same formula vs expert consensus). Gate for promote/submit experiments here: **honest S < 0.12** (not met).

---

## Environment / paths

```bash
cd /home/chocheng9/kaggle_UMUD
export PYTHONPATH=src

# Main PyTorch stack (training / Frundles SMP / geometry eval)
.venv/bin/python            # torch 2.7.x + CUDA; sees 4× L40

# TensorFlow env (DL_Track / TF-only helpers)
.venv_tf/bin/python         # tensorflow 2.17.x
```

| Path | Role |
|---|---|
| `data/raw/umud/` | Competition apo/fasc imgs+masks, `test_images_v2/`, `sample_submission.csv` |
| `data/external/frundles_weights/` | Public Frundles apo+fasc ckpts |
| `data/external/umud_expert_arch_v0.1.0/` | MultExp expert images + `expert_consensus.csv` |
| `reports/phase3/chase018/` | Status, scores JSON, logs, map caches, MultExp submissions |
| `reports/phase3/chase018/cache/maps_multexp/` | Cached MultExp apo/fasc `.npz` maps |
| `reports/phase3/chase018/seg_ckpts/` | Local fasc FT / scratch checkpoints |
| `configs/protocol.yaml` | Frozen scorer pin + audit summary |
| `src/umud/` | Metric, geometry, seg loaders, calibration |

Minimal deps: `requirements.txt`. Training also needs `segmentation_models_pytorch`, OpenCV, CUDA torch (already in `.venv` on this host).

---

## Layout (scaffold + chase)

```text
configs/protocol.yaml
src/umud/
  metric/          official S + submission validator
  geometry/        host-faithful + texture angle helpers
  calibration/     anisotropic scale
  seg/             Frundles SMP load/predict
  models/          early anatomy-query stubs (blocked path)
scripts/
  chase018/        MultExp map infer, eval, adapt, fasc train, joint search
  audits/          phase2 data gates
  pull_scorer.py / validate_submission.py / score_predictions.py
reports/phase3/chase018/   living experiment status (read these)
tests/             contract fixtures (no real images required)
```

---

## Main chase018 scripts

All under `scripts/chase018/`. Use `.venv/bin/python` unless noted. **No Kaggle submit from these scripts.**

### `infer_multexp_maps.py`
Dump MultExp apo+fasc probability maps (`float16` `.npz`, 384×576) for offline geometry. Frundles apo always; fasc from public weights or `--fasc-ckpt`.

```bash
.venv/bin/python scripts/chase018/infer_multexp_maps.py \
  --cache-dir reports/phase3/chase018/cache/maps_multexp \
  --device cuda:0
# optional: --fasc-ckpt reports/phase3/chase018/seg_ckpts/<fasc>.pt --force
```

### `eval_promoted_hough_multexp.py`
Honest MultExp LOOCV over promoted Hough/texture geometry variants on cached maps. LOOCV affine on PA+MT; FL raw. Writes score JSON under `reports/phase3/chase018/`.

```bash
.venv/bin/python scripts/chase018/eval_promoted_hough_multexp.py
# --promoted-only   # only floor variants
# --cache-dir ... --out-json ... --tag ...
```

### `adapt_multexp_fasc.py`
MultExp-domain fasc **map adaptation** (corridor / ridge / morph variants) on Frundles maps with a **frozen** FL/geo recipe. Fold-pure selection among adapt variants. Historical **leaky** outer-S rule-family pick produced the invalid hold **0.13195** — use honest nested numbers from joint status instead.

```bash
.venv/bin/python scripts/chase018/adapt_multexp_fasc.py \
  --cache-dir reports/phase3/chase018/cache/maps_multexp \
  --out-dir reports/phase3/chase018
```

Related: `adapt_multexp_fasc_v2.py`, `adapt_multexp_fasc_v2b.py`, `multexp_joint_foldpure.py` (joint fasc-adapt × sin_w × PA nested search).

### `train_fasc_frundles_recipe.py`
Train fasc ResNet34 U-Net from **ImageNet** with Frundles-style aug (aspect-jitter crop, ±10° rot already in recipe, gamma/brightness/speckle), BCE+Dice, OneCycleLR, dHash group split. Requires `--img-dir` / `--mask-dir`.

```bash
.venv/bin/python scripts/chase018/train_fasc_frundles_recipe.py \
  --img-dir data/raw/umud/fasc_imgs_v1/fasc_images_new_model_v1 \
  --mask-dir data/raw/umud/fasc_masks_v1/fasc_masks_new_model_v1 \
  --out-dir reports/phase3/chase018/seg_ckpts \
  --tag fasc_resnet34_384x576_scratch --device cuda:0
```

### `train_fasc_smp_ft.py`
Fine-tune **from Frundles fasc weights** on competition fasc masks (Dice+BCE+soft-clDice, dHash groups). Defaults point at `data/raw/umud/fasc_{imgs,masks}_v1` and `data/external/frundles_weights/fasc_resnet34_384x576_f0.pt`.

```bash
.venv/bin/python scripts/chase018/train_fasc_smp_ft.py \
  --device cuda:0 --epochs 40
```

Other notable scripts (see reports for when they were used): `eval_multexp_fl_postprocess.py`, `multexp_aporelpa_flgeo.py`, `train_fasc_expdrawn_frundles.py`, DL_Track helpers (`run_dltrack*.sh` → `.venv_tf`).

---

## Reproduction / evaluation workflow

Typical MultExp held-out loop:

1. **Infer maps** → `infer_multexp_maps.py` (Frundles or override fasc ckpt).
2. **Geometry + LOOCV score** → `eval_promoted_hough_multexp.py` / FL post-process / adapt / joint fold-pure scripts.
3. **Record** JSON + short status under `reports/phase3/chase018/`; update `status.md` / `multexp_joint_status.md`.
4. **Promote** only if honest nested S beats the current honest baseline **and** approaches gate **< 0.12**.
5. **Kaggle** only after a new recipe that is not just MultExp-lowest geometry (see Finding above). Validate CSV first:

```bash
export PYTHONPATH=src
.venv/bin/python scripts/validate_submission.py path/to/submission.csv \
  --expected-ids path/to/test_ids.txt
.venv/bin/python scripts/score_predictions.py path/to/pred.csv path/to/gt.csv
```

**Submission format that scores:** comma-separated CSV, no BOM; `image_id` = exact `test_images_v2` / `test_set_v2` basename (mix of `.tif` and `.png`; 251+58=309). Prefer `KAGGLE_API_TOKEN` (KGAT_) over legacy 32-char key. The sample_submission semicolon+BOM stub is misleading.

Contract tests (fixtures only — not an LB claim):

```bash
export PYTHONPATH=src
bash scripts/run_contract_tests.sh
```

Pull / pin official scorer (needs Kaggle token for live download):

```bash
.venv/bin/python scripts/pull_scorer.py
```

---

## Constraints (do not violate)

- **Honest** LOOCV / nested / group-safe (dHash) evaluation only — no oracle or outer-S recipe promotion.
- **No test hand-labeling**; no private-LB blending.
- MultExp held out of any train used to choose the evaluated recipe.
- **No attention training** until protocol gates say otherwise (`reports/phase3/attention_gate_verdict.md`).
- **No 3D-first** (Fast3R / DUSt3R / VGGT-style) as the primary plan.
- Gate **S < 0.12** is still the promote bar for MultExp experiments; MultExp-lowest alone is **not** enough to trust LB transfer.
- External data must stay declared in `reports/phase3/external_data_declare.md`.

---

## Next candidate

Do **not** run more LabTrain / ExpDrawn Dice fine-tunes expecting MultExp transfer (crop/scale + tiny-rot FT already failed at **0.15111**).

Next levers:

1. **Geometry on Frundles MultExp maps themselves** (or closer-domain maps) — FL/PA post, texture, scale — with MultExp held out of rule selection where possible.
2. Treat MultExp as a weak / complementary signal, not the sole LB proxy; prefer recipes that stay near the public floor (`texture_rl` **0.29895**) unless a clear domain fix appears.
3. Only burn new Kaggle quota on a **new** recipe class, not another MultExp-lowest selector CSV.

Point detailed logs and ablations to `reports/phase3/chase018/` rather than duplicating them here.

---

## Scaffold notes (preserved)

Original verification-first scaffold still applies for anything that touches the official metric:

1. Pin official scorer (`configs/protocol.yaml` SHA).
2. Audit overlap / scale / duplicate-aware splits (`scripts/audits/`).
3. Reproduce calibrated seg + classical geometry before claiming LB wins.
4. Only then ablate blocked attention modules under a frozen protocol.

Phase 2 audit runner:

```bash
export PYTHONPATH=src
bash scripts/audits/run_phase2.sh \
  --data-root ~/kaggle_UMUD/data/raw/umud \
  --out-dir ~/kaggle_UMUD/reports/phase2
```

Reports are evidence only — a successful script run ≠ Phase 2 passed.

---

## Open TODOs

- Work geometry / postprocess on Frundles MultExp maps (or closer domain) — **not** more LabTrain/ExpDrawn Dice FT.
- Keep Frundles apo maps unless a new fasc clearly beats Frundles floor on MultExp without leak **and** shows LB-relevant transfer.
- Do not resubmit MultExp-lowest geometry batches; wait for a new recipe before next Kaggle spend.
- Keep `reports/phase3/chase018/status.md` synced with any new public LB refs (submission ref + publicScore).
