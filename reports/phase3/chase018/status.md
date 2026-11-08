# MultExp crop/scale + tiny-rot Frundles FT — 2026-11-06 22:27:25 UTC+08:00

## Result
| metric | value |
|---|---|
| **honest champ S** | **0.15111** (raw_crop_aug) |
| MAE PA / FL / MT | 1.027° / 2.754 mm / 0.158 mm |
| raw frozen S | 0.15111 (PA 1.027 / FL 2.754 / MT 0.158) |
| nested sin_w S | 0.15196 |
| Frundles map ref S | 0.13294 |
| vs honest hold 0.13449 | MISS (+0.01662) |
| gate 0.12 | FAIL |
| promote | False |
| domain-shift vs Frundles maps | LOSES (+0.01817) |
| train mix | {'labtrain': 1841, 'expdrawn': 199} |
| dropped near MultExp | {'labtrain': 13, 'expdrawn': 0} |
| best val Dice | 0.48484038871570223 |
| init | Frundles fasc_resnet34_384x576_f0.pt |
| aug | crop/scale [0.6, 1.0] + rot ±8° p=0.5 (same affine on image+mask) |
| epochs / bs / lr / device | 25 / 6 / 0.0001 / cuda:2 |
| elapsed | 16.0 min |
| no Kaggle | true |

## Protocol
MultExp dHash-held-out. Frozen recipe `hough_tex0.4_thr035_apo60_ff0.9_sin0.35_agNone`. Nested sin_w over [0.25, 0.28, 0.3, 0.35, 0.4] is honest LOOCV (pick on 34, score 1). Blend alphas are diagnostic only and not used for promotion. Comparator is honest joint S=0.13449, not the leaky 0.13195.

## Artifacts
- ckpt: `/home/chocheng9/kaggle_UMUD/reports/phase3/chase018/seg_ckpts/fasc_resnet34_384x576_crop_scale_aug_best.pt`
- maps: `/home/chocheng9/kaggle_UMUD/reports/phase3/chase018/cache/maps_multexp_crop_scale_aug`
- json: `/home/chocheng9/kaggle_UMUD/reports/phase3/chase018/multexp_crop_scale_aug.json`

---

﻿# chase018 next-day submit check — 2026-11-06 09:05 Asia/Hong_Kong

## Kaggle quota / submits
- Competition: umud-challenge-muscle-architecture-in-ultrasound-data (team data work `ch`)
- Today UTC 2026-11-06: **0 submits so far** (daily spare available)
- Best public still **texture_rl 0.29895** (ref 58005465, 2026-11-05)
- Prior: texture_pa 0.31903 (58004990); geo_floor 0.33641 (58003985)

## Gate decision this run
- Prefer `submission_texture_rl.csv` if MultExp LOOCV S < texture ch_only: **gate_pass=true** (rl_loocv_S=0.15743 < ch_only 0.17178)
- But CSV **already submitted** (identical to cal_pa_mt); **no new gated CSV** overnight
- Scratch/fasc_ft MultExp texture S (0.181 / 0.175) **worse** than ch_only — do not prefer over texture_rl
- Honest MultExp nested best **0.13449** (FR joint); gate **S<0.12 FAIL** — overnight fasc/ExpDrawn/fold-pure tracks stay **no Kaggle**

## Action
- **No submit** (nothing new; leave at least 1 spare unused)
- Stay quiet to Lewis per routine

## Next lever
MultExp-domain fasc with crop+scale and tiny-rot aug FT on LabTrain/ExpDrawn, MultExp held out. Frundles+A selector grid exhausted. Do not paste Track B GH-ET onto chase018; do not reopen MultExp FL selectors.

---

# MultExp joint fold-pure — 2026-11-05 17:37:37 Asia/Hong_Kong

## Honest nested (promote criterion)
| track | S | PA | FL | MT |
|---|---|---|---|---|
| nested baseline-path (Frundles×sin×PA) | 0.13751 | 0.935 | 2.447 | 0.158 |
| nested all-fixed | 0.13866 | 0.932 | 2.495 | 0.158 |
| FR joint (variant×sin×PA nested) | 0.13449 | 0.921 | 2.367 | 0.158 |
| C honest FR sin0.35 a0 | 0.14418 | 1.018 | 2.522 | 0.158 |
| **best honest** | **0.13449** | | | fr_joint |
| hold C (reported) | 0.13195144580335014 | leaky outer rule-family pick in adapt_multexp_fasc | | |
| gate 0.12 | FAIL | | | |
| promote | False | | | |
| no Kaggle | true | | | |

## Post-hoc (NOT promotable; selection on outer S)
| track | S | notes |
|---|---|---|
| fixed grid best | 0.13172 | corridor_dilate fl=0.3 a=0.15 |
| FR grid best (sin/alpha picked outer) | 0.13300 | sin=0.3 a=0.15 |

## Diagnosis
Honest joint nesting of fasc-adapt × fl_sin_w × A2 PA cannot beat reported hold 0.13195 (best honest **0.13449**). Reported hold C itself used leaky outer selection of feature_rule family (honest reproduction **0.14418**). Post-hoc corridor_dilate+sin0.30+a0.15 hits **0.13172** but is not fold-pure selectable (nested_all → 0.13866).

## Next
Stop this grid. Non-search lever: MultExp-domain fasc with crop+scale (+tiny rot) aug FT on LabTrain/ExpDrawn, MultExp held out — Frundles+A selector search is exhausted.

---

# MultExp joint fold-pure — 2026-11-05 17:37:37 Asia/Hong_Kong

## Result
| track | S | notes |
|---|---|---|
| C sanity (FR sin0.35 a0) | 0.14418 | expect ~0.13195 |
| fixed grid best | 0.13172 | corridor_dilate fl=0.3 a=0.15 |
| nested baseline-path | 0.13751 | trainS pick among Frundles×sin×PA |
| nested all-fixed | 0.13866 | trainS pick among all variants×sin×PA |
| FR grid best | 0.13300 | sin=0.3 a=0.15 |
| FR joint (sin×PA nested) | 0.13449 | full joint feature_rule |
| **champ** | **0.13172** | fixed_best |
| hold C | 0.13195144580335014 | |
| gate 0.12 | FAIL | |
| promote | True | |
| no Kaggle | true | |

## MAE (champ)
{'pa_deg': 0.9033125123143603, 'fl_mm': 2.302359431977445, 'mt_mm': 0.15826250681487788}

## Next
PROMOTABLE under hold — still no Kaggle until gate 0.12.

Elapsed_min: 2.7

---

# MultExp apo-rel PA + FL geo — 2026-11-05 17:24:33 Asia/Hong_Kong

## Result (honest; MultExp held out of train; no Kaggle)
| track | S | PA° | FL mm | MT mm |
|---|---|---|---|---|
| A soft-PA FT (ExpDrawn→Frundles) | 0.14388 | 1.020 | 2.507 | 0.158 |
| A2 PA-head blend0.25 | 0.13291 | **0.901** | 2.349 | 0.158 |
| B FL `sin_w0.30` (Frundles maps) | 0.13241 | 0.928 | **2.278** | 0.158 |
| **combo A2+B** | **0.13240** | 0.901 | 2.331 | 0.158 |
| holdcombo fasc-adapt+sin030 (nested) | 0.13241 | 0.928 | 2.278 | 0.158 |
| Frundles floor (thr035/tex0.4/sin0.35) | 0.13294 | 0.928 | 2.297 | 0.158 |
| **hold C** | **0.13195** | 0.912 | 2.292 | 0.158 |
| gate 0.12 | FAIL | | | |
| promote vs hold | **False** (gap +0.00045) | | | |

## Notes
- A: 152 ExpDrawn apo-rel PA targets (stroke vs deep-apo); BCE+Dice+soft_pa FT 12ep valDice~0.28; MultExp maps worse.
- A2: light PA head on ExpDrawn targets; **PA improves** 0.928→0.901 but FL worsens via sin path.
- B: tip/deep/cond/sin/ff micro on Frundles maps; best fixed `sin_w0.30` → 0.13241.
- Honest nested fasc-adapt feature_rule does **not** beat fixed baseline (leaky outer-rule search that hit ~0.128–0.129 was discarded).
- Artifacts: `multexp_aporelpa_flgeo_summary.json`, `_A.json`, `_A2.json`, `_B.json`, `_holdcombo.json`, `_targets.json`, ckpt `fasc_resnet34_384x576_aporelpa_best.pt`

## Next
Missed hold by ~0.00045. Next ≤3: (1) fold-pure joint search over fasc-adapt × sin_w∈[0.28,0.35] × A2 PA blend with train-only selection, (2) deep-apo geodesic FL on Frundles fasc, (3) MultExp-domain fasc GT proxy beyond ExpDrawn overlays.

---
# MultExp ExpDrawn→Frundles FT (consolidated) — 2026-11-05 16:58:38 UTC+08:00

## Result
| metric | value |
|---|---|
| **champ S** | **0.13903** (expdrawn_only_blend) |
| MAE PA / FL / MT | 1.002° / 2.368 mm / 0.158 mm |
| thr | 0.3 |
| vs hold 0.13195 | MISS (+0.00708) |
| gate 0.12 | FAIL |
| promote | False |
| usable ExpDrawn pairs | 199 / 249 (50 fail corr<0.97: {'shape ann': 7, 'corr_below_0.97': 43}) |
| ExpDrawn-only | valDice=0.2904 rawS=0.25391 blendα=0.15 S=0.13903478572889152 |
| ExpDrawn+LabTrain309 | valDice=0.3790 rawS=0.14370 blendα=0.25 S=0.14044712448014207 |
| Frundles ref S | 0.13294 (PA 0.928 / FL 2.297 / MT 0.158) |
| loss | 0.55*BCE(pos_w=4)+0.40*Dice+0.05*orient; Frundles init; MultExp held out |
| no Kaggle | true |

## Diagnosis
ExpDrawn red-stroke overlays usable (199). Frundles-init FT preserved val Dice (0.29–0.38, not collapsed). Mask-domain ExpDrawn (±LabTrain) still worse than hold C 0.13195 / Frundles thr0.35 floor 0.13294.

## Next
ExpDrawn overlay FT exhausted vs hold. Next ≤3: (1) apo-relative soft PA on Frundles init using ExpDrawn stroke angles as weak geo labels, (2) keep Frundles fasc maps and attack FL geometry (insertion/sin) outside mask FT, (3) seek MultExp-safe public fasc with VL-domain match beyond ExpDrawn panels.

Artifacts: multexp_expdrawn_summary.json, multexp_expdrawn_frundles_only.json, multexp_expdrawn_frundles_mix309.json, multexp_expdrawn_extract.json

---

# MultExp ExpDrawn→Frundles FT — 2026-11-05 16:56:48 UTC+08:00

## Result
| metric | value |
|---|---|
| **champ S** | **0.13903** (blend_a0.15) |
| MAE PA / FL / MT | 1.002° / 2.368 mm / 0.158 mm |
| thr | 0.3 |
| vs hold 0.13195 | MISS (+0.00708) |
| gate 0.12 | FAIL |
| promote | False |
| usable ExpDrawn pairs | 199 (extract ok=199) |
| train mix | {'expdrawn': 199} |
| init | Frundles fasc |
| best val Dice | 0.2904 |
| Frundles ref S | 0.13294 |
| raw ExpDrawn-FT S | 0.25391 |
| best blend | alpha=0.15 S=0.13903478572889152 |
| elapsed | 2.3 min |
| no Kaggle | true |

## Diagnosis
ExpDrawn red-stroke overlays (ann[:,200:1000]↔raw) → fasc masks; Frundles-init BCE+Dice+light orient FT; MultExp held out.

## Next
ExpDrawn FT did not beat hold — next: (1) apo-relative soft PA on Frundles init with ExpDrawn stroke angles, (2) thicker multi-dilate ensemble, (3) keep Frundles maps + seek better FL geometry outside mask FT.

Artifacts: `multexp_expdrawn_frundles.json`, extract `multexp_expdrawn_extract.json`, cache `maps_multexp_expdrawn_only`, ckpt `fasc_resnet34_384x576_expdrawn_frundles_best.pt`

---

# MultExp ExpDrawn→Frundles FT — 2026-11-05 16:50:41 UTC+08:00

## Result
| metric | value |
|---|---|
| **champ S** | **0.14045** (blend_a0.25) |
| MAE PA / FL / MT | 0.992° / 2.440 mm / 0.158 mm |
| thr | 0.3 |
| vs hold 0.13195 | MISS (+0.00850) |
| gate 0.12 | FAIL |
| promote | False |
| usable ExpDrawn pairs | 199 (extract ok=199) |
| train mix | {'expdrawn': 199, 'labtrain': 296} |
| init | Frundles fasc |
| best val Dice | 0.3790 |
| Frundles ref S | 0.13294 |
| raw ExpDrawn-FT S | 0.14370 |
| best blend | alpha=0.25 S=0.14044712448014207 |
| elapsed | 3.2 min |
| no Kaggle | true |

## Diagnosis
ExpDrawn red-stroke overlays (ann[:,200:1000]↔raw) → fasc masks; Frundles-init BCE+Dice+light orient FT; MultExp held out.

## Next
ExpDrawn FT did not beat hold — next: (1) apo-relative soft PA on Frundles init with ExpDrawn stroke angles, (2) thicker multi-dilate ensemble, (3) keep Frundles maps + seek better FL geometry outside mask FT.

Artifacts: `multexp_expdrawn_frundles.json`, extract `multexp_expdrawn_extract.json`, cache `maps_multexp_expdrawn`, ckpt `fasc_resnet34_384x576_expdrawn_frundles_best.pt`

---

# MultExp numeric-geo LabTrain — 2026-11-05 16:38:32 UTC+08:00

## Result
| metric | value |
|---|---|
| **champ S** | **0.13662** (blend_a0.25) |
| MAE PA/FL/MT | 0.952 / 2.381 / 0.158 |
| vs hold 0.13195 | MISS (+0.00467) |
| gate 0.12 | FAIL |
| numeric geo pairs | 309 |
| val Dice | 0.020039680569122236 |
| Frundles ref | 0.13294 |
| ExpDrawn | FOUND (40Y+85O subjects; Young+Old OSF zips) |
| no Kaggle | true |

---
# MultExp geoloss LabTrain+comp — 2026-11-05 16:12:29 UTC+08:00

## Result
| metric | value |
|---|---|
| **champ S** | **0.13777** (blend_a0.40) |
| MAE PA / FL / MT | 1.011° / 2.304 mm / 0.158 mm |
| thr | 0.25 |
| vs hold 0.13195 | MISS (+0.00582) |
| gate 0.12 | FAIL |
| promote | False |
| init | frundles_state encoder=resnet34 size=(384, 576) |
| train mix | {'labtrain': 1841, 'comp': 2750} (ExpDrawn missing=True) |
| loss | BCE 0.35 + Dice 0.35 + clDice 0.2 + orient 0.1 |
| best val Dice | 0.3407 |
| Frundles ref S | 0.13294 |
| raw geoloss S | 0.14448 |
| best blend | alpha=0.4 S=0.1377727313565292 |
| elapsed | 10.6 min |
| no Kaggle | true |

## Diagnosis
Geometry-aware multi-dataset FT (MultExp held out). ExpDrawn skipped (not local; OSF Benchmark Models 404).

## Next
Geo loss did not beat hold — next: ExpDrawn overlays if URL found, or MultExp-safe PA/FL numeric proxy from paired apo belly on LabTrain apo+fasc co-located subsets.

Artifacts: `multexp_geoloss_labtrain_comp.json`, cache `maps_multexp_geoloss`, ckpt `fasc_resnet34_384x576_geoloss_labtrain_comp_best.pt`

---

# MultExp chase018 status
Updated: 2026-11-05 14:54:15 UTC+08:00

## Outside-family fasc (this turn)
- Hold S: **0.13195**
- Best outside-family: **0.52740** (classical_frangi, thr=0.35) PA=1.497 FL=15.359 MT=0.158
- Frundles baseline frozen: **0.13294**
- Gate 0.12: **FAIL**; beats hold: **False**; no Kaggle
- Inventory: public non-Frundles weights on disk=NO; MultExp fasc pixel GT=NO; phase3 lab fasc ckpts exist (Dice~0.28)
- Artifacts: `multexp_nonfrundles_fasc_classical.json`
- Next: Classical + blend did not beat 0.13195. Next: (1) dump phase3 MSTinyUNet v5/v12 MultExp maps despite low Dice as true outside-family DL, (2) apo-bounded multi-orientation Frangi with dark-ridge bias + fasc skeleton thinning as fasc channel, (3) seek documented Mawra/open fasc weight URL in refs before any new download.

---

# chase018 status — 2026-11-05 09:43 HKT (Asia/Hong_Kong)

## Public scores
| submission | ref | publicScore | notes |
|---|---|---|---|
| **texture_pa** | **58004990** | **0.31903** | Frundles apo+fasc + texture tex0.5/fl_sin0.25/fl×0.9 |
| chase018_geo_floor | 58003985 | 0.33641 | prior floor |
| trackB_v5_rl_hybrid | 58003328 | 1.95264 | old Track B (do NOT reuse blend_gh50_et50 on chase018) |

**No Kaggle submit this session** (last daily spare held).

## Fasc scratch train FINISHED
- best val_hard_dice **0.4136247794220458** @ ep28 (30/30 done, 958s)
- ckpt mtime: 2026-11-05 09:25:44 HKT
- path: `reports/phase3/chase018/seg_ckpts/fasc_resnet34_384x576_scratch_best.pt`
- log: `reports/phase3/chase018/fasc_scratch_train.log`

## MultExp S_expert_NOT_LB (n=35)

### Frundles apo+fasc (texture base)
| variant | S_expert | MAE PA° | FL mm | MT mm |
|---|---|---|---|---|
| floor | 0.19344 | 1.731 | 2.704 | 0.199 |
| **texture tex0.5+fl_sin0.25** | **0.17178** | 1.410 | 2.566 | 0.199 |
| cal_pa_mt (PA+MT affine; FL from ch) | **0.15743** | 1.236 | 2.566 | 0.158 |
| **RL LOOCV SoftmaxBandit** | **0.15743** GATE PASS | 1.236 | 2.566 | 0.158 |

Gate: loocv_S (0.15743) < ch_only/texture_pa (0.17178) → **PASS**. Policy always picks `cal_pa_mt` (35/35 LOOCV, 309/309 test).

### Scratch fasc + Frundles apo + texture
| variant | S_expert | notes |
|---|---|---|
| texture tex0.5+fl_sin0.25 | 0.18112 | worse than Frundles fasc |
| residual_or_scratch / blend_ch_scratch50 | 0.17445 | still worse than ch_only |

**Takeaway:** MultExp prefers Frundles fasc; scratch fasc is LB-domain ablation only.

### RL action baselines (full-set, not LOOCV)
v5/ET blends destroy MultExp (S≈0.55–1.04). fl_factor 0.85/0.95 hurt. Only cal_pa_mt beats ch_only.

## New submit-ready CSVs (309 rows, comma, no BOM, exact texture_pa basenames)
1. **`submission_texture_rl.csv`** (= cal_pa_mt; MultExp champ) ← **first submit next quota**
2. `submission_cal_pa_mt.csv` (identical numbers; explicit affine path)
3. `submission_scratch_fasc_texture.csv` (secondary LB ablation; MultExp 0.181)
4. `submission_texture_pa.csv` (already submitted, public 0.31903)

## RL+v5 merge (chase018 SoftmaxBandit)
- `src/umud/rl/chase018_bandit.py` — new 8-action env; Track B untouched
- `scripts/chase018/{build_multexp_chase_rows,train_rl_refiner,infer_rl_refiner}.py`
- artifacts: `reports/phase3/chase018/rl_refiner/`, `multexp_chase_rows.csv`, `multexp_rl_boost_scores.json`, `multexp_chase_affine.json`
- v5 sweet point kept: frozen post-seg refine, MultExp reward, heavy L2 SoftmaxBandit, LOOCV gate

## Next-quota submit plan (leave ≥1 spare)
1. **`submission_texture_rl.csv`** (or cal_pa_mt) — MultExp-ranked unsubmitted champ
2. Optional: `submission_scratch_fasc_texture.csv` (competition-domain fasc hypothesis)
3. Hold ≥1 spare

## Key paths (lab `~/kaggle_UMUD` = source of truth; box mirror synced)
- reports/phase3/chase018/submission_texture_rl.csv
- reports/phase3/chase018/submission_cal_pa_mt.csv
- reports/phase3/chase018/submission_scratch_fasc_texture.csv
- reports/phase3/chase018/multexp_rl_boost_scores.json
- reports/phase3/chase018/multexp_scratch_fasc_scores.json
- reports/phase3/chase018/seg_ckpts/fasc_resnet34_384x576_scratch_best.pt


---

# MultExp FL post-process status — 2026-11-05 13:02 Asia/Hong_Kong

## Result (Frundles maps only; no lab fasc ckpt; no Kaggle)
| metric | value |
|---|---|
| **best S_loocv** | **0.13294** |
| recipe | hough_tex0.4_thr035_apo60_ff0.9_sin0.35_agNone |
| MAE PA / FL / MT | 0.928° / 2.297 mm / 0.158 mm |
| vs prior floor 0.13761 | **BEAT** (-0.00467) |
| vs prior selector 0.13550 | **BEAT** (-0.00256) |
| gate S&lt;0.12 | **FAIL** (gap +0.01294) |

### Kw
{"fasc_thr": 0.35, "apo_thr": 0.6, "tex_w": 0.4, "fl_factor": 0.9, "fl_sin_w": 0.35, "agree_thr": null, "pa_mode": "hough"}

## Ladder
| step | S |
|---|---|
| Re-inferred promoted hough_tex03_thr032_apo60 | 0.13851 |
| thr/apo/tex grid best (thr035_apo60 tex0.3) | 0.13422 |
| **FL sin_w/agree post on tex0.4 thr035** | **0.13294** |
| FL-source oracle (ffsweep cands) | 0.08612 |
| Transferable FL selector (train-best / ols5) | 0.13744 / 0.14100 |

## Tried (FL post-process)
- Endpoint protocols (centre/deep_insert/shallow/both_inner), fl_factor grid, sin blends, ill-conditioned reject→fallback
- Fold-pure Softmax/L1 residual / kNN / simplex / nested cal-PA blends — none transferable below ~0.137
- Oracle proves headroom (~0.08–0.09) but sources are image-idiosyncratic (no stable selector)

## Ceiling
Frundles-map FL post-process + thr/tex micro-search ceiling **~0.133**. Gate 0.12 not reachable without better MultExp-domain fasc geometry.

## Next lever (non-FL-post)
**MultExp-domain fasc adaptation** (do not swap lab scratch/FT MultExp maps — already failed 0.14–0.15). Adapt fasc to MultExp appearance while keeping Frundles apo + current thr035/tex0.4/sin0.35 recipe.

## Artifacts
- cache/maps_multexp (Frundles re-infer)
- multexp_fl_postprocess_ffsweep.json / _scores.json / _grid.json / _summary.json
- math_map_promoted_recipe.json (new best)
- multexp_fl_postprocess_best_rows.csv
- scripts: eval_multexp_fl_postprocess.py, eval_multexp_fl_ffsweep.py