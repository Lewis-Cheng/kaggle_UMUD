# External data declaration (UMUD / RL track)

**Date (Asia/Hong_Kong):** 2026-11-05 01:32 HKT  
**Rules basis:** Kaggle UMUD Challenge Rules §2.6 External Data and Tools — public / free for all participants (or reasonably accessible); declare use; **do not hand-label** competition test/validation.  
**Does not claim Kaggle LB.** Attention training remains **blocked**.

---

## Declared datasets

| Name | URL | License | Local path | Role | Leakage notes |
|---|---|---|---|---|---|
| **UMUD MultExpData — Expert Analysed Benchmark architecture v0.1.0** | OSF project [xbawc](https://osf.io/xbawc) · file download `https://osf.io/download/hb5xj/` · doi:10.17605/OSF.IO/XBAWC | **CC-BY-4.0** | `data/external/umud_expert_arch_v0.1.0/` (unzipped `benchmark_dataset_architecture_v0.1.0/`; consensus `expert_consensus.csv`) | **RL reward / calibration GT** — multi-expert PA / FL / MT + `Scale_pixel_per_cm`. Metric label: **`S_expert_NOT_LB`**. Not used as competition submission labels. | Organizer-endorsed held-out expert set; **not** related to Kaggle train / DL_Track overlap (per organizer clarifications). Safe primary reward GT. |

### MultExpData details
- **n images:** 35 static B-mode (`im_01_arch.tif` … `im_35_arch.tif`)
- **Muscles:** GM, SOL, VL  
- **Devices:** Hitachi Aloka Alpha-10, Telemed Echo Blaster 128, Philips HD11, Telemed ArtUs EXT-1H  
- **Expert protocol (README):** 3 MT lines, 3 fascicles, 3 PA angles → mean per rater; sheet columns **R1–R7** (some images missing R5 → n_raters 6 or 7)
- **Consensus used here:** per-image **median(R1…R7)** for MT, FL, PA (nan-aware), plus mean / IQR / n_raters and SMA / DLTrack columns for non-RL baselines
- **Parse script:** `scripts/external/parse_multexpdata.py`

### Explicit non-uses (this declare)
- No hand labels on UMUD competition `test/` or private val.
- FALLMUD (CC BY-NC) **not** declared for prize submission.
- DL_Track LabTrain masks **not** used as clean held-out RL reward without hash/dedupe (leakage risk vs Kaggle train).
- ExpDrawnData / dynamic MultExpData zips: **not yet downloaded**; will amend this file if added.

### One-liner (for Kaggle discussion / code README)
> External data: UMUD Expert Analysed Benchmark architecture v0.1.0 (OSF xbawc / download hb5xj, CC-BY-4.0, 35 multi-expert PA/FL/MT) for calibration and RL reward (`S_expert_NOT_LB`). No hand labels on competition test.

