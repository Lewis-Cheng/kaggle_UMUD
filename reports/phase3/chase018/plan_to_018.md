# Plan: 0.336 → ~0.18 (single-factor ablations)

Current floor **0.33641** matches Frundles open notebook. LB #1 is **0.17651** (closed). Best open claim Mawra **~0.306**. Path below is ordered by expected LB gain / risk; **one factor per submit**.

## Next ablations (priority order)
1. **PA near deep apo only (already partially via depth-fit)** — Mawra texture-angle / depth-bin PA; compare vs our weighted-median. Target: 0.336 → ~0.32.
2. **Submit `submission_geo_floor_cal_pa_mt.csv`** (MultExp PA+MT affine, FL untouched). Cheap; may help or hurt LB (domain shift). One try.
3. **Fascicle Dice ≥0.40** — Frundles/Mawra public fasc is stronger than our old v5 (0.28). Train SMP ResNet34 fasc with orientation-aware / skeleton loss on competition masks; keep apo weights. Expect biggest move toward 0.30→0.25 if fasc direction improves.
4. **Texture-angle PA (Mawra)** — structure-tensor / Radon angle inside fasc mask instead of SVD segment angle; public ~0.307.
5. **Ensemble apo+fasc folds** — Frundles notes larger 512×768 fasc hurt; try 2–3 fold average at 384×576 first.
6. **Per-device FL factor** instead of global ×0.9 (Frundles §5); MultExp suggests FL affine unstable — prefer device medians from MultExp / residual audit.
7. **MT centreline sampling refinements** — Vera centerline notes (public 0.45 worse; only steal MT tricks that help).
8. **Do not** chase attention/RL until floor ≤0.28; prior 1.95 was wrong paradigm.

## Hard constraints
- No hand-label test; no private-LB blending.
- MultExpData already declared in `reports/phase3/external_data_declare.md`.
- Leave ≥1 Kaggle submit spare each day.
- Prefer lab L40s (`chocheng9@144.214.209.211`) for fasc retrain; box CPU is fine for geometry.

## Success gates
| gate | publicScore | action |
|---|---|---|
| Floor done | ≤0.34 | ✅ 0.33641 |
| Open SOTA | ≤0.31 | Mawra-class texture/ensemble |
| Opting zone | ≤0.22 | fasc+geo polish |
| Chase target | ≤0.18 | needs closed-stack secrets or major fasc+scale wins |
