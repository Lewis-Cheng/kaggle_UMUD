# MultExp fold-pure 25-var selector — 2026-11-05 14:44 Asia/Hong_Kong

## Result (honest nested LOOCV)
- **Hold / prior C unchanged: S = 0.13195** (`nested_feature_rule_fasc_adapt`) — **not beaten**
- Best *new* transferable from this job: **S = 0.13294** (`nested_trainS_reduced` / safe = Frundles baseline)
  PA 0.928° / FL 2.297 mm / MT 0.158 mm
- Best learned selector this job: SoftmaxBandit v1 **S = 0.13328** (acc@1=0.057)
- vs C Δ_new = +0.00098 | leaky oracle **0.10485** | oracle−selector gap **~0.028**
- Acc@1/@3 (trainS_reduced) = 0.143 / 0.400
- **Gate 0.12: FAIL** | Promote: False | **No Kaggle**

## What was tried
1. Rebuilt 25-variant A/blend pred matrices (same pool as leaky oracle 0.10485)
2. Rich fold-pure features (map stats + PA/FL disagreement across variants)
3. Nested LOOCV: SoftmaxBandit, L1/Ridge residual, ExtraTrees/RF/GB/logreg/kNN/MLP, feature_rule, hybrids
4. v2: reduced (14) and safe (6) pools + imitation bandit

## Top methods (this job)
| Method | S | beats C |
|--------|---|---------|
| leaky_oracle | 0.10485 | yes (not nested) |
| **hold C feature_rule_fasc_adapt** | **0.13195** | — |
| nested_trainS_reduced/safe | 0.13294 | no |
| nested_feature_rule_safe | 0.13345 | no |
| nested_softmax_bandit (v1) | 0.13328 | no |
| nested_residual_full | 0.13416 | no |
| nested_trainS_full | 0.13506 | no |
| trees / MLP / knn | 0.14–0.20 | no |

## Diagnosis
Oracle ceiling ~0.105 clears 0.12, but fold-pure selectors top out at ~0.133 (gap ~0.028). Acc@1 stays ≤0.20 — features do not identify the per-image oracle variant. Expanding the A/blend pool does not help transfer; trainS on reduced pool collapses to baseline 0.13294.

## Next lever
Stop spending on fold-pure selectors for Frundles+A. Need fasc maps outside that family (MultExp-paired labels / non-U-Net generators). Hold remains S=0.13195; no Kaggle.
