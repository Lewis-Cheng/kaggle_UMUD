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
