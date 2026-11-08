# Residual AmbrosM-unknown scale audit

- unknown rows: **1373** / 4118
- by pool: `{'fasc': 987, 'apo': 386}`

## By family

| family | n_unknown |
|---|---:|
| `other_1080x1640_RGB` | 645 |
| `rgb_800x1200_other` | 277 |
| `other_500x760_RGB` | 216 |
| `other_652x800_RGB` | 118 |
| `other_600x800_RGB` | 79 |
| `other_556x660_RGB` | 23 |
| `other_688x1234_RGB` | 13 |
| `other_600x800_L` | 1 |
| `other_688x1074_RGB` | 1 |

## Decision

- parsers added: **none**
- reason: Remaining unknown families lack AmbrosM/xiaoleilian formulas and have no documented mm-per-tick q. Tick-like edges without q remain unidentifiable (sy=q/d). Do not invent q.

Test coverage remains 309/309 from AmbrosM v2; residual unknowns are train/apo/fasc only.

