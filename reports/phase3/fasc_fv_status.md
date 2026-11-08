# Fasc FV status — `fasc_thin_ceiling`

**Date (Asia/Hong_Kong):** 2026-11-04 22:39 HKT  
**Harness:** `scripts/baselines/fasc_fv_stages.py`  
**JSON:** `reports/phase3/fasc_fv_status.json`

## Block

- **id:** `fasc_thin_ceiling`
- **failing number:** native macro Dice **0.2846** ≪ usable bar **0.40**
- **attention_blocked:** **YES** (unchanged)
- **Official S:** null

## Stage histogram

### Box smoke (synthetic + rank JSON)
| Stage | Result |
|---|---|
| `align_nearest` | PASS |
| `edt_radius` | PASS |
| `mstiny_forward` | PASS |
| `deep_sup_aux` | PASS |
| `loss_mild_weights` | PASS |
| `native_macro_dice` | PASS (rank JSON; eval JSON absent on box) |
| `vs_v5_floor` | PASS (no candidate yet) |
| `aux1_probe` | PASS |
| ckpt_smoke | absent on box |

### SSH (2026-11-04 22:38 HKT) — real v5 ckpt + eval JSON
| Stage | Result |
|---|---|
| `align_nearest` | PASS |
| `edt_radius` | PASS |
| `mstiny_forward` | PASS |
| `deep_sup_aux` | PASS |
| `loss_mild_weights` | PASS |
| `native_macro_dice` | PASS (eval JSON matched floor) |
| `vs_v5_floor` | PASS (skipped_no_candidate) |
| `aux1_probe` | PASS |
| ckpt_smoke | **loaded** `seg_fasc_v5_best.pt` |

## Mutant `fasc_v12_aux1_ms`

- Code: `MSTinyUNet(n_aux=3)` + aux1 w=0.1; all other v5 knobs frozen
- Train script: `scripts/baselines/train_seg_fasc_v12.py`
- **Training started: YES**
  - host: SSH `chocheng9@144.214.209.211`
  - tmux: `umud_fasc_v12`
  - device: **cuda:1** (GPU0 left free; GPU2/3 busy with other users' dro-scale-check)
  - warm-start: `seg_fasc_v5_best.pt` (aux1 random init)
  - log: `reports/phase3/run_fasc_v12.log`
  - ckpt out: `reports/phase3/seg_ckpts/seg_fasc_v12_best.pt`

## Next gate

1. Wait for v12 epochs; native macro Dice must **> 0.2846** (`vs_v5_floor`) or abort/reject mutant.
2. If still ≪ 0.40 → keep `attention_blocked=YES`.
3. Do not start attention / encoder-family trains.