# Fasc formal-verification protocol (v5 structure)

**Date (Asia/Hong_Kong):** 2026-11-04  
**Method:** Anthropic-style formal verification (adapted from tri_attack skill) — turn a property into an **aborting** check (`raise` / nonzero exit), instrument named stages, fix smallest cause, re-check. Soft `assert` is not a property.

## Block

| Field | Value |
|---|---|
| **Block id** | `fasc_thin_ceiling` |
| **Failing number** | native macro Dice **0.2846** (`fasc_v5_edt_cldice_ms`) |
| **Usable / reopen bar** | ~**0.40** Dice (seg→geometry path) |
| **Official S** | `null` (no expert PA/FL/MT GT) |
| **attention_blocked** | **YES** — do not start attention; do not launch another encoder-family train |

Winner ckpt: `reports/phase3/seg_ckpts/seg_fasc_v5_best.pt`  
Eval record: `reports/phase3/fasc_native_rank.json` / `eval_seg_fasc_v5_best_native.json`

## Frozen v5 structure (baseline — do not mutate these knobs)

- NEAREST mask→img align (no letterbox / no thick-soft GT rewrite)
- EDT soft thickness **r = 2.5**
- Mild soft-clDice **w = 0.5**
- Hard Dice weight **0.5**
- MSTinyUNet **base 48 @ 512**
- Deep-sup aux@dec3/dec2 weights **0.3 / 0.2**
- Loss = BCE(soft) + softDice + 0.5·hardDice + 0.5·clDice

## Trusted stage chain

Harness: `scripts/baselines/fasc_fv_stages.py`

| Stage id | Property |
|---|---|
| `align_nearest` | Mask HW matches image after NEAREST resize; binary 0/1 only |
| `edt_radius` | Soft target radius frozen at 2.5 |
| `mstiny_forward` | MSTinyUNet returns logits; `return_aux` returns logits+aux |
| `deep_sup_aux` | aux3/aux2 present, same spatial size as logits; `n_aux=2` has no `aux1` (v5 ckpt compat) |
| `loss_mild_weights` | clDice w == 0.5 and hard Dice w == 0.5 (not amplified) |
| `native_macro_dice` | Eval path documented; if JSON present, recorded best ≈ 0.2846 ± 1e-4 |
| `vs_v5_floor` | Any candidate / mutant score must not fall below v5 floor 0.2846 |

Abort with the **named stage id** when a link breaks. Report stage histogram + pass/fail in `reports/phase3/fasc_fv_status.md` (+ JSON).

## Forbidden shortcuts (under this block)

- ImageNet / timm encoder swap (R18, R34, HRNet, EffNet, …)
- `edt_radius > 3.0`
- `clDice_w ≥ 1.0`
- Skeleton-primary GT (centerline as hard target)
- Attention training while `attention_blocked=YES`
- Deleting or overwriting `seg_fasc_v5_best.pt`
- Claiming LB / inventing official S

## Only allowed next mutant (under this structure)

**`fasc_v12_aux1_ms`** — complete deep-sup as the MSTinyUNet docstring claimed:

- Add **`aux1` on dec1** with small weight **0.1**
- Keep **all other v5 knobs identical** (align, EDT 2.5, clDice 0.5, hardDice 0.5, base48@512, aux3/aux2 = 0.3/0.2)
- API: `MSTinyUNet(..., n_aux=2|3)` default **2** so v5 ckpt loading stays strict-compatible

### Mutant property (aborting)

- Native macro Dice must **beat v5 0.2846** or harness / train gate aborts (`vs_v5_floor`).
- If still **≪ 0.40**, keep **`attention_blocked=YES`**.
- Do not start a second encoder-family train from this mutant’s failure.

## Workflow

1. Name block `fasc_thin_ceiling` + quote 0.2846.
2. Run `python scripts/baselines/fasc_fv_stages.py --check-aux1`.
3. Only if harness green: optionally launch `scripts/baselines/train_seg_fasc_v12.py` on a **free** GPU (leave busy GPUs alone).
4. Re-check `vs_v5_floor` with `--candidate-score <score> --require-candidate`.
5. Update status md/json; never edit a gate number without regenerating JSON.

## Commands

```bash
# smoke (CPU / box ok; synthetic + rank JSON)
python scripts/baselines/fasc_fv_stages.py --check-aux1

# on SSH with real ckpt + eval JSON
python scripts/baselines/fasc_fv_stages.py --check-aux1 \
  --ckpt reports/phase3/seg_ckpts/seg_fasc_v5_best.pt \
  --eval-json reports/phase3/eval_seg_fasc_v5_best_native.json

# after v12 eval
python scripts/baselines/fasc_fv_stages.py --require-candidate --candidate-score 0.XXXX
```
