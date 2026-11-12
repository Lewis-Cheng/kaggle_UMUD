# Spatial-attention + medical-RL geometry: specification

Status: specification for an untrained pipeline. Nothing here is a public-score claim.
Incumbent: `texture_rl`, public 0.29895. Target: public 0.16.

This follows the Beneficial-AI-Foundation formal-verification-skills (FVS) order:
natural-language spec (this file), then Lean statements (`formal/umud_rl/`), then a
fresh review of the statements, then code whose contracts mirror the statements.
FVS verifies Lean and Aeneas-translated Rust. It cannot verify a PyTorch network.
The Lean proofs cover the discrete RL environment and the angle definition. The
network is covered only by executable contracts and held-out measurements.

## What the pipeline is for

Geometry measured on the competition masks differs from geometry measured on
Frundles maps by 4.11 deg PA, 22.1 mm FL, and 0.30 mm MT
(`oracle_mask_gap.json`, 18 images). The masks annotate the fascicles that
annotators chose to draw, and the maps light up every fascicle-like streak. The
pipeline learns that choice from the 1048 images that have both apo and fasc
masks, then refines each chosen line with an RL agent.

This does not reach 0.16 by construction. To reach 0.16, PA MAE must fall by
2.50 deg, or FL MAE by 5.00 mm, or some mix of the two. The map-to-mask gap is
large enough to cover that, but only if the masks agree with the hidden labels
better than the maps do. That is the open hypothesis this pipeline tests.

## Data

- Input per image: grayscale image, frozen Frundles `p_apo` and `p_fasc`, all
  resized to the model grid (default 192x288).
- Supervision: competition apo and fasc masks. Never MultExp. Any training image
  within dHash Hamming distance 8 of a MultExp image is dropped.
- Splits: dHash groups (union-find, max distance 12). No group is in both train
  and validation.

## Feed-forward model (`umud.models.spatial_attention_geo`)

Outputs on the model grid:

1. `attn`: probability that a pixel lies on an annotated fascicle.
2. `orient`: `(cos 2t, sin 2t)` of the fascicle direction `t`, in pixel axes.
3. `sup`, `deep`: two separate aponeurosis band maps. The model, not a
   "first two bands" rule, decides which band is superficial and which is deep.

Horizontal flip test-time augmentation maps `t` to `-t`, so the flipped `sin 2t`
is negated before averaging. The averaged output is flip-equivariant by
construction.

## RL agent (`umud.rl.fascicle_agent`)

- State: a line hypothesis `(y, t)` through a fixed anchor column `x0`. Units
  are 1 grid pixel for `y` and `DELTA_DEG = 0.5` deg for `t`.
- Actions: `up`, `down`, `rot+`, `rot-`, `stop`. Each move changes one
  coordinate by exactly one unit.
- Observation: a patch rotated to the current line, with channels image,
  `p_fasc`, and `attn`, plus the angle relative to the deep aponeurosis.
- Reward: potential-based shaping matched to the learner's discount `g`:
  `r_t = phi(s_t) - g * phi(s_{t+1}) - c`, where `phi` is the distance to the
  episode's target annotated line and `c >= 0` is a step cost. `stop` earns 0.
- Discount: `g = 1`. The deployment objective is "end close, in few moves":
  minimise `phi(last) + c * T`. Only `g = 1` ranks trajectories of any two
  lengths by that objective (`shaped_return_objective`). With `g < 1` the
  return is `phi(first) - g^T phi(last) - c (1 + ... + g^(T-1))`. That ranks
  equal-length trajectories correctly (`disc_return_order`), but across lengths
  the `g^T` weight pays the agent for running out the clock. A `g = 0.95` agent
  learned exactly that on synthetic data: training return rose while held-out
  line angle error went from 0.73 to 3.66 deg (`synthetic_run_v3_gamma095.json`).
  A test exhibits a longer, farther episode with the higher `g = 0.95` return.
- Stabilising `g = 1`: unscaled `g = 1` DQN diverged (TD loss 0.03 to 0.88 over
  20k steps, `dqn_gamma1_diverged.log`). The learner now scales rewards by 0.1
  (a positive scale leaves the greedy policy unchanged), with lr 5e-4 and target
  sync every 1000 steps.
- Undiscounted shaping fed to a discounted learner also breaks the equal-length
  ordering (a test shows a violation).
- Episode: at most `T_MAX` moves. The environment ends the episode on `stop`,
  and it treats a move into an already visited state as `stop`, in training
  and at inference. The observation includes `moves / T_MAX`.
- Seeds at inference: attention peaks inside the deep-superficial compartment,
  with `t` read from `orient`.

## Readout (`umud.geometry.attended_readout`)

These are the frozen host-faithful definitions:

- PA: `|fold(angle_mm(line) - slope_deg(deep, x))|`, with
  `fold(a) = ((a + 90) mod 180) - 90`.
- FL: the distance between the line's intersections with the superficial and
  deep centre curves, i.e. the straight extrapolated line. Not `T / sin(PA)`.
- MT: the deep centre minus the superficial centre, minus half of each band's
  thickness, averaged at 20/50/80 % of the shared extent.
- Pixel-to-mm conversion uses the frozen per-image `(sx, sy)` from
  `scale_aniso.csv` (test) or `Scale_pixel_per_cm` (MultExp). Nothing is learned
  about scale.
- No FL affine. PA and MT calibration is off by default. If enabled, it is the
  LOOCV affine fitted on MultExp only.

## Proven in Lean (`formal/umud_rl/UmudRl/Basic.lean`)

| id | statement |
|---|---|
| `shaped_return_telescopes` | Over any trajectory, the shaped return equals `phi(first) - phi(last) - c * moves`. |
| `shaped_return_order` | Between two trajectories from the same start with the same number of moves, the higher return is exactly the one that ends closer to the target. |
| `shaped_return_order_free` | With `c = 0`, the same holds for any two trajectory lengths. |
| `shaped_return_objective` | Undiscounted, any two lengths: return order is exactly the order of `phi(last) + c * moves`, the deployment objective. |
| `disc_return_telescopes` | With shaping `phi s - g phi s' - c` and discount `g`, the return equals `phi(first) - g^T phi(last) - c * geom(g, T)`. |
| `disc_return_order` | For `g > 0`, same start and same number of moves: higher discounted return iff closer final state. |
| `rollout_length_le` | A rollout under any history-dependent policy has at most `T + 1` states. |
| `rollout_drift` | Every state in a rollout is within `T` units of the seed in both `y` and `t`. |
| `fold_range` | `-90 <= fold a < 90`. |
| `pa_range` | `0 <= pa a <= 90`. |
| `pa_flip` | `pa (-a) = pa a`: negating the relative angle does not change PA. |

`pa_flip` covers only the fold-and-absolute-value step. That the whole readout
is invariant under a horizontal flip (the line angle and the deep slope both
change sign) is checked end to end by a test, not proven.

Scope of the Lean model: `Line` is the integer lattice the agent moves on, so
the rollout statements are about the environment itself. The reward identities
are stated over `Int`. `disc_return_telescopes` is a polynomial identity with
integer coefficients that holds for every integer input, so it holds in every
commutative ring, including the reals with `0 < g < 1`. `disc_return_order` is
proven for integer `g >= 1`. For real `g` in `(0, 1)` the ordering follows from
the identity and `g^T > 0`, and the mirror tests check it at `g = 0.9`. Python
floats are checked to 1e-9. The angle statements are on whole degrees. Float
`fold` is tested separately, including the `x % 180 == 180.0` rounding
boundary.

The axiom audit must show only `propext`, `Classical.choice`, and `Quot.sound`,
or fewer, and there must be no `sorry`. The statement text of each theorem is
pinned in the tests, so a weakened statement fails even if it still proves.

## Mirrored executable contracts (`tests/test_attention_rl_contracts.py`)

The Python code is not translated to Lean. Each Lean statement has a numeric
mirror that runs against the actual Python functions, and each mirror is paired
with a mutant that must fail:

- Shaped rewards from `FascicleEnv` telescope to `phi0 - phiT - c * moves`
  (mutant: an environment subclass that pays a terminal bonus).
- With `g`-matched shaping, the discounted return from `FascicleEnv` equals the
  closed form for `g` in {0.5, 0.9, 0.95}; the case `g = 1` is the
  telescoping test above.
- Same-move-count returns order by final potential, undiscounted and at
  `g = 0.9` (mutant: undiscounted shaping fed to a `g = 0.9` learner violates
  the order).
- Undiscounted returns of a 1-5 move and a 20-40 move episode order by
  `phi(last) + c * moves` (witness: at `g = 0.95` a longer episode that ends
  farther away earns more).
- Rollout length is at most `T_MAX + 1`, states are never revisited, and drift
  is at most `T_MAX` units, for random and greedy Q policies (mutants: a larger
  `t_max`, and an environment whose moves are two units).
- `fold` and `pa` match the Lean integer definitions on `[-1000, 1000]`, stay in
  range on random and boundary floats, and `pa` is flip-invariant.
- The flip-TTA model output is equivariant (mutant: forgetting to negate
  `sin 2t`).
- The full readout gives the same PA, FL, and MT on a horizontally flipped
  image (mutant: flipping positions but not line angles).
- The readout's FL is the extrapolated-line length, not `MT / sin(PA)`
  (mutant: the sine formula differs on curved aponeuroses).
- The readout equals host-faithful `measure` when fed the same masks.
- The training manifest contains no image within dHash distance 8 of MultExp
  (mutant: a planted near-duplicate raises).

## Promotion gates (`scripts/chase018/fv_attention_rl_stages.py`)

Each gate aborts with its stage id. A pipeline that is all green has earned one
Kaggle submission, not a score.

1. `contracts`: all mirrored contracts pass.
2. `lean_build`: `lake build` succeeds, there is no `sorry`, and the axioms are
   on the allow-list.
3. `val_attention_beats_maps`: on dHash-held-out competition images, attended
   geometry is closer to mask geometry than Frundles-map geometry, for both PA
   and FL.
4. `val_rl_beats_init`: on the same held-out images, RL-refined lines have a
   lower angle and offset error than their feed-forward seeds. If not, RL is
   disabled and the stage records BLOCKED. RL is never kept on faith.
5. `multexp_not_worse`: on MultExp, which no model saw, the candidate is not
   worse than host-faithful `measure` on the same maps. This is a stress test,
   not a leaderboard proxy: the honest MultExp score of 0.134 coexisted with a
   public score of about 0.44.
6. `movement_budget`: the mean normalized movement against `texture_rl` must be
   at least the 0.16 requirement of 0.13895. If it is smaller, the candidate
   cannot reach 0.16 even if every change helps.
7. `no_joint_length_shrink`: the median FL and MT ratios against `texture_rl`
   must not both fall below 0.999. The 0.97 shrink lost publicly (0.35218).
8. `official_ids`: 309 literal ids, unique, matching the official list.

Single-target CSVs (PA only, FL only) are written next to the joint CSV so that
a public result can be attributed to one target.

## Synthetic results (CPU, `SYNTHETIC_NOT_LB`)

Generator: `umud.data.synthetic_us`, 800 train / 150 val / 200 test images,
with 3 annotated fascicles among unannotated streaks and central-band
distractors. Host `measure` on the ground-truth masks reproduces the labels to
0.19 deg PA, 0.69 mm FL, and 0.04 mm MT. Same feed-forward checkpoint
throughout (`synthetic_ckpts/ff_v1.pt`, 20 epochs).

| test, n = 200 | MAE PA | MAE FL | MAE MT | S1 |
|---|---|---|---|---|
| host `measure` on maps | 4.87 | 15.40 | 1.29 | 0.842 |
| attention readout | 0.41 | 2.14 | 0.075 | 0.090 |
| + RL, `g = 0.9`, undiscounted shaping (v1) | 0.72 | 3.82 | 0.075 | 0.154 |
| + RL, `g = 0.95`, matched shaping (v3) | 1.73 | 6.25 | 0.075 | 0.278 |
| + RL, `g = 1`, scaled rewards (v4, current default) | 0.67 | 2.51 | 0.075 | 0.115 |

RL by seed type on the test images, v4 agent (`rl_seed_diag.json`):

| seed | n | angle err before -> after | lines improved |
|---|---|---|---|
| feed-forward lines | 866 | 0.83 -> 1.66 deg | 13 % |
| small perturbation (sd 2 deg, 2 px) | 600 | 1.66 -> 1.45 deg | 52 % |
| large perturbation (+-12 deg) | 600 | 6.00 -> 2.09 deg | 83 % |

The v3 agent averaged 13.6 moves from feed-forward seeds against 5.7 for v4,
which is the run-out-the-clock behaviour that `g < 1` rewards.

Reading: the feed-forward attention model does the work on synthetic data, and
`val_rl_beats_init` is false, so RL stays disabled. The RL agent is a useful
corrector only when its seed is several degrees off. On synthetic data almost
every image's feed-forward error is under 2 deg. Real feed-forward lines will be
worse (real fasc Dice is about 0.41), so the lab validation reports
`rl_by_init_error` to show which regime the real data is in.

## Explicit non-claims

- No public-score improvement is shown by any synthetic or offline number.
- The texture_rl bandit chose `cal_pa_mt` on all 309 test images. It is a
  constant affine and does not show that RL is required.
- RL here is a line-refinement policy trained on thousands of mask-annotated
  lines. It is not trained on the 35 MultExp labels.
