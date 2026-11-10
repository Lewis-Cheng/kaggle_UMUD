/-
Statements for `umud.rl.fascicle_agent` and the PA definition in
`umud.geometry.attended_readout`. Spec: reports/phase3/chase018/attention_rl/SPEC.md.

Scope: `Line` is the discrete lattice the agent moves on (`y` in model-grid
pixels, `t` in `DELTA_DEG` steps), so `rollout_length_le` and
`rollout_drift` are the statements about the Python environment itself. The
reward identities are stated over `Int` and hold in any ordered ring, so they
also hold for real-valued potentials; Python floats are covered by the mirror
tests with a 1e-9 tolerance. `fold`/`pa` are stated on whole degrees. The
Python code is not extracted; tests/test_attention_rl_contracts.py runs each
statement against the real functions, with a mutant per statement.
-/

namespace UmudRl

structure Line where
  y : Int
  t : Int
deriving Repr, DecidableEq

inductive Act where
  | up | down | rotP | rotN | stop
deriving Repr, DecidableEq

def Act.apply : Act → Line → Line
  | .up, l => ⟨l.y - 1, l.t⟩
  | .down, l => ⟨l.y + 1, l.t⟩
  | .rotP, l => ⟨l.y, l.t + 1⟩
  | .rotN, l => ⟨l.y, l.t - 1⟩
  | .stop, l => l

/-- The policy sees the visited states, newest first, so a revisit stop is
a policy choice and needs no separate case. -/
def rollout (pi : List Line → Act) : Nat → List Line → Line → List Line
  | 0, _, l => [l]
  | n + 1, h, l =>
    if pi (l :: h) = Act.stop then [l]
    else l :: rollout pi n (l :: h) ((pi (l :: h)).apply l)

theorem rollout_length_le (pi : List Line → Act) :
    ∀ (n : Nat) (h : List Line) (l : Line), (rollout pi n h l).length ≤ n + 1
  | 0, h, l => by simp [rollout]
  | n + 1, h, l => by
    by_cases hs : pi (l :: h) = Act.stop
    · simp [rollout, hs]
    · have ih := rollout_length_le pi n (l :: h) ((pi (l :: h)).apply l)
      simp [rollout, hs]
      omega

theorem apply_step (a : Act) (l : Line) :
    l.y - 1 ≤ (a.apply l).y ∧ (a.apply l).y ≤ l.y + 1 ∧
    l.t - 1 ≤ (a.apply l).t ∧ (a.apply l).t ≤ l.t + 1 := by
  cases a <;> simp [Act.apply] <;> omega

theorem rollout_drift (pi : List Line → Act) :
    ∀ (n : Nat) (h : List Line) (l m : Line), m ∈ rollout pi n h l →
      l.y - (n : Int) ≤ m.y ∧ m.y ≤ l.y + (n : Int) ∧
      l.t - (n : Int) ≤ m.t ∧ m.t ≤ l.t + (n : Int)
  | 0, h, l, m, hm => by
    simp [rollout] at hm
    subst hm
    omega
  | n + 1, h, l, m, hm => by
    by_cases hs : pi (l :: h) = Act.stop
    · simp [rollout, hs] at hm
      subst hm
      omega
    · simp [rollout, hs] at hm
      rcases hm with hm | hm
      · subst hm
        omega
      · have ih := rollout_drift pi n (l :: h) _ m hm
        have st := apply_step (pi (l :: h)) l
        omega

/-- Last state of `s :: rest`. -/
def final : Line → List Line → Line
  | s, [] => s
  | _, b :: rest => final b rest

/-- Sum over moves of `phi s - phi s' - c`. -/
def shapedReturn (phi : Line → Int) (c : Int) : List Line → Int
  | [] => 0
  | [_] => 0
  | a :: b :: rest => (phi a - phi b - c) + shapedReturn phi c (b :: rest)

theorem shaped_return_telescopes (phi : Line → Int) (c : Int) :
    ∀ (s : Line) (rest : List Line),
      shapedReturn phi c (s :: rest) = phi s - phi (final s rest) - c * (rest.length : Int)
  | s, [] => by simp [shapedReturn, final]
  | s, b :: rest => by
    have ih := shaped_return_telescopes phi c b rest
    simp only [shapedReturn, final, ih, List.length_cons]
    rw [Int.natCast_add, Int.mul_add]
    omega

/-- Same start, same number of moves: higher return iff closer final state. -/
theorem shaped_return_order (phi : Line → Int) (c : Int) (s : Line) (r1 r2 : List Line)
    (hlen : r1.length = r2.length) :
    shapedReturn phi c (s :: r1) ≤ shapedReturn phi c (s :: r2) ↔
      phi (final s r2) ≤ phi (final s r1) := by
  rw [shaped_return_telescopes, shaped_return_telescopes, hlen]
  generalize c * (r2.length : Int) = k
  omega

/-- Without a step cost the path does not matter, only where it ends. -/
theorem shaped_return_order_free (phi : Line → Int) (s : Line) (r1 r2 : List Line) :
    shapedReturn phi 0 (s :: r1) ≤ shapedReturn phi 0 (s :: r2) ↔
      phi (final s r2) ≤ phi (final s r1) := by
  rw [shaped_return_telescopes, shaped_return_telescopes]
  simp only [Int.zero_mul]
  omega

/-- Undiscounted, any two lengths: the return ranks trajectories exactly by
final potential plus step cost, which is the deployment objective. -/
theorem shaped_return_objective (phi : Line → Int) (c : Int) (s : Line) (r1 r2 : List Line) :
    shapedReturn phi c (s :: r1) ≤ shapedReturn phi c (s :: r2) ↔
      phi (final s r2) + c * (r2.length : Int) ≤ phi (final s r1) + c * (r1.length : Int) := by
  rw [shaped_return_telescopes, shaped_return_telescopes]
  generalize c * (r1.length : Int) = k1
  generalize c * (r2.length : Int) = k2
  omega

/-- `g`-consistent shaping for a discounted learner: a move pays
`phi s - g * phi s' - c`, and later moves are discounted by `g`. -/
def discReturn (phi : Line → Int) (g c : Int) : List Line → Int
  | [] => 0
  | [_] => 0
  | a :: b :: rest => (phi a - g * phi b - c) + g * discReturn phi g c (b :: rest)

/-- `1 + g + ... + g^(n-1)`. -/
def geom (g : Int) : Nat → Int
  | 0 => 0
  | n + 1 => 1 + g * geom g n

/-- A polynomial identity in `g`, `c` and the potentials. Proven for all
integers, it holds in every commutative ring, so also for real `g`. -/
theorem disc_return_telescopes (phi : Line → Int) (g c : Int) :
    ∀ (s : Line) (rest : List Line),
      discReturn phi g c (s :: rest) =
        phi s - g ^ rest.length * phi (final s rest) - c * geom g rest.length
  | s, [] => by simp [discReturn, final, geom, Int.pow_zero]
  | s, b :: rest => by
    have ih := disc_return_telescopes phi g c b rest
    simp only [discReturn, final, ih, List.length_cons, geom, Int.pow_succ]
    rw [Int.mul_sub, Int.mul_sub, Int.mul_add, Int.mul_one, Int.mul_comm (g ^ rest.length) g,
      Int.mul_assoc g (g ^ rest.length), Int.mul_left_comm g c]
    omega

theorem pow_pos_of_pos {g : Int} (hg : 0 < g) : ∀ n : Nat, 0 < g ^ n
  | 0 => by simp [Int.pow_zero]
  | n + 1 => by
    rw [Int.pow_succ]
    exact Int.mul_pos (pow_pos_of_pos hg n) hg

/-- With `g > 0`, same start and same number of moves: higher discounted
return iff closer final state. The equal-length hypothesis is load-bearing:
for `0 < g < 1` the final potential is weighted by `g ^ T`, so a longer
episode that ends farther away can earn more. -/
theorem disc_return_order (phi : Line → Int) (g c : Int) (hg : 0 < g) (s : Line)
    (r1 r2 : List Line) (hlen : r1.length = r2.length) :
    discReturn phi g c (s :: r1) ≤ discReturn phi g c (s :: r2) ↔
      phi (final s r2) ≤ phi (final s r1) := by
  rw [disc_return_telescopes, disc_return_telescopes, hlen]
  have hp : 0 < g ^ r2.length := pow_pos_of_pos hg r2.length
  constructor
  · intro h
    have : g ^ r2.length * phi (final s r2) ≤ g ^ r2.length * phi (final s r1) := by omega
    exact Int.le_of_mul_le_mul_left this hp
  · intro h
    have := Int.mul_le_mul_of_nonneg_left h (Int.le_of_lt hp)
    omega

def fold (a : Int) : Int := (a + 90) % 180 - 90

def pa (a : Int) : Int := if fold a < 0 then -fold a else fold a

theorem fold_range (a : Int) : -90 ≤ fold a ∧ fold a < 90 := by
  unfold fold
  omega

theorem pa_range (a : Int) : 0 ≤ pa a ∧ pa a ≤ 90 := by
  unfold pa fold
  split <;> omega

theorem pa_flip (a : Int) : pa (-a) = pa a := by
  unfold pa fold
  split <;> split <;> omega

end UmudRl
