# Seven blocks against the coupled PA/FL diagnosis

None of the seven blocks is solved. `any_of_seven_solved` is false. Production stays `BLOCKED`.

The private data and weight repos named in `STORAGE.md` on `main` are not visible to this token, so no held-out image was scored. The numbers below are identities and a static reading of this tree. They are not a new leaderboard result, and they do not show that S can fall below 0.12.

## What the score check does close

`score_components` already returns the mean, not the sum. On the supplied MAEs 1.127°, 2.489 mm, and 0.162 mm:

- normalized sum = 0.44925
- S1 = 0.14975
- shares = PA 41.8%, FL 46.2%, MT 12.0%

That matches the pinned scorer. It does not remeasure those MAEs.

## What the geometry check does not close

At 50 mm and 15°, a +1° pennation error with thickness held fixed moves fascicle length by −3.257 mm to first order. The identity is in `src/umud/verification/coupling.py`. No per-image signed residuals were available, so it has not been shown to explain the FL error.

`mm_per_prob_pixel` is the right resize Jacobian when it is given px/mm on each axis. Ten MultExp call sites pass the same `px_per_mm` for both axes. The test infer path (`infer_texture_pa.py`, `infer_geo_floor.py`) passes `sx` and `sy` separately. Forcing `sy = sx` on a pixel step `(1, 1)` with mm/px `(0.1, 0.2)` moves the physical angle from 63.43° to 45°. The earlier synthetic `coordinate_pullback` pass used unequal spacing on purpose, so it did not see this call-site split.

`measure` estimates MT from centreline separation minus half the band thickness. Pennation uses the centreline polynomial slope, not the inner-edge tangent. `line_curve_intersection` does not reject a near-parallel crossing. The frozen affine is least squares (`np.polyfit`), not absolute deviation. The 3D tractography branch was not opened.

Machine-readable verdict: [`seven_blocks_math_check.json`](seven_blocks_math_check.json).
