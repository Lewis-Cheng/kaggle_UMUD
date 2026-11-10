# UMUD: a defensible path toward public 0.16

Research and executable audit, 9 November 2026. The recommended next experiment is **annotation-aware, multi-hypothesis geometry using frozen maps**, with image registration used to verify fascicle identity where genuine neighboring frames exist. External manual endpoint data should help develop and falsify that estimator, not become another surrogate leaderboard. No public improvement has been established, no new network was trained, and no submission was made.

## Decision and current status

Keep `texture_rl`, public **0.29895**, as the incumbent. The supplied `submission_ref_geometry_v2.csv` remains an untested candidate, not a proven improvement; its harness pass establishes neither label correctness nor public 0.16.

The research identified three useful directions:

- **Best new estimator hypothesis:** retain multiple aponeurosis-pair and fascicle-line hypotheses until image evidence selects a consistent muscle compartment and a small set of distinct fascicles. Preserve the existing inner-border MT, deep-aponeurosis PA, and extrapolated straight-line FL definitions.
- **Best primary external validation lead:** the original Drazan/GasFasTrack manual measurements, because its FL and PA definitions use fascicle intersections and the deep aponeurosis, respectively ([Drazan study](https://peerj.com/articles/7120/)). Its full study contains five participants and 75 analyzed videos, but manual labels sample only six frames per video; it is not thousands of independent subjects or frames ([Drazan study](https://peerj.com/articles/7120/)).
- **Best immediately inspectable endpoint resource:** the released manual files used in the UltraTimTrack study, including explicit endpoint coordinates and a source-specific anisotropic conversion for the TA sequence ([manual annotation code](https://github.com/timvanderzee/RUB_ultrasound_study/blob/main/processing/ultrasound/ManualTracking.m), [coordinate conversion code](https://github.com/timvanderzee/RUB_ultrasound_study/blob/main/visualization/fig7.m)). They support implementation tests, not replacement of the public baseline.

### Repository access and abort

Direct cloud SSH to `chocheng9@144.214.209.211` failed with a connection reset. A second attempt from the connected Windows device timed out. The accessible GitHub mirror was cloned read-only and inspected at commit `83613c6`, dated 8 November 2026; its README, chase018 status reports, calibration manifest, geometry implementation, and external-data declaration were read.

That mirror is behind the handoff: `src/umud/calibration/executable.py` and `scripts/chase018/fv_output_stages.py` are absent, and `multexp_chase_affine.json` still contains `"fl_mm": {"applied": true}`. **Reproduction from this snapshot was aborted.** The flag was not changed, FL affine was not enabled, `physical.py` was not modified, and the mirror remained clean. All new research and verification files are outside that repository.

The user's latest handoff supersedes historical README language that calls MultExp an offline promotion gate. Neither the current server candidate nor the current checkpoints were executed in this audit.

## What the score gap actually requires

Using the supplied leading score term,

\[
S_1=\frac{\operatorname{MAE}_{PA}}{18}
    +\frac{\operatorname{MAE}_{FL}}{36}
    +\frac{\operatorname{MAE}_{MT}}{9},
\]

the required reduction from 0.29895 to 0.16 is **0.13895**, larger than the 0.12895 reduction needed for 0.17. These are error reductions, not merely changes in predictions.

| Target score | Required reduction in \(S_1\) | PA-only MAE reduction | FL-only MAE reduction | MT-only MAE reduction |
|---|---:|---:|---:|---:|
| 0.17 | 0.12895 | 2.3211° | 4.6422 mm | 1.1606 mm |
| 0.16 | 0.13895 | 2.5011° | 5.0022 mm | 1.2506 mm |

A planning example is a 1.5° PA MAE reduction plus a 2.1 mm FL MAE reduction, with MT unchanged. That would reduce the leading score by 0.14167 to approximately 0.15728; it is an engineering budget, not a forecast.

### Movement is necessary, not evidence of improvement

For baseline predictions \(b\), candidate predictions \(c\), and the actual scored subset \(P\), the reverse triangle inequality gives:

\[
\left|S_1(c,y;P)-S_1(b,y;P)\right|
\leq
\frac1{|P|}\sum_{i\in P}
\left(
\frac{|c_{i,PA}-b_{i,PA}|}{18}
+\frac{|c_{i,FL}-b_{i,FL}|}{36}
+\frac{|c_{i,MT}-b_{i,MT}|}{9}
\right).
\]

Using the handoff's approximate whole-test mean changes, a PA-only 0.50° edit has a whole-test movement budget of only 0.02778. The geometry-v2 changes of 0.50°, 6.0 mm, and 0.49 mm have a whole-test budget of about 0.24889, sufficient in magnitude but not direction.

**Whole-test averages are not public-subset certificates.** Without the public membership, a rigorous bound needs either the largest per-row movement, or the mean of the largest \(k\) movements if the public count \(k\) is known. The accompanying script makes this distinction and includes conservative bounds for the small median-AE and RMSE terms in the repository's score formula.

The numerical target screen also allows a half-unit uncertainty for a five-decimal displayed baseline score, defaulting to 0.000005. Its conclusions are conditional on the supplied score formula, an authenticated baseline file/score association and the scored-subset assumptions; the Python calculation is not interval-arithmetic verification.

The failed 3% joint FL/MT shrink is useful empirical evidence against repeating that operation. It does not prove that expanding both targets will help, identify which length caused the loss, or recover any image's physical scale.

## External datasets: which are actually useful?

### Prioritized qualification

| Resource | Verified evidence | Recommended role | Blocking caveat |
|---|---|---|---|
| **Drazan / GasFasTrack** | Five adults, medial gastrocnemius, 75 videos; three examiners on three days; six manually sampled frames per video. FL uses intersections and PA uses the deep aponeurosis ([Study](https://peerj.com/articles/7120/)). | First candidate for subject-held-out FL/PA evaluation and tracking-based hypothesis selection | Download and parse original manual labels, not automatic tracks. Verify frame IDs, scales, permissions and overlap before use; MT is not a supplied independent target in the study ([Study](https://peerj.com/articles/7120/)). |
| **UltraTimTrack manual comparison files** | Three observers; published comparison uses 100 TA frames and 84 GM frames; released code stores endpoints and applies separate x/y scaling for TA ([Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11784871/), [conversion](https://github.com/timvanderzee/RUB_ultrasound_study/blob/main/visualization/fig7.m)). | Immediate endpoint/parser/geometry test fixture; later a small external diagnostic | Only two comparison sequences, not a generalization gate. Raw unannotated-video alignment, provenance overlap, and the particular manual-file license still need clearance; do not conflate these files with the separate eight-person dataset ([Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11784871/)). |
| **TimTrack four-dataset bundle** | Paper reports 339 manually compared images: 153 VL, 56 GL, 100 GM from Drazan, and 30 GM from SMA. Software and the first two image sets are CC0; the reused image sets are CC BY 4.0 ([Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265752), [repository](https://github.com/timvanderzee/ultrasound-automated-algorithm)). | Primitive orientation/boundary testing, particularly VL coverage | Do not import its scalar PA/FL as if they were UMUD labels: its automatic code uses superficial-aponeurosis PA and `thickness/sin(angle)`. Audit each manual field separately ([Executable algorithm](https://github.com/timvanderzee/ultrasound-automated-algorithm/blob/master/auto_ultrasound.m)). |
| **Original SMA data** | GM samples A/B and TA sample C; CC BY 4.0. The paper's manual reference uses three fascicles and three thickness sites per image, averaged per analysis ([Data record](https://data.mendeley.com/datasets/dpmf9bz8pt/2), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7015391/)). | Measurement-aggregation diagnostic, after recovering original manual annotations | The data-record page alone does not establish a ready-to-use scalar-label table. The TimTrack version reuses these images, so it is not a second independent cohort ([Data record](https://data.mendeley.com/datasets/dpmf9bz8pt/2), [TimTrack paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265752)). |
| **AnkleImage** | Ten participants overall; TA and plantarflexor recordings, with PA/FL obtained through UltraTrack and task-specific manual material; some MT outputs are available ([Data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC11680872/)). | Later motion-consistency work, after a field-level audit | Tracked labels are not independent expert truth; the BioStudies page did not expose files in this session, and exact spatial-scale/MT definitions and dataset licensing were not established ([Data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC11680872/), [archive entry](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1482)). |
| **FALLMUD** | 812 images combining Cunningham and Cronin material with complementary masks; its Kaggle record states CC BY-NC 4.0 ([Provider](https://kalisteo.cea.fr/index.php/fallmud/), [Kaggle record](https://www.kaggle.com/datasets/angeliqueloesch/fallmud)). | Do not prioritize | More masks are not the requested lever; shared ancestry creates overlap risk. Retain the repository's existing non-use decision unless there is a separately justified, license-cleared need. |
| **2026 multimodal TA gait dataset** | PA/FL are UltraTimTrack outputs with manual initialization but no later manual correction; MT is calculated as FL × sin(PA) ([Data paper](https://www.nature.com/articles/s41597-026-08156-5_reference.pdf?error=cookies_not_supported&code=ac299134-97ab-488b-9119-5bd1484bc465)). | Reject as independent scalar supervision for this effort | It would teach the prohibited measurement shortcut and reproduce another algorithm's errors, not supply independent architecture ground truth. |

The original Drazan Zenodo record is CC BY 4.0 and lists `matlab.zip` (487.4 kB), `Sample Videos.zip` (591.3 MB), and `validation_data.zip` (1.9 GB), alongside much larger participant archives ([Zenodo record](https://zenodo.org/records/2598553)). Start with the small code archive and the validation manifest rather than blindly downloading all raw videos.

The separate eight-participant UltraTimTrack Figshare dataset has verified CC BY 4.0 metadata and downloadable files, including `fascicle_tracking_estimates.zip` ([Figshare metadata](https://api.figshare.com/v2/articles/26502868)). That license does not by itself establish the license or independence of every older comparison-sequence/manual-label file in the study's GitHub repository, which uses previously published TA and GM videos ([study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11784871/)).

### What was inspected rather than merely found

The TimTrack repository was downloaded at commit `d3ba5e2b75fc57870e7a75b2fa68b5b7f53b10e9`. Its raw image directories contain 153, 56, 100 and 30 files respectively, matching the four reported subsets; its observer workbooks were opened without modification. The workbooks are not a universal ready-made table of UMUD PA/FL/MT labels.

For example, `dataset2_for_manual_estimation_obs2.xlsx`, sheet `range_of_motion`, cell `C1` says “Superficial aponeurosis angle”; cells `D1:E1` describe vertical aponeurosis points. In the VL workbook, sheet `TZ`, `D1:H1` are `alpha`, `betha`, `spoint`, `dpoint`, and `height`, with `H2 = G2-F2`. Those are primitive measurements, not verified deep-aponeurosis PA and extrapolated FL labels. The external label schema must retain that distinction.

The UltraTimTrack study repository was downloaded at commit `e0b3b7fe5a636c0ce7b78c0b20db33ba026c68eb`. Its endpoint files were parsed and aligned by original frame number, without changing 601 to 600 or filling missing rater frames. This is a fresh computation using released annotations and existing stored UTT predictions, not a rerun of UTT inference or an evaluation of `texture_rl`.

| Sequence | Frames common to all three raters | PA MAE | FL MAE | PA RMSE | FL RMSE |
|---|---:|---:|---:|---:|---:|
| TA comparison | 100 | 0.614° | 2.064 mm | 0.769° | 2.554 mm |
| GM calf raise | 84 | 0.635° | 1.806 mm | 0.896° | 2.223 mm |

These audit results use the released manual endpoints and existing model outputs from the [study repository](https://github.com/timvanderzee/RUB_ultrasound_study). They are broadly consistent with, but not claimed to exactly reproduce, the paper's rounded comparison results ([published comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC11784871/)).

For TA, the inspected metadata uses x/y spacings of 13.6842105263 and 13.6 pixels/mm; for GM, the inspected author script uses 508/65 pixels/mm under an isotropic assumption. These are source-specific author conversions, not an observed scale anchor for any competition frame. The study's TA conversion explicitly recomputes angles and lengths from endpoint differences rather than trusting stored unscaled fields ([TA conversion](https://github.com/timvanderzee/RUB_ultrasound_study/blob/main/visualization/fig7.m)).

There is no independent MT target in this two-sequence audit. Consequently, no full UMUD score is calculated, and these numbers must not become a new 0.16 promotion gate.

## The proposed new estimator

### Hypothesis: early geometric decisions discard the useful evidence

In the inspected mirror, `aponeuroses_rich()` keeps at most the first two retained depth bands, `measure()` aggregates fragments using length-weighted medians, and the texture pipeline already pools PA information within dHash groups. The repository also already contains structure-tensor/streamline FL work. Therefore, “add a structure tensor,” “smooth neighboring predictions,” or “copy TimTrack” is not a sufficiently new proposal.

The proposed change is **late selection among joint anatomy hypotheses**, rather than a different global angle statistic. It is new relative to the inspected implementation, not a claim of research-publication novelty.

There is a specific domain-shift reason to test this: the competition lists rectus femoris among test muscles but not among training muscles ([competition data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/data)). Rectus femoris can contain a visible central aponeurosis in addition to superficial structures, creating a plausible ambiguity for a first-two-bands rule ([anatomical review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12362324/)). This is a hypothesis about potential failures, not evidence that a particular hidden-test image or its target annotation uses a specific compartment.

### Proposed inference sequence

- **Preserve alternatives:** retain plausible aponeurosis bands and candidate ordered pairs rather than immediately selecting the shallowest two. Score whether fascicle evidence lies inside and connects the proposed compartment; do not assume the widest or thickest pair is correct.
- **Merge fragments by identity:** assemble collinear/ridge-supported fragments into distinct fascicle-line hypotheses. Long, duplicated fragments should not receive multiple votes merely because thresholding broke one physical fascicle into several components.
- **Use several visible fascicles:** select a small, spatially diverse set of supported lines and evaluate a three-line aggregation branch. This is motivated by the host's three-visible-fascicle annotation procedure, but the exact within-image aggregation and selection policy must not be silently asserted as known ([competition annotation description](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/data)).
- **Verify with neighboring images where available:** register tissue regions using classical forward/backward optical flow or local correlation, reject inconsistent matches, and support or reject line identities across frames. Do not average complete images without motion compensation, infer sequence membership from adjacent IDs, or carry a mistaken line indefinitely.
- **Keep the measurement implementation:** use the frozen host-faithful intersections and local deep-aponeurosis tangent. Do not output streamline arc length, switch to superficial-aponeurosis PA, or replace extrapolated FL with T/sin(PA).
- **Abstain when evidence is inadequate:** return the incumbent output for unresolved cases. The decision must depend on geometry and image evidence, not on whichever prediction is longer or on a hidden-label guess.

A candidate hypothesis can be ranked by a fixed, interpretable objective:

\[
J(h)=
\lambda_1\,\text{ridge support}
+\lambda_2\,\text{map support}
+\lambda_3\,\text{registration agreement}
-\lambda_4\,\text{intersection sensitivity}
-\lambda_5\,\text{duplicate-fragment penalty}.
\]

This is a proposed design, not fitted weights. Keep the initial candidate family small, select any weights inside grouped external development folds, and use a fixed rule or a small tabular ranker rather than another fascicle network.

### Why this could matter, and why it could fail

Changing which physical fascicles and aponeurosis pair are measured can alter PA and FL together by much more than a small affine adjustment. The existing 6 mm FL reordering candidate demonstrates that material movement is possible, but does not show that the new ordering is correct.

The largest risks are coherent false tracks, external-data definition mismatch, too few independent subjects, an incorrect compartment selection, and off-frame extrapolation sensitivity. A confidence rule that accepts only easy images may also affect too few cases to close the required score gap.

Initially copy MT exactly from `texture_rl`, so the experiment asks whether supported fascicle/pair selection improves PA and FL rather than mixing in a third uncertain change. Keep the current scale assumptions frozen and explicitly labeled as assumptions; even angles are affected if the x/y scale ratio is wrong.

UltraTimTrack itself remains a methodological precedent, not a drop-in contestant: its software has an interactive setup workflow, although its initial Kalman state is automatically derived from TimTrack estimates over selected initial frames ([software workflow](https://github.com/timvanderzee/UltraTimTrack), [study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11784871/)). Its principle of combining image-local line detections with motion information can be independently adapted with automatic setup from the frozen maps. Results on its two comparison sequences are not Kaggle evidence.

## Evaluation that can reject the idea

### Dataset admission before model selection

Every admitted record should carry dataset/version, original subject/sequence/frame IDs, image hash, label origin, annotated aponeurosis identity, FL definition, separate x/y scale provenance, coordinate transformations, rater ID, and license. Missing MT remains missing; it must not be synthesized from PA and FL.

Run exact and transformed-image overlap checks against competition training, test and MultExp data before using external labels. Group by original participant and acquisition sequence, including across repackaged datasets; a Drazan image appearing again in TimTrack is one source, not two validation domains. Test matches must be excluded from supervision, not used as label lookups.

The RF search did not establish a ready, openly downloadable, independently labeled longitudinal architecture set: one relevant study makes data available on request, while another describes 299 RF images but says data availability is “Not applicable” ([RF reliability study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9478722/), [RF image study](https://www.mdpi.com/2673-7426/3/2/27)). These are leads, not currently usable training resources.

### Four predeclared arms

| Arm | Changes from frozen incumbent | Purpose |
|---|---|---|
| A | None | Exact baseline reproduction |
| B | Joint aponeurosis-pair and distinct-fascicle selection, single frame | Test structural hypothesis selection |
| C | B plus verified cross-frame line correspondence | Test information from neighboring tissue motion |
| D | B plus the three-fascicle aggregation branch | Test annotation-aligned aggregation separately |

Do not launch a broad selector grid or combine C and D before their separate effects are understood. If B fails, C is not automatically justified merely because it produces smoother trajectories.

Report PA/FL MAE separately, grouped by participant, sequence, muscle and acquisition family; add MT only where independently annotated. Use participant- or sequence-level resampling for uncertainty, not thousands of correlated frames as independent samples.

Use an untouched external cohort as a final development check and MultExp as a named stress test. Neither can certify the public score. The two UTT comparison sequences are especially useful for parser and geometric regression tests but are too narrow to become the replacement reference model.

### Practical stop conditions

- **Insufficient candidate coverage:** external expert endpoints/angles are not represented by the candidate hypotheses. Better ranking cannot rescue a missing hypothesis; stop rather than train another prohibited fascicle network.
- **Wrong definition or unknown scale:** do not include that target in metric validation. Record the gap rather than estimating a metric from the desired output.
- **No reproducible PA/FL gain:** reject if improvements appear only in one fitted external set or depend on a single subject.
- **Wrong prediction direction:** reject another joint-length shrink under the user's policy, but do not force expansion to satisfy it.
- **Unstable inference:** reject hypotheses that flip with tiny image perturbations, disconnected batch additions, input row ordering, or equivalent coordinate transforms.
- **Too little movement:** calculate an honest movement bound; if it rules out the target, classify the candidate as insufficient even if it improves an external metric.

## Verification delivered and its limits

The requested `formal-verification` skill was unavailable. A standalone alternative was implemented using Z3 exact-real proofs plus executable contract tests; this is not presented as formal verification of the image pipeline.

`umud_score_certificate.py` completed:

- **Four SMT checks:** absolute-error Lipschitz bound; its positive normalization; unbounded horizontal metric length under only a vertical anchor; and a constructive example showing that the known scalar baseline does not determine whether a candidate improves.
- **Thirteen rejected mutants:** malformed/rebuilt IDs, extension mismatch, duplicate/missing/extra rows, NaN, infinity, BOM, wrong delimiter, extra fields, enabled/missing FL-affine flag, and wrong executable hash.
- **Two thousand numerical bound checks:** randomized scored subsets under known and unknown public-subset size.
- **Additional contracts:** row-order-independent ID alignment, exact zero movement, a joint-length-shrink flag, rounded-score uncertainty handling, and rejection of invalid baseline-score values.

For the known-scalar counterexample, take baseline FL 70 mm and candidate FL 76 mm, with PA and MT exact. A truth of 59.2378 mm and a truth of 80.7622 mm both give baseline \(S_1=0.29895\); the candidate worsens the first and improves the second. A scalar baseline and a large candidate gap cannot guarantee the next submission beats it.

The manifest helper checks explicit flags and a caller-supplied executable hash. It does **not** prove code semantics or reproduce the server's executable specification; the current `fv_output_stages.py` and a baseline byte-identity reproduction remain required.

The scale proof is a mathematical non-identifiability check, not a way to estimate scale. For a fixed positive \(g_{yy}\), choosing \(G=\operatorname{diag}(c,g_{yy})\) permits arbitrarily large horizontal lengths as \(c\) grows, while retaining the same vertical anchor. No change to `physical.py` is justified by the external-data audit.

### Reproduction

Run the standalone contracts:

```bash
python -m pip install z3-solver
python umud_score_certificate.py --self-test --prove
```

Run the external endpoint audit against the pinned public checkout:

```bash
git clone https://github.com/timvanderzee/RUB_ultrasound_study.git
git -C RUB_ultrasound_study checkout e0b3b7fe5a636c0ce7b78c0b20db33ba026c68eb
python -m pip install numpy scipy
python audit_external_utt.py RUB_ultrasound_study
```

The audit computes errors for existing stored external outputs, never a competition submission. It emits input hashes, frame-count checks, scale provenance, and explicit `false` flags for public improvement, competition scale resolution, and raw-input clearance.

## Submission decision

**No candidate is cleared by this report.** `submission_ref_geometry_v2.csv` should not be uploaded simply because it moves FL enough; it remains a separate existing candidate whose actual public performance is unknown.

The next actionable implementation step is to obtain the current server snapshot, reproduce `texture_rl` byte-for-byte under the current executable specification, and inspect candidate-hypothesis coverage on qualified external labels. After that, implement only arm B as a shadow path, with no training, no scale modification and no automatic submission.

Before any upload, check the current submission history and quota, preserve at least one daily submission, copy official filenames exactly, and require an actual submission receipt plus score rather than assuming CLI success. The competition permits five submissions per day and requires reproducible, documented methods; the user's stricter reserve policy applies ([competition rules](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/rules), [reproducibility requirements](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)).

The immediate blocker is access to the current README/chase018 snapshot, executable specification, harness, baseline CSV and candidate diagnostics. Research and standalone verification are complete; the public baseline remains 0.29895 and the public target remains 0.16.
