#!/usr/bin/env python3
"""Read-only UMUD score-distance certificates, not a public-score predictor.

Does not train, change geometry, fit scale, generate predictions, or submit.
The current on-server fv_output_stages.py remains authoritative.

Usage:
  python umud_score_certificate.py --self-test
  python umud_score_certificate.py --prove       # requires z3-solver
  python umud_score_certificate.py --baseline baseline.csv \
      --candidate candidate.csv --expected-ids official_ids.txt

CSV mode requires an exact official ID list, copied from the input inventory.
For a known public-set size, add --public-count K. Without it, the certificate
is conservative for every nonempty subset, not just the whole 309-row set.
Optional manifest audit requires a pinned executable SHA256; no code execution.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import statistics
from pathlib import Path

TARGETS = ("pa_deg", "fl_mm", "mt_mm")
TOLS = (6.0, 12.0, 3.0)
HEADER = ("image_id", *TARGETS)
ID_PATTERN = re.compile(r"IMG_[0-9]{5}\.(?:tif|png)\Z")


def validate_ids(ids):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Official IDs must be nonempty and unique")
    if not all(ID_PATTERN.fullmatch(x) for x in ids):
        raise ValueError("IDs must be copied official IMG_#####.tif/.png names")


def read_predictions(text, expected_ids):
    validate_ids(expected_ids)
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != HEADER:
        raise ValueError("Require exact comma-separated header; no BOM")
    result = {}
    for row in reader:
        name = row["image_id"]
        if name in result:
            raise ValueError("Duplicate image_id")
        if None in row:
            raise ValueError("Extra CSV fields")
        try:
            values = tuple(float(row[k]) for k in TARGETS)
        except (TypeError, ValueError):
            raise ValueError("Missing or nonnumeric target") from None
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite target")
        result[name] = values
    if set(result) != set(expected_ids):
        raise ValueError("Exact official ID set mismatch")
    return [result[name] for name in expected_ids]


def audit_manifest(manifest, executable_bytes, expected_sha256):
    # Intentionally fail-closed. No reinterpretation of an enabled FL affine.
    if manifest.get("fl_mm", {}).get("applied") is not False:
        raise ValueError("ABORT: FL affine flag is not explicitly false")
    for key in ("pa_deg", "mt_mm"):
        if manifest.get(key, {}).get("applied") is not True:
            raise ValueError(f"ABORT: {key} affine is not explicitly enabled")
    observed = hashlib.sha256(executable_bytes).hexdigest()
    if observed != expected_sha256:
        raise ValueError("ABORT: executable hash mismatch")
    return {"sha256": observed, "flags_pass": True,
            "semantic_reproduction_proved": False}


def score(pred, truth):
    """Formula supplied in the task/repository; no live scorer authentication."""
    terms = []
    for j, tol in enumerate(TOLS):
        e = [abs(p[j] - y[j]) for p, y in zip(pred, truth)]
        terms.append((statistics.mean(e)
                      + 1e-6 * statistics.median(e)
                      + 1e-9 * math.sqrt(statistics.mean(x*x for x in e))) / tol)
    return statistics.mean(terms)


def distance_certificate(baseline, candidate, public_count=None):
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("Nonempty aligned prediction arrays required")
    if any(len(row) != 3 or not all(math.isfinite(x) for x in row)
           for row in [*baseline, *candidate]):
        raise ValueError("Require three finite targets per row")
    n = len(baseline)
    if public_count is not None and (type(public_count) is not int
                                     or not 1 <= public_count <= n):
        raise ValueError("public_count must be in [1,n]")
    d = [[abs(c[j]-b[j]) for j in range(3)]
         for b, c in zip(baseline, candidate)]
    per_row = [sum(row[j]/TOLS[j] for j in range(3))/3 for row in d]
    k = public_count or 1  # k=1 is conservative for any nonempty unknown subset
    s1 = statistics.mean(sorted(per_row, reverse=True)[:k])
    # Median is Lipschitz in the infinity norm; NOT generally in mean abs delta.
    s2 = sum(max(row[j] for row in d)/TOLS[j] for j in range(3))/3
    # Reverse triangle inequality for Euclidean norm, with conservative top-k.
    s3 = sum(math.sqrt(statistics.mean(sorted(
        (row[j]**2 for row in d), reverse=True)[:k]))/TOLS[j]
             for j in range(3))/3
    full = statistics.mean(per_row)
    shifts = [statistics.mean(c[j]-b[j] for b,c in zip(baseline,candidate))
              for j in range(3)]
    return {
        "n_rows": n,
        "public_count": public_count,
        "full_set_S1_distance": full,
        "subset_S1_bound": s1,
        "subset_S2_bound": s2,
        "subset_S3_bound": s3,
        "absolute_score_change_bound": s1 + 1e-6*s2 + 1e-9*s3,
        "mean_absolute_changes": dict(zip(TARGETS, [
            statistics.mean(row[j] for row in d) for j in range(3)])),
        "mean_signed_changes": dict(zip(TARGETS, shifts)),
        "both_length_means_shrink": shifts[1] < 0 and shifts[2] < 0,
        "meaning": "Necessary movement budget only; not expected gain or approval",
        "public_improvement_proved": False,
    }


def assess_target_bound(bound, baseline_score, target, uncertainty):
    """Conservative numerical screen, including displayed-score rounding."""
    values = (bound, baseline_score, target, uncertainty)
    if not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("Score, target, uncertainty and bound must be finite and nonnegative")
    minimum_gap = max(0.0, baseline_score - uncertainty - target)
    # A tolerance is not an interval-arithmetic proof of the Python program.
    epsilon = 1e-10 * max(1.0, bound, baseline_score, target, uncertainty)
    ruled_out = bound + epsilon < minimum_gap
    return {
        "baseline_displayed_score": baseline_score,
        "baseline_score_uncertainty": uncertainty,
        "target_gap_nominal": baseline_score - target,
        "target_gap_lower_bound": minimum_gap,
        "numerical_comparison_tolerance": epsilon,
        "target_ruled_out_by_numerical_distance_screen": ruled_out,
        "public_improvement_proved": False,
        "interpretation": (
            "Conditional numerical screen under the supplied score formula, "
            "baseline-score interval, exact baseline file and scored-subset assumptions; "
            "not formal verification of floating-point execution or an authenticated score."
        ),
    }


def proofs():
    """Exact-real SMT proofs; scope excludes image/label correctness and floats."""
    import z3
    a, b, y, tol = z3.Reals("a b y tol")
    absz = lambda x: z3.If(x >= 0, x, -x)
    claims = []

    def prove(name, assumptions, counterexample):
        solver = z3.Solver()
        solver.set(timeout=10000)
        solver.add(*assumptions, counterexample)
        status = solver.check()
        if status != z3.unsat:
            raise AssertionError(f"{name}: expected unsat, got {status}")
        claims.append({"name": name, "counterexample_search": str(status)})

    prove("absolute_error_is_1_lipschitz", [],
          absz(absz(a-y)-absz(b-y)) > absz(a-b))
    prove("positive_normalization_preserves_bound", [tol > 0],
          absz(absz(a-y)/tol-absz(b-y)/tol) > absz(a-b)/tol)

    # A complete two-dimensional metric is not identified by a vertical anchor.
    dx, gy, bound = z3.Reals("dx gy bound")
    # Choose gxx=(bound^2+1)/dx^2, gxy=0, gyy=gy. It is SPD under assumptions.
    gxx = (bound*bound+1)/(dx*dx)
    prove("vertical_anchor_allows_arbitrarily_long_horizontal_vector",
          [dx > 0, gy > 0, bound >= 0],
          z3.Or(gxx <= 0, gxx*gy <= 0, dx*dx*gxx <= bound*bound))

    # Same known S1 baseline value, opposite effects of FL+6 mm.
    base = z3.RealVal(70)
    cand = z3.RealVal(76)
    s = z3.RealVal("0.29895")
    offset = 36*s
    lower, upper = base-offset, base+offset
    prove("known_scalar_does_not_determine_improvement", [],
          z3.Not(z3.And(
              absz(base-lower)/36 == s,
              absz(base-upper)/36 == s,
              absz(cand-lower)/36 > s,
              absz(cand-upper)/36 < s,
              lower >= 30, upper <= 200)))
    return {"solver": z3.get_version_string(), "exact_real_claims": claims,
            "not_proved": ["public score improvement", "physical scale",
                           "annotation compatibility", "floating-point code"]}


def self_test():
    ids = ["IMG_00001.tif", "IMG_00252.png"]
    good = "image_id,pa_deg,fl_mm,mt_mm\nIMG_00001.tif,15,70,20\nIMG_00252.png,20,80,25\n"
    baseline = read_predictions(good, ids)
    cases = {
        "rebuilt_stem": good.replace("IMG_00001.tif", "IMG_00001"),
        "changed_extension": good.replace("IMG_00252.png", "IMG_00252.tif"),
        "duplicate": good.replace("IMG_00252.png", "IMG_00001.tif"),
        "missing_row": "\n".join(good.splitlines()[:2])+"\n",
        "extra_row": good+"IMG_00300.png,15,70,20\n",
        "nan": good.replace(",15,", ",nan,", 1),
        "inf": good.replace(",15,", ",inf,", 1),
        "bom": "\ufeff"+good,
        "delimiter": good.replace(",", ";"),
        "extra_field": good.replace(",15,70,20", ",15,70,20,99"),
    }
    rejected = []
    for name, text in cases.items():
        try:
            read_predictions(text, ids)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(f"Mutant accepted: {name}")
    m = {"fl_mm":{"applied":False}, "pa_deg":{"applied":True},
         "mt_mm":{"applied":True}}
    digest = hashlib.sha256(b"fixture").hexdigest()
    audit_manifest(m, b"fixture", digest)
    for name, manifest, hashed in [
        ("enabled_FL", {**m,"fl_mm":{"applied":True}}, digest),
        ("missing_FL", {k:v for k,v in m.items() if k!="fl_mm"}, digest),
        ("wrong_executable_hash", m, "0"*64)]:
        try:
            audit_manifest(manifest,b"fixture",hashed)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(name)
    reversed_rows = good.splitlines()[0]+"\n"+"\n".join(good.splitlines()[1:][::-1])+"\n"
    assert read_predictions(reversed_rows,ids) == baseline
    rng = random.Random(20261109)
    for _ in range(1000):
        n = rng.randrange(1,13)
        b = [tuple(rng.uniform(1,150) for _ in range(3)) for _ in range(n)]
        c = [tuple(v+rng.uniform(-10,10) for v in row) for row in b]
        y = [tuple(rng.uniform(1,150) for _ in range(3)) for _ in range(n)]
        k = rng.randrange(1,n+1)
        subset = rng.sample(range(n),k)
        gap = abs(score([b[i] for i in subset],[y[i] for i in subset])
                  -score([c[i] for i in subset],[y[i] for i in subset]))
        for count in (None,k):
            assert gap <= distance_certificate(b,c,count)["absolute_score_change_bound"]+1e-10
    shrink = [(p, f*.97,t*.97) for p,f,t in baseline]
    assert distance_certificate(baseline,shrink)["both_length_means_shrink"]
    assert distance_certificate(baseline,baseline)["absolute_score_change_bound"] == 0
    assert not assess_target_bound(.138949, .29895, .16, .000005)[
        "target_ruled_out_by_numerical_distance_screen"]
    assert assess_target_bound(.1, .29895, .16, .000005)[
        "target_ruled_out_by_numerical_distance_screen"]
    for bad in (float("nan"), float("inf"), -1.0):
        try:
            assess_target_bound(.1, bad, .16, .000005)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid baseline score accepted")
    return {"rejected_mutants":rejected, "random_subset_bound_checks":2000,
            "exact_ID_alignment":"pass","zero_change":"pass",
            "joint_length_shrink_flag":"pass",
            "rounded_score_interval_and_invalid_score_checks":"pass",
            "scope":"Synthetic contract tests, not real-image validation"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prove", action="store_true")
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--expected-ids", type=Path)
    ap.add_argument("--public-count", type=int)
    ap.add_argument("--baseline-score", type=float, default=.29895)
    ap.add_argument("--baseline-score-uncertainty", type=float, default=.000005,
                    help="Half-unit for a five-decimal displayed score; increase if needed")
    ap.add_argument("--target", type=float, default=.16)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--executable", type=Path)
    ap.add_argument("--expected-executable-sha256")
    args = ap.parse_args()
    if (args.executable or args.expected_executable_sha256) and not args.manifest:
        ap.error("Executable audit options require --manifest")
    out = {}
    if args.self_test:
        out["tests"] = self_test()
    if args.prove:
        out["proofs"] = proofs()
    if args.baseline or args.candidate:
        if not all((args.baseline,args.candidate,args.expected_ids)):
            ap.error("CSV audit needs baseline, candidate and copied expected IDs")
        ids = args.expected_ids.read_text().splitlines()
        bb, cb = args.baseline.read_bytes(), args.candidate.read_bytes()
        b,c = (read_predictions(x.decode("utf-8"),ids) for x in (bb,cb))
        cert = distance_certificate(b,c,args.public_count)
        cert["baseline_sha256"] = hashlib.sha256(bb).hexdigest()
        cert["candidate_sha256"] = hashlib.sha256(cb).hexdigest()
        cert["target_screen"] = assess_target_bound(
            cert["absolute_score_change_bound"], args.baseline_score,
            args.target, args.baseline_score_uncertainty)
        cert["submission_authorized"] = False
        out["certificate"] = cert
    if args.manifest:
        if not args.executable or not args.expected_executable_sha256:
            ap.error("Manifest audit needs executable and independently pinned SHA256")
        out["manifest"] = audit_manifest(json.loads(args.manifest.read_text()),
            args.executable.read_bytes(), args.expected_executable_sha256)
    if not out:
        ap.error("Use --self-test, --prove or --baseline with --candidate")
    print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
