#!/usr/bin/env python3
"""Audit released manual endpoints and existing UTT outputs, not UMUD predictions.

Requires numpy/scipy and a checkout of timvanderzee/RUB_ultrasound_study.
No training, annotation, scale fitting, CSV generation, or Kaggle operation.
Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC11784871/
Source coordinate conversions: visualization/fig7.m and visualization/fig8.m
of the above repository. Calibration values remain author-reported, not an
observed calibration anchor for any competition image.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angle(vector):
    return np.degrees(np.arctan2(vector[1], vector[0]))


def audit(root):
    data = root / "data/ultrasound"
    settings_path = data / "TA_video/HybridTrack/exampleVideoTA_trackingResults.mat"
    settings = loadmat(settings_path, simplify_cells=True)["settings"]
    results = []
    for sequence, relative, scales, provenance in [
        ("TA_video", "UltraTimTrack/exampleVideoTA_Verheul_2022_tracked_Q=0001.mat",
         (float(settings["horzmm"]), float(settings["vertmm"])),
         "Author metadata settings.horzmm/vertmm in HybridTrack MAT; px/mm"),
        ("calf_raise", "UTT/calf_raise_tracked_Q=0001.mat",
         (508/65, 508/65),
         "Author conversion 508/65 px/mm in visualization/fig8.m; isotropic assumption"),
    ]:
        observers = {}
        counts = {}
        hashes = {}
        for observer in ("BR", "PT", "TZ"):
            path = data / sequence / "manual" / f"Manual_Tracking_{observer}.mat"
            hashes[str(path.relative_to(root))] = digest(path)
            contents = loadmat(path, simplify_cells=True)
            fasc, apo = contents["FasData"], contents["ApoData"]
            frames = np.atleast_1d(fasc["digitizedFrames"]).astype(int)
            aframes = np.atleast_1d(apo["digitizedFrames"]).astype(int)
            assert len(set(frames)) == len(frames), "Duplicate manual frame"
            assert np.array_equal(frames, aframes), "Endpoint frame alignment mismatch"
            labels = {}
            for i, frame in enumerate(frames):
                fpoints = np.asarray(fasc["pts"][i], dtype=float)
                apoints = np.asarray(apo["pts"][i], dtype=float)
                assert fpoints.shape == apoints.shape == (2,2)
                # Translation/padding offsets cancel in coordinate differences.
                fv = (fpoints[1]-fpoints[0])/np.asarray(scales)
                av = (apoints[1]-apoints[0])/np.asarray(scales)
                pa = abs((angle(fv)-angle(av)+90) % 180-90)
                labels[int(frame)] = (pa, float(np.linalg.norm(fv)))
            observers[observer] = labels
            counts[observer] = len(labels)
        common = sorted(set.intersection(*(set(x) for x in observers.values())))
        consensus = np.asarray([
            np.mean([x[frame] for x in observers.values()], axis=0)
            for frame in common])
        path = data / sequence / relative
        hashes[str(path.relative_to(root))] = digest(path)
        region = loadmat(path,simplify_cells=True)["Fdat"]["Region"]
        predictions = np.stack((region["PEN"],region["FL"]), axis=1)
        frame_indices = np.asarray(common)-1
        assert frame_indices.min() >= 0 and frame_indices.max() < len(predictions)
        errors = predictions[frame_indices]-consensus
        results.append({
            "sequence":sequence,
            "manual_frame_counts":counts,
            "all_three_raters_common_frames":len(common),
            "frame_range_one_based":[common[0],common[-1]],
            "pixels_per_mm_xy":list(scales),
            "scale_provenance":provenance,
            "manual_consensus":"mean of three raters, joined on original frame number",
            "MAE_PA_deg":float(np.abs(errors[:,0]).mean()),
            "MAE_FL_mm":float(np.abs(errors[:,1]).mean()),
            "RMSE_PA_deg":float(np.sqrt((errors[:,0]**2).mean())),
            "RMSE_FL_mm":float(np.sqrt((errors[:,1]**2).mean())),
            "MT_available":False,
            "full_UMUD_score":None,
            "input_sha256":hashes,
        })
    return {
        "status":"Existing external outputs versus released manual annotations only",
        "baseline_texture_rl_evaluated":False,
        "competition_scale_resolved":False,
        "public_improvement_established":False,
        "raw_video_alignment_verified":False,
        "raw_input_license_and_overlap_cleared":False,
        "source":"https://github.com/timvanderzee/RUB_ultrasound_study",
        "settings_sha256":digest(settings_path),
        "results":results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout",type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.checkout),indent=2))
