#!/usr/bin/env python3
"""Write consolidated Lewis-pivot status for non-Frundles MultExp fasc chase."""
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path("/home/chocheng9/kaggle_UMUD")
OUT = ROOT / "reports/phase3/chase018"
HK = timezone(timedelta(hours=8))
now = datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z")
HOLD = 0.13195144580335014

def load(p):
    p = Path(p)
    if not p.exists():
        return None
    return json.loads(p.read_text())

classical = load(OUT / "multexp_nonfrundles_fasc_classical.json")
lab = load(OUT / "multexp_nonfrundles_fasc_labtrain.json")
ft = load(OUT / "multexp_nonfrundles_fasc_labtrain_comp_ft.json")
blend = load(OUT / "multexp_nonfrundles_fasc_blend.json")
dl = load(OUT / "multexp_nonfrundles_fasc_dltrack.json")
inv = load(ROOT / "data/external/osf_xbawc/osf_xbawc_inventory.json")

rows = []
if classical:
    b = classical.get("best_outside_family") or {}
    rows.append(("classical_frangi", b.get("S"), b.get("thr"), "outside-family classical"))
    fb = classical.get("frundles_baseline_best") or {}
    rows.append(("frundles_baseline", fb.get("S"), fb.get("thr"), "reference"))
if lab:
    b = lab.get("best") or {}
    rows.append(("labtrain_imagenet", b.get("S"), b.get("thr"), "OSF LabTrain 1854; MultExp never in train"))
if ft:
    b = ft.get("best") or {}
    rows.append(("labtrain_comp_ft", b.get("S"), b.get("thr"), "LabTrain→FT LabTrain+comp; MultExp dHash-excluded"))
if blend and blend.get("best"):
    b = blend["best"]
    rows.append(("blend_best", b.get("S"), b.get("thr"), "tag={} a={}".format(b.get("tag"), b.get("alpha_b"))))
if dl and dl.get("best"):
    b = dl["best"]
    rows.append(("dltrack_vgg16", b.get("S"), b.get("thr"), "Zenodo DLTrack fasc + Frundles apo"))

# best outside-family (exclude frundles_baseline and pure frundles blend a=0)
outside = []
for name, s, thr, note in rows:
    if s is None:
        continue
    if name in ("frundles_baseline",):
        continue
    if name == "blend_best" and blend and blend["best"].get("alpha_b") == 0.0:
        continue
    outside.append((name, s, thr, note))
best_out = min(outside, key=lambda x: x[1]) if outside else None

md = []
md.append("# Lewis pivot — MultExp fasc outside Frundles+A")
md.append(f"Updated: {now}\n")
md.append("## Scores vs hold 0.13195 / gate 0.12\n")
md.append("| run | S | thr | note |")
md.append("|---|---:|---:|---|")
for name, s, thr, note in rows:
    flag = ""
    if s is not None:
        if s < 0.12:
            flag = " GATE"
        elif s < HOLD:
            flag = " HOLD"
        s_s = f"{s:.5f}{flag}"
    else:
        s_s = "pending"
    md.append(f"| {name} | {s_s} | {thr} | {note} |")
md.append("")
if best_out:
    md.append(
        f"**Best outside-family MultExp S = {best_out[1]:.5f}** ({best_out[0]}, thr={best_out[2]}) — "
        f"{'BEATS' if best_out[1] < HOLD else 'MISS'} hold; "
        f"{'PASS' if best_out[1] < 0.12 else 'FAIL'} gate."
    )
md.append("")
md.append("## LabTrain / OSF assets landed\n")
if inv:
    md.append(f"- fasc images/masks: {inv.get('fasc_images_n')}/{inv.get('fasc_masks_n')}")
    md.append(f"- apo images zip: {inv.get('apo_images_zip')} bytes; apo masks n={inv.get('apo_masks_n')}")
    md.append(f"- DLTrack h5: {inv.get('dltrack_h5_n')} — {inv.get('dltrack_h5')}")
    md.append(f"- OSF Muscle Architecture Fascicles/Aponeurosis folders: empty")
    md.append(f"- tensorflow in main venv: {inv.get('tensorflow')} (DLTrack uses `.venv_tf` if present)")
else:
    md.append("- inventory json missing")
md.append("- filename overlap LabTrain↔competition: **0**")
md.append("- MultExp never in train; FT dropped 24 near-MultExp dHash neighbors")
md.append("")
md.append("## Train protocol\n")
md.append("1. Classical Frangi/ridge/Gabor fasc maps + Frundles apo → frozen recipe (done).")
md.append("2. ImageNet-init SMP ResNet34 U-Net, 25 epochs, LabTrain fasc 1482/372 group-split, 384×576.")
md.append("3. Fine-tune that ckpt 15 epochs on LabTrain+competition (4591 after MultExp dHash filter).")
md.append("4. Blend Frundles fasc with LabTrain(/FT) fasc alphas; apo Frundles.")
md.append("5. Zenodo DL_Track_US fasc VGG16-BCE-512 h5 dump (if TF venv succeeds).")
md.append("")
md.append("## Promote rule\n")
md.append("Promote only if MultExp S < 0.13195. Current outside-family best does **not** promote.")
md.append("No Kaggle. No attention heads.")
md.append("")

status = "\n".join(md) + "\n"
(OUT / "multexp_nonfrundles_fasc_status.md").write_text(status)
summary = {
    "ts": now,
    "hold": HOLD,
    "gate": 0.12,
    "rows": [{"name": n, "S": s, "thr": t, "note": note} for n, s, t, note in rows],
    "best_outside_family": (
        {"name": best_out[0], "S": best_out[1], "thr": best_out[2], "note": best_out[3]}
        if best_out
        else None
    ),
    "promoted": bool(best_out and best_out[1] < HOLD),
    "gate_pass": bool(best_out and best_out[1] < 0.12),
    "labtrain_assets": inv,
}
(OUT / "multexp_nonfrundles_fasc_lewis_pivot.json").write_text(json.dumps(summary, indent=2))
print(status)
print("best_outside", best_out)
