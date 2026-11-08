#!/usr/bin/env python3
import json
from pathlib import Path
p = Path("/home/chocheng9/kaggle_UMUD/reports/phase3/chase018/multexp_nonfrundles_fasc_classical.json")
d = json.loads(p.read_text())
print("best_outside_family:")
print(json.dumps(d.get("best_outside_family"), indent=2))
print("frundles_baseline_best:")
print(json.dumps(d.get("frundles_baseline_best"), indent=2))
print("promoted", d.get("promoted"), "gate_pass", d.get("gate_pass"), "beats_hold", d.get("beats_hold"))
ars = d.get("all_results") or []
print("n_results", len(ars))
top = sorted(ars, key=lambda r: r.get("S", 999))[:10]
for t in top:
    print(
        "S={:.5f} src={} thr={} ok={}".format(
            t.get("S", 999), t.get("source"), t.get("thr"), t.get("ok_n", t.get("n"))
        )
    )
