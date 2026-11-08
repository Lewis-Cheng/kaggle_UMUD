import json
from pathlib import Path
p=Path("/home/chocheng9/kaggle_UMUD/reports/phase3/chase018/multexp_nonfrundles_fasc_classical.json")
d=json.loads(p.read_text())
bo=d.get("best_outside_family") or {}
bb=d.get("frundles_baseline_best") or {}
print("BEST_OUTSIDE", bo.get("S"), bo.get("tag"), bo.get("thr"), bo.get("MAE"))
print("BASELINE", bb.get("S"), bb.get("MAE"))
rows=sorted([r for r in d["all_results"] if "S" in r], key=lambda x:x["S"])[:10]
for r in rows:
    print(r["tag"], r["thr"], round(r["S"],5), round(r["MAE"]["pa_deg"],3), round(r["MAE"]["fl_mm"],3))
print("gate", d.get("gate_pass"), "beats_hold", d.get("beats_hold"))
