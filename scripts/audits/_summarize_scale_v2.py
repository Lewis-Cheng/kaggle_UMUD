#!/usr/bin/env python3
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.home() / "kaggle_UMUD"
rows = list(csv.DictReader((ROOT / "reports/phase3/ambrosm_scale_v2_rows.csv").open()))
# also load v1 for before/after
v1 = list(csv.DictReader((ROOT / "reports/phase3/ambrosm_scale_rows.csv").open()))

def cov(rs):
    n = len(rs)
    w = sum(1 for r in rs if r.get("px_per_mm"))
    prov = Counter(r["provenance"] for r in rs)
    return n, w, prov

print("=== BEFORE (v1) ===")
n,w,p = cov(v1); print("all", n, w, dict(p), f"{w/n:.1%}")
for pool in ["test","apo","fasc"]:
    rs=[r for r in v1 if r["pool"]==pool]
    n,w,p=cov(rs); print(pool, n, w, dict(p), f"{w/n:.1%}")

print("\n=== AFTER (v2) ===")
n,w,p = cov(rows); print("all", n, w, dict(p), f"{w/n:.1%}")
for pool in ["test","apo","fasc"]:
    rs=[r for r in rows if r["pool"]==pool]
    n,w,p=cov(rs); print(pool, n, w, dict(p), f"{w/n:.1%}")

print("\n=== v2 by family ===")
fam = defaultdict(list)
for r in rows:
    fam[r["family"]].append(r)
for f, items in sorted(fam.items(), key=lambda kv: -len(kv[1])):
    n,w,p = cov(items)
    notes = Counter(r["scale_note"] for r in items)
    print(f"{f}: n={n} with_px={w} prov={dict(p)} notes={dict(notes.most_common(4))}")

print("\n=== fasc unknown remaining ===")
fu=[r for r in rows if r["pool"]=="fasc" and r["provenance"]=="unknown"]
print(Counter(r["family"] for r in fu))

print("\n=== delta ===")
print("gained_px", cov(rows)[1]-cov(v1)[1])
print("fasc_gained", cov([r for r in rows if r["pool"]=="fasc"])[1]-cov([r for r in v1 if r["pool"]=="fasc"])[1])
