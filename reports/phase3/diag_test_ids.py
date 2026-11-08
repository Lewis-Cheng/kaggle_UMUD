from pathlib import Path
import sys
ROOT = Path.home() / "kaggle_UMUD"
sys.path.insert(0, str(ROOT / "src"))
from umud.data.layout import discover_layout, list_images

test_root = ROOT / "data/raw/umud/test_images_v2"
files = sorted(p for p in test_root.rglob("*") if p.is_file())
print("test_files", len(files))
print("first", files[:3])
print("last", files[-3:] if files else None)

L = discover_layout(ROOT / "data/raw/umud")
imgs = list_images(L.test_imgs)
print("layout.test_imgs", L.test_imgs)
print("list_images", len(imgs))
print("first_imgs", imgs[:3])
print("last_imgs", imgs[-3:] if imgs else None)

# v1 submission ids
import csv
v1 = ROOT / "reports/phase3/submission_tier_b_geometry.csv"
rows = list(csv.DictReader(v1.open(encoding="utf-8-sig"), delimiter=";"))
print("v1_n", len(rows), "first", rows[0]["image_id"], "last", rows[-1]["image_id"])

# do v1 ids match test stems?
stems = {p.stem for p in imgs}
v1_stems = {Path(r["image_id"]).stem for r in rows}
print("v1_in_test", len(v1_stems & stems), "v1_only", len(v1_stems - stems), "test_only", len(stems - v1_stems))
# scale v2 coverage for v1 ids
from umud.baselines.ambrosm_scale_io import load_ambrosm_scale_rows, scale_map_by_stem
scales = scale_map_by_stem(load_ambrosm_scale_rows(ROOT / "reports/phase3/ambrosm_scale_v2_rows.csv"))
cov = {"measured":0,"family_prior":0,"unknown":0,"missing":0}
for s in v1_stems:
    sc = scales.get(s)
    if sc is None: cov["missing"] += 1
    else: cov[sc.provenance] = cov.get(sc.provenance,0)+1
print("v1_id_scale_cov", cov)
