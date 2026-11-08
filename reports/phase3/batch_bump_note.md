# Batch bump policy (2026-11-04)

- Mid-run skip for live v6 (cuda:3 bs4 ~10GB) and v6b (cuda:2 bs6 ~6.6GB): **do not** kill mid-epoch.
- On next *resume* after death: prefer `--batch-size 8` (v6@640) / `--batch-size 10` (v6b@512) + AMP patch if added.
- v7 launched on cuda:1 with AMP + bs12 @576 from start.
- GPU0 left alone (other owner ~15GB).
