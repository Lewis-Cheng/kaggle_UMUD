# External data admission

Nothing in this file was downloaded or used for training.

Sealed-benchmark order, highest first:

1. Drazan, Hullfish, Baxter 2019 (Zenodo `10.5281/zenodo.2598553`) — repeated human traces.
2. SMA dataset v2 (DOI `10.17632/dpmf9bz8pt.2`) — non-network geometry baseline. Dense masks are not confirmed.
3. MuscleUS_Toolbox — local orientation. Human ROI, so not a fully automatic control.
4. DL_Track_US — full pipeline reproduction. Paper mean difference is not competition S. The Zenodo Examples record is not the training archive.

MultExp (35 architecture images) is already opened. It is a sealed diagnostic: score a frozen recipe once, do not select on it.

LabTrain and ExpDrawn stay development candidates. DeepACSA is representation learning only.

Field-level status is in [`external_admission_manifest.json`](external_admission_manifest.json). Competition rules were not re-checked in this run, so every new dataset stays `unchecked`.
