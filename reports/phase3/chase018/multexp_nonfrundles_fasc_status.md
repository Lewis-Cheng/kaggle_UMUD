# Lewis pivot — MultExp fasc outside Frundles+A
Updated: 2026-11-05 15:52:22 UTC+08:00

## Scores vs hold 0.13195 / gate 0.12

| run | S | thr | note |
|---|---:|---:|---|
| classical_frangi | 0.52740 | 0.35 | outside-family classical |
| frundles_baseline | 0.13294 | 0.35 | reference |
| labtrain_imagenet | 0.13818 | 0.25 | OSF LabTrain 1854; MultExp never in train |
| labtrain_comp_ft | 0.13779 | 0.25 | LabTrain→FT LabTrain+comp; MultExp dHash-excluded |
| blend_best | 0.13294 | 0.35 | tag=labtrain a=0.0 |
| dltrack_vgg16 | 0.14880 | 0.3 | Zenodo DLTrack fasc + Frundles apo |

**Best outside-family MultExp S = 0.13779** (labtrain_comp_ft, thr=0.25) — MISS hold; FAIL gate.

## LabTrain / OSF assets landed

- fasc images/masks: 1854/1854
- apo images zip: 305141391 bytes; apo masks n=1722
- DLTrack h5: 2 — ['labtrain_DLTrack_models/DLTrack_models/model-apo-VGG16-BCE-512.h5', 'labtrain_DLTrack_models/DLTrack_models/model-fasc-VGG16-BCE-512.h5']
- OSF Muscle Architecture Fascicles/Aponeurosis folders: empty
- tensorflow in main venv: False (DLTrack uses `.venv_tf` if present)
- filename overlap LabTrain↔competition: **0**
- MultExp never in train; FT dropped 24 near-MultExp dHash neighbors

## Train protocol

1. Classical Frangi/ridge/Gabor fasc maps + Frundles apo → frozen recipe (done).
2. ImageNet-init SMP ResNet34 U-Net, 25 epochs, LabTrain fasc 1482/372 group-split, 384×576.
3. Fine-tune that ckpt 15 epochs on LabTrain+competition (4591 after MultExp dHash filter).
4. Blend Frundles fasc with LabTrain(/FT) fasc alphas; apo Frundles.
5. Zenodo DL_Track_US fasc VGG16-BCE-512 h5 dump (if TF venv succeeds).

## Promote rule

Promote only if MultExp S < 0.13195. Current outside-family best does **not** promote.
No Kaggle. No attention heads.

