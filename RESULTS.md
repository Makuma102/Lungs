# Phantom sanity-check results (superseded; real-CT results are in paper/ and results/msd/)

> These numbers come from synthetic data. The build sandbox could not reach Hugging Face, Kaggle or Zenodo. They show the pipeline works end to end. They are **not** clinical accuracy. Re-run on MSD Task06_Lung or LUNA16 before reporting anything.

**Setup:** SmallUNet with 1,942,323 parameters. Data were 36 phantom volumes (32×128×128, spacing 2.5×2×2 mm), split 24/4/8 at the case level. Training ran for 25 epochs on CPU (~23 s/epoch) with seed 42. The selected checkpoint was the best on validation. Command: `python -m lungseg.train --epochs 25 --out runs/phantom`.

## Test set (8 held-out volumes), mean ± std
| Structure | Dice | IoU | HD95 (mm) | Sensitivity | Precision |
|---|---|---|---|---|---|
| Lung  | 0.999 ± 0.000 | 0.999 ± 0.001 | 0.00 ± 0.00 | 0.999 | 1.000 |
| Tumor | 0.939 ± 0.050 | 0.888 ± 0.087 | 2.06 ± 1.22 | 0.921 | 0.962 |

Detection: lesion-wise sensitivity was 1.00 with 0.0 false positives per scan. Case-level accuracy, sensitivity and specificity were all 1.00.

## 3D reconstruction (test case 0)
`results/case0.obj` holds the `lungs` and `tumor` meshes (39,664 triangles, in mm). `results/case0_report.json` shows lung volume 881 mL and one nodule of 15.6 mL (equivalent diameter 31 mm, band ≥8 mm, mean probability 0.99).

## Limitations
The phantom is much easier than real CT: it has no pleura-attached nodules, ground-glass opacity, motion or scanner variation. Expect far lower tumor Dice on MSD Task06 (published 2D/3D baselines are roughly 0.55–0.70). The lung labels in MSD are derived by thresholding. A 2D model ignores through-plane context, so a 2.5D or 3D U-Net is the natural next step.
