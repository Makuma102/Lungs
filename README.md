# Lungs — small U-Net lung and tumor segmentation with 3D reconstruction

This is a compact, reproducible pipeline that:

1. **Segments** lungs and tumors/nodules on axial CT slices, or on chest X-rays, with a small 2D U-Net (about 1.9M parameters).
2. **Reconstructs in 3D** by stacking the slice predictions into a volume and cleaning it up with 3D connected components. It then extracts marching-cubes surface meshes in millimetres (`.obj` with objects `lungs` and `tumor`).
3. **Reports** each nodule's volume (mL), equivalent diameter, centroid, a Lung-RADS-style size band, and a scan-level suspicious/clear call.

> ⚠️ This is a research prototype. It is not a medical device and must not be used for diagnosis.

## Layout
| File | Purpose |
|---|---|
| `lungseg/model.py` | `SmallUNet`: 4 levels, base width 16, InstanceNorm + LeakyReLU, Kaiming init |
| `lungseg/losses.py` | Dice + weighted cross-entropy loss |
| `lungseg/metrics.py` | Dice, IoU, HD95 (mm), sensitivity, precision, lesion-wise detection |
| `lungseg/data.py` | HU windowing, augmentation, MSD/PNG loaders, CT and CXR phantoms |
| `lungseg/reconstruct3d.py` | Volume inference with flip TTA, 3D post-processing, meshes, JSON report |
| `lungseg/train.py` | Training and evaluation CLI (case-level splits, OneCycle, checkpointing) |
| `scripts/download_datasets.py` | Downloads the public datasets |
| `tests/test_smoke.py` | Smoke tests for shapes, loss, metrics, 3D export and end-to-end training |

## Public datasets
| Dataset | Modality | Labels | Source |
|---|---|---|---|
| Medical Segmentation Decathlon **Task06_Lung** | CT, 63 labelled cases (NSCLC) | tumor (lung mask derived by HU thresholding) | medicaldecathlon.com, CC-BY-SA 4.0 |
| **kmader/finding-lungs-in-ct-data** | CT slices | lung masks | Kaggle |
| **nikhilpandey360/chest-xray-masks-and-labels** | CXR (Montgomery + Shenzhen) | lung masks, TB labels | Kaggle |
| LUNA16 / LIDC-IDRI | CT | nodules ≥3 mm | luna16.grand-challenge.org / TCIA |
| Any Hugging Face mirror | — | — | `download_datasets.py hf --repo <id>` |

```bash
pip install -r requirements.txt
python scripts/download_datasets.py msd
python -m lungseg.train --data msd --root data/Task06_Lung --size 256 --epochs 100 --out runs/msd
python scripts/download_datasets.py kaggle-cxr   # then arrange as data/cxr/{images,masks}
python -m lungseg.train --data png --root data/cxr --size 256 --out runs/cxr
```

## Quick start (offline)
```bash
pytest -q                                       # smoke tests, ~5 s
python -m lungseg.train --epochs 25 --out runs/phantom
```
This writes `runs/phantom/{best.pt, results.json, case0.obj, case0_report.json, case0_labels.npy}`. The `.obj` file opens in MeshLab, Blender or 3D Slicer.

## Methods (paper-style)
**Preprocessing.** HU is clipped to [−1000, 400] and min-max scaled. Slices are resampled in-plane to 256² for real data and 128² for the phantom. Splits are made at the patient/case level, so slices from one case never leak across splits.
**Network.** 2D U-Net with widths 16-32-64-128-256. Each block is two 3×3 convs with InstanceNorm and LeakyReLU(0.01). The bottleneck uses dropout 0.1, and upsampling uses transposed convolutions. Output is 3 classes (background, lung, tumor).
**Training.** Loss is soft Dice (batch-level, foreground classes) plus cross-entropy with class weights (0.5, 1, 3). The optimizer is AdamW (lr 2e-3, wd 1e-4) with a OneCycle schedule and gradient clipping at 12. Batch size is 16. Tumor-containing slices are oversampled 50%. Augmentations are horizontal flip, ±15° rotation, gamma, contrast and Gaussian noise. The checkpoint with the best mean validation Dice (lung and tumor) is kept.
**Inference and 3D.** Softmax is averaged over the original and flipped image. 3D post-processing keeps the two largest lung components and drops tumor components under 20 mm³ or outside the closed lung hull. Meshes come from marching cubes on a Gaussian-smoothed mask (σ = 0.8 voxel) with physical spacing.
**Evaluation.** Per case: Dice, IoU, HD95 (mm), voxel sensitivity and precision. Tumor metrics are averaged over tumor-positive cases only. Detection is scored per lesion (sensitivity, FP/scan) and per case (accuracy, sensitivity, specificity). Results are reported as mean ± std.

## Results
See `RESULTS.md`. Numbers there come from the synthetic phantom, because this build environment had no network access to Hugging Face or Kaggle. **They are not clinical performance.** Re-run on MSD or LUNA16 before making any claims.

## References
Ronneberger et al., *U-Net*, MICCAI 2015 · Isensee et al., *nnU-Net*, Nat. Methods 2021 · Simpson et al., *Medical Segmentation Decathlon*, arXiv:1902.09063 · Setio et al., *LUNA16*, MedIA 2017 · Jaeger et al., *Montgomery/Shenzhen CXR*, QIMS 2014.
