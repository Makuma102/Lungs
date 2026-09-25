# Lungs: small 3D U-Net lung tumor segmentation and anatomy-level 3D reconstruction

This is a reproducible pipeline for real chest CT. It does three things:

1. **Tumor segmentation.** A small 3D U-Net (3.2M parameters, trainable on a CPU) segments lung tumors. It is trained and tested on real patient CT from the Medical Segmentation Decathlon.
2. **Anatomy-level 3D reconstruction.** For each scan it builds:
   - the five lung lobes;
   - the airway tree, with a pruned centreline, bifurcations, endpoints and branch generations;
   - the pulmonary arteries and veins;
   - the tumor, both the expert label and our prediction.

   Meshes are exported in patient RAS millimetres as OBJ files (open them in 3D Slicer or Blender) and as a web bundle for the included viewer.
3. **Evaluation built to catch false success.** Scores are measured at native resolution with bootstrap confidence intervals. Results are compared against a naive threshold baseline, an empty predictor, a shuffled-label control and a resampling ceiling. A test suite checks the metrics, splits, geometry and inference.

The paper in `paper/` (HTML + PDF) is generated from the result files, so no number in it is typed by hand.

> ⚠️ This is a research prototype. It is not a medical device, and it does not diagnose cancer or tell malignant from benign nodules.

## Data (public)
| Dataset | Use | Access |
|---|---|---|
| **MSD Task06_Lung** (63 labelled CTs, NSCLC, CC-BY-SA 4.0) | tumor training and testing | `python scripts/download_datasets.py msd` |
| **TotalSegmentator** models (Wasserthal et al. 2023) | lobes, trachea, airways, arteries, veins | `pip install totalsegmentator` (weights come from GitHub releases) |
| Kaggle `kmader/finding-lungs-in-ct-data`, `nikhilpandey360/chest-xray-masks-and-labels` | optional 2D lung-field models (CT slice / CXR) | `download_datasets.py kaggle-ct / kaggle-cxr` |
| Any Hugging Face mirror | — | `download_datasets.py hf --repo <id>` |

## Pipeline
```bash
pip install -r requirements.txt totalsegmentator pyfqmr playwright
python scripts/download_datasets.py msd                      # 9.2 GB
python -m lungseg.prep_msd                                   # crop to lungs, 1.5 mm isotropic, labels bg/lung/tumor
python -m lungseg.train --data msd3d --root data/prep --dim 3 --isotropic --patch 64 96 96 \
    --base 12 --batch 2 --epochs 20 --samples-per-volume 8 --tumor-oversample 0.66 --lr 3e-3 \
    --n-train 45 --n-val 5 --n-test 13 --val-every 4 --out runs/msd3d
python scripts/evaluate_msd.py --ckpt runs/msd3d/best.pt     # test metrics, controls, figures
python scripts/reconstruct_case.py lung_004 --ckpt runs/msd3d/best.pt   # anatomy + viewer bundle
python scripts/build_paper.py --run-tests                    # paper/paper.html + paper.pdf
pytest -q                                                     # smoke + validity tests
```

## Layout
| Path | Purpose |
|---|---|
| `lungseg/model.py` | `SmallUNet` (2D) and `SmallUNet3D` (isotropic or anisotropic pooling) |
| `lungseg/losses.py` | Dice + weighted cross-entropy loss (2D/3D) |
| `lungseg/metrics.py` | Dice, IoU, HD95 in mm, sensitivity, precision, lesion detection |
| `lungseg/data.py` | HU windowing, 2D slice and 3D patch samplers with tumor oversampling, phantom |
| `lungseg/prep_msd.py` | MSD preprocessing and exact mapping back to the native grid |
| `lungseg/reconstruct3d.py` | Sliding-window 3D inference (Gaussian weighting, flip TTA), post-processing |
| `lungseg/anatomy.py` | Lobes/airway/vessels/tumor fusion, airway centreline graph, Taubin-smoothed meshes |
| `lungseg/train.py` | Training/evaluation CLI (case-level splits, one-cycle LR, checkpointing) |
| `scripts/` | download, evaluate, reconstruct, render viewer, build paper |
| `viewer/index.html` | three.js 3D viewer (lobes, airway + centreline + endpoints, vessels, tumor) |
| `tests/` | `test_smoke.py` (does it run) and `test_validity.py` (can the result be trusted) |

## False-success checks (`tests/test_validity.py`)
- Metric edge cases: an empty prediction gives Dice 0 and HD95 ∞ (never NaN or 0). Tumor-free cases don't inflate tumor Dice. An all-foreground prediction is penalised. HD95 scales with voxel spacing.
- Negative controls: an untrained network and a constant background predictor score about 0. Scoring against another case's label gives low Dice.
- Leakage: splits are disjoint and deterministic, and slices never mix volumes.
- Geometry: sliding-window inference exactly matches whole-volume inference. Nodule volume is correct under anisotropic spacing. Post-processing removes specks. Tumor mesh centroids match label centroids in RAS within 3 mm. The native-grid round trip keeps Dice above 0.85.

## Results
Held-out test split (13 patients), native resolution: tumor Dice 0.61 [0.45, 0.74] (default post-processing) and 0.66 [0.49, 0.79] with the validation-selected largest-component rule (0.08 FP/scan). Ablation: an anatomical lung class (v2) improved validation Dice but *reduced* test Dice (paired difference -0.07 [-0.18, +0.04]); v1 remains the main model. Full details in `paper/paper.pdf`, `results/msd/test_eval.json`, `results/compare_v1_v2.json`. The earlier synthetic-phantom results (`RESULTS.md`) only served as a code sanity check.
