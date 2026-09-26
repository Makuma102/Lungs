# Running the project on your own PC

Tested design: Linux or Windows + WSL2, NVIDIA RTX 50-series (e.g. RTX 5070, 12 GB), Python 3.10–3.12.

## 1. Get the code
```bash
git clone https://github.com/makuma102/Lungs.git
cd Lungs
git checkout claude/modest-feynman-8ogcl4
```
On **Windows**, use **WSL2 (Ubuntu)**. The pipeline uses bash scripts, and TotalSegmentator/nnU-Net are best supported on Linux. Install the NVIDIA Windows driver; WSL2 sees the GPU automatically.

## 2. Environment
```bash
python -m venv .venv && source .venv/bin/activate
# RTX 50-series needs a CUDA 12.8+ PyTorch build:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
playwright install chromium            # paper PDF + figure rendering
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q                              # all tests should pass (a few skip until data is downloaded)
```
For the .docx paper you also need Node.js (≥18). Then run `cd paper/docx && npm install && cd ../..`.

## 3. Run everything (one command, resumable)
```bash
scripts/run_all_gpu.sh
```
This downloads MSD Task06 (9.2 GB) and preprocesses it. It then runs TotalSegmentator on all 63 scans, the 5-fold cross-validation, the anatomical-lung ablation, and the showcase reconstructions with airways and vessels, and finally rebuilds `paper/journal/manuscript.docx` and `.pdf`.
Every step skips work that is already done, so you can stop and re-run any time. Training resumes from `runs/*/last.pt`.
Rough GPU time: about 1–2 h for the cross-validation, plus TotalSegmentator and evaluation.

## 4. What is already in the repo
| Path | Content |
|---|---|
| `runs/msd3d/best.pt` | trained v1 model (development split = CV fold 0) |
| `runs/msd3d_anat/best.pt` | trained v2 model (anatomical lung class) |
| `runs/cv/fold0/` | fold 0 of the CV (identical to the development run) |
| `results/` | all evaluation JSONs used by the paper (development split, ablation, comparison) |
| `paper/journal/` | current manuscript draft (development-split numbers) |
| `viewer/` | interactive 3D viewer (open `viewer/index.html` through a local web server) |

Raw data (`data/`) and prediction caches are **not** in git. `run_all_gpu.sh` recreates them.

## 5. Useful single steps
```bash
# view the 3D viewer locally
python -m http.server -d viewer 8000     # then open http://localhost:8000
# train one model
python -m lungseg.train --data msd3d --root data/prep --dim 3 --isotropic --patch 64 96 96 --base 12 --batch 2 --epochs 20 \
  --samples-per-volume 8 --tumor-oversample 0.66 --lr 3e-3 --n-train 45 --n-val 5 --n-test 13 --val-every 4 --resume --out runs/my_run
# rebuild the paper from whatever results exist
python scripts/build_journal.py
```

## 6. Next experiments for a Q1 submission (GPU)
1. **nnU-Net baseline on identical folds.** See `scripts/nnunet/README.md` (≈8–10 h per fold with the 250-epoch trainer on a 5070).
2. **Longer and larger training of our model** (e.g. `--epochs 200`, `--base 16/24`) for a compute–accuracy curve.
3. **External validation** on NSCLC-Radiomics (TCIA) or LIDC-IDRI: download locally, convert to NIfTI, and score with `scripts/nnunet/score_predictions.py`-style evaluation.
4. **Full anatomy for all 63 patients** (`TS_DEVICE=gpu python scripts/reconstruct_case.py <case>`), for population-level anatomy statistics and expert review.

## Notes
- **Determinism:** fixed seeds; the development run and CV fold 0 match exactly on CPU. GPU runs use mixed precision and cuDNN autotuning, so numbers will differ slightly from the CPU results (report them as a separate run).
- **LibreOffice** is optional. The PDF is rendered from the same content with Chromium.
