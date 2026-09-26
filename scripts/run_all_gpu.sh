#!/bin/bash
# End-to-end reproduction on a CUDA machine (tested design: RTX 50-series, 12 GB).
# Every step is resumable: finished outputs are skipped on re-run.
set -eu
export TS_DEVICE=${TS_DEVICE:-gpu}
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))"

# 1. data
[ -d data/Task06_Lung/labelsTr ] || python scripts/download_datasets.py msd
# 2. preprocessing (threshold lung class = v1)
python -m lungseg.prep_msd --workers 4
# 3. anatomy for all 63 cases (lobes + trachea), then anatomical-lung preprocessing (v2)
scripts/lobes_all.sh
python -m lungseg.prep_msd --lung-source anat --out data/prep_anat --workers 4
# 4. 5-fold cross-validation (v1) - the paper's primary result
scripts/run_cv.sh
python scripts/aggregate_cv.py --out results/cv/cv_summary.json
python scripts/clinical_measures.py results/cv/fold*/test_eval.json --cache-dirs results/cv/fold*/pred_cache --out results/cv/clinical.json
# 5. development-split ablation (v2, anatomical lung class) + paired comparison
[ -f runs/msd3d_anat/results.json ] || python -m lungseg.train --data msd3d --root data/prep_anat --dim 3 --isotropic \
   --patch 64 96 96 --base 12 --batch 2 --epochs 20 --samples-per-volume 8 --tumor-oversample 0.66 --lr 3e-3 \
   --n-train 45 --n-val 5 --n-test 13 --val-every 4 --resume --out runs/msd3d_anat
mkdir -p results/msd_anat
[ -f results/msd_anat/postproc_selection.json ] || python scripts/select_postproc.py --ckpt runs/msd3d_anat/best.pt --out results/msd_anat/postproc_selection.json
[ -f results/msd_anat/test_eval.json ] || python scripts/evaluate_msd.py --ckpt runs/msd3d_anat/best.pt --out results/msd_anat --postproc results/msd_anat/postproc_selection.json
python scripts/compare_runs.py results/msd/test_eval.json results/msd_anat/test_eval.json --names v1 v2 --out results/compare_v1_v2.json
python scripts/fp_analysis.py results/msd_anat; python scripts/fp_analysis.py results/msd
# 6. showcase reconstructions WITH airways + vessels (fast on GPU)
for c in lung_014 lung_075; do python scripts/reconstruct_case.py $c --ckpt runs/msd3d/best.pt --note "held-out test"; done
# 7. manuscript (docx + pdf), tests
python -m pytest -q tests
python scripts/build_journal.py
echo "ALL DONE -> paper/journal/manuscript.docx"
