# nnU-Net v2 baseline on identical folds (run on a GPU machine)

```bash
pip install nnunetv2
export nnUNet_raw=$PWD/nnUNet_raw nnUNet_preprocessed=$PWD/nnUNet_preprocessed nnUNet_results=$PWD/nnUNet_results
nnUNetv2_convert_MSD_dataset -i data/Task06_Lung            # -> Dataset006_Lung
nnUNetv2_plan_and_preprocess -d 6 --verify_dataset_integrity
python scripts/nnunet/make_splits.py --out $nnUNet_preprocessed/Dataset006_Lung/splits_final.json
for f in 0 1 2 3 4; do nnUNetv2_train 6 3d_fullres $f; done   # ~1-2 GPU-days total
# nnU-Net's own validation predictions of each fold ARE our test folds (val == our test fold):
for f in 0 1 2 3 4; do
  python scripts/nnunet/score_predictions.py --fold $f \
    --pred-dir $nnUNet_results/Dataset006_Lung/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_$f/validation \
    --cases-json $nnUNet_preprocessed/Dataset006_Lung/splits_final_ourfolds.json \
    --out results/nnunet/fold$f/test_eval.json
done
python scripts/aggregate_cv.py --cv-dir results/nnunet --out results/nnunet/cv_summary.json
```
nnU-Net's file names are `lung_XXX.nii.gz` after MSD conversion, matching ours. Note that nnU-Net uses the test
fold as its "validation" set only for its final report (it does not select checkpoints on it), so this is a
fair out-of-fold comparison; its predictions are scored with our metric code (`evaluate_msd.case_metrics`).
