"""Score any external model's NIfTI predictions (e.g. nnU-Net) with OUR metric
code on the native grid, producing a test_eval.json compatible with
compare_runs.py / aggregate_cv.py.

  python scripts/nnunet/score_predictions.py --pred-dir nnunet_preds/ --cases-json splits_final_ourfolds.json --fold 0 \
      --out results/nnunet/fold0/test_eval.json

Predictions: <pred-dir>/<case>.nii.gz, any label > 0 counted as tumor, same grid as the MSD image.
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.evaluate_msd import case_metrics  # noqa: E402  (identical metric definitions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--cases-json", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--root", default="data/Task06_Lung")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = json.load(open(a.cases_json))["test_folds"][a.fold]
    rows = []
    for c in cases:
        ref = nib.load(os.path.join(a.root, "labelsTr", c + ".nii.gz"))
        g = np.asanyarray(ref.dataobj) > 0
        pi = nib.load(os.path.join(a.pred_dir, c + ".nii.gz"))
        assert pi.shape == ref.shape and np.allclose(pi.affine, ref.affine, atol=1e-3), f"{c}: grid mismatch"
        p = np.asanyarray(pi.dataobj) > 0
        rows.append({"case": c, "ours": case_metrics(p, g, ref.header.get_zooms()[:3])})
        print(c, round(rows[-1]["ours"]["dice"], 3), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": {"n_test": len(rows)}, "cases": rows}, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
