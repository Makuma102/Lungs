"""Export our exact 5-fold patient splits for nnU-Net v2 (splits_final.json),
so an nnU-Net baseline is trained and tested on identical folds.

Our folds: seeded permutation (seed 42) of the sorted MSD case list, split with
np.array_split into 5 test folds (lungseg.train, --n-folds 5). nnU-Net needs
train/val per fold; we give it train = all non-test cases (nnU-Net selects its
own checkpoint by its schedule; it never sees the test fold).

  python scripts/nnunet/make_splits.py --out $nnUNet_preprocessed/Dataset006_Lung/splits_final.json
"""
import argparse
import glob
import json
import os

import numpy as np


def our_folds(prep_dir="data/prep", seed=42, k=5):
    cases = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(prep_dir, "lung_*.npz")))
    idx = np.random.default_rng(seed).permutation(len(cases))
    return cases, [[cases[i] for i in f] for f in np.array_split(idx, k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", default="data/prep")
    a = ap.parse_args()
    cases, folds = our_folds(a.prep)
    splits = [{"train": [c for c in cases if c not in f], "val": f} for f in folds]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(splits, open(a.out, "w"), indent=1)
    json.dump({"test_folds": folds}, open(os.path.splitext(a.out)[0] + "_ourfolds.json", "w"), indent=1)
    print(f"wrote {a.out}: {[len(f) for f in folds]} test cases per fold")


if __name__ == "__main__":
    main()
