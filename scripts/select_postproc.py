"""Select tumor post-processing on the VALIDATION split only (never the test split).

Candidate rules vary the minimum component volume, a minimum mean tumor
probability per component, and keeping only the largest component. Metrics are
computed on the 1.5 mm grid against the expert label. The chosen rule
(max mean Dice; ties -> fewer FP/scan) is written to
results/msd/postproc_selection.json and later applied unchanged to the test set.

  python scripts/select_postproc.py --ckpt runs/msd3d/best.pt
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lungseg.metrics import dice, lesion_detection  # noqa: E402
from lungseg.model import SmallUNet3D  # noqa: E402
from lungseg.reconstruct3d import postprocess, predict_volume  # noqa: E402

RULES = [dict(min_tumor_mm3=v, min_mean_prob=p, largest_only=l)
         for v, p, l in itertools.product([20, 250, 1000], [0.0, 0.7, 0.85], [False, True])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/msd3d/best.pt")
    ap.add_argument("--out", default="results/msd/postproc_selection.json")
    a = ap.parse_args()
    st = torch.load(a.ckpt, weights_only=False)
    args = st["args"]
    model = SmallUNet3D(base=args["base"], anisotropic=not args.get("isotropic", False))
    model.load_state_dict(st["model"])
    val = [args["cases"][i] for i in args["split"]["val"]]
    test = {args["cases"][i] for i in args["split"]["test"]}
    assert not test & set(val)
    preds = []
    for c in val:
        d = np.load(os.path.join("data/prep", c + ".npz"))
        lab, prob = predict_volume(model, d["img"].astype(np.float32))
        preds.append((c, lab, prob, d["lab"], tuple(d["spacing"])))
        print("predicted", c, flush=True)
    res = []
    for r in RULES:
        ds, tp, fn, fp = [], 0, 0, 0
        for c, lab, prob, gt, sp in preds:
            pp = postprocess(lab, sp, tumor_prob=prob, **r) == 2
            g = gt == 2
            ds.append(dice(pp, g))
            t, n_, f = lesion_detection(pp, g)
            tp += t; fn += n_; fp += f
        res.append({"rule": r, "dice": float(np.mean(ds)), "per_case": [float(x) for x in ds],
                    "lesion_sens": tp / max(1, tp + fn), "fp_per_scan": fp / len(preds)})
        print(r, round(res[-1]["dice"], 3), res[-1]["fp_per_scan"], flush=True)
    best = max(res, key=lambda x: (round(x["dice"], 3), -x["fp_per_scan"]))
    default = next(x for x in res if x["rule"] == dict(min_tumor_mm3=20, min_mean_prob=0.0, largest_only=False))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"val_cases": val, "candidates": res, "selected": best, "default": default}, open(a.out, "w"), indent=1)
    print("selected", best["rule"], "val dice", round(best["dice"], 3), "vs default", round(default["dice"], 3))


if __name__ == "__main__":
    main()
