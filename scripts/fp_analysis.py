"""Characterise the largest false-positive tumor components of a model on the
test split (native grid, default post-processing, from the prediction cache).

For each test case where the largest predicted component does NOT touch the
expert tumor (so the largest-component rule would discard the true tumor), we
record its volume, its distance to the expert tumor, and its lateral distance
from the tracheal midline (a crude proxy for a hilar/mediastinal location).
Nothing here is a visual or anatomical verification.

  python scripts/fp_analysis.py results/msd_anat --out results/msd_anat/fp_components.json
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = "data/Task06_Lung"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ev = json.load(open(os.path.join(a.run_dir, "test_eval.json")))
    rows = []
    for r in ev["cases"]:
        c = r["case"]
        cz = np.load(os.path.join(a.run_dir, "pred_cache", c + ".npz"))
        shp = tuple(cz["shape"])
        p = np.unpackbits(cz["p"], count=int(np.prod(shp))).reshape(shp).astype(bool)
        img = nib.load(os.path.join(ROOT, "labelsTr", c + ".nii.gz"))
        g = np.asanyarray(img.dataobj) > 0
        vml = float(np.prod(img.header.get_zooms()[:3])) / 1000
        lab, n = ndimage.label(p)
        if n == 0:
            continue
        sizes = ndimage.sum(p, lab, range(1, n + 1))
        big = int(np.argmax(sizes)) + 1
        comp = lab == big
        if (comp & g).any():
            continue  # largest component is (part of) the true tumor
        c_fp = nib.affines.apply_affine(img.affine, ndimage.center_of_mass(comp))
        c_gt = nib.affines.apply_affine(img.affine, ndimage.center_of_mass(g))
        tr = os.path.join("data/anat", c, "total", "trachea.nii.gz")
        mid_x = None
        if os.path.exists(tr):
            t = np.asanyarray(nib.load(tr).dataobj) > 0
            if t.any():
                mid_x = float(nib.affines.apply_affine(img.affine, ndimage.center_of_mass(t))[0])
        rows.append({"case": c, "fp_volume_ml": round(float(sizes[big - 1]) * vml, 1),
                     "gt_volume_ml": round(float(g.sum()) * vml, 1),
                     "distance_to_tumor_mm": round(float(np.linalg.norm(c_fp - c_gt)), 0),
                     "lateral_offset_from_trachea_mm": None if mid_x is None else round(abs(float(c_fp[0]) - mid_x), 0),
                     "centroid_ras_mm": np.round(c_fp, 0).tolist()})
    out = a.out or os.path.join(a.run_dir, "fp_components.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
