"""Clinically oriented agreement between predicted and expert tumors.

Per patient (native grid, largest connected component on each side, which is
what a reader would measure):
  * RECIST-style longest axial diameter (mm): the maximum, over axial slices, of
    the maximum Feret diameter of the tumor cross-section.
  * volume (mL).
  * lobe localisation: the TotalSegmentator lobe with the largest overlap with
    the tumor (dilated by 3 mm so tumors abutting the lobe surface still get a
    lobe), for expert and predicted tumor; agreement = same lobe.
Outputs agreement statistics (bias, 95% limits of agreement, mean absolute
error, exact lobe agreement) and per-patient rows.

  python scripts/clinical_measures.py results/cv/fold*/test_eval.json --cache-dirs results/cv/fold*/pred_cache \
      --out results/cv/clinical.json
"""
import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

ROOT = "data/Task06_Lung"
LOBES = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
         "lung_upper_lobe_left", "lung_lower_lobe_left"]


def largest(m):
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    return lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)


def longest_axial_diameter(m, zooms):
    """Max Feret diameter over axial slices, in mm (in-plane spacing assumed square)."""
    best = 0.0
    for z in np.nonzero(m.any((0, 1)))[0]:
        sl = m[:, :, z]
        for rp in regionprops(sl.astype(np.uint8)):
            best = max(best, rp.feret_diameter_max * float(zooms[0]))
    return best


def lobe_of(m, case, zooms):
    if not m.any():
        return None
    md = ndimage.binary_dilation(m, iterations=max(1, int(round(3 / zooms[0]))))
    best, best_ov = None, 0
    for lobe in LOBES:
        f = os.path.join("data/anat", case, "total", lobe + ".nii.gz")
        if not os.path.exists(f):
            return None
        ov = int((md & (np.asanyarray(nib.load(f).dataobj) > 0)).sum())
        if ov > best_ov:
            best, best_ov = lobe, ov
    return best


def agreement(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    return {"n": len(d), "bias": float(d.mean()), "loa": [float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
            "mae": float(np.abs(d).mean()), "median_ae": float(np.median(np.abs(d))), "pearson_r": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--key", default="p_sel", help="cached prediction to use: p (default rule) or p_sel")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cache = {}
    for d in a.cache_dirs:
        for f in glob.glob(os.path.join(d, "*.npz")):
            cache[os.path.basename(f)[:-4]] = f
    rows = []
    for ev in a.evals:
        for r in json.load(open(ev))["cases"]:
            c = r["case"]
            cz = np.load(cache[c]); shp = tuple(cz["shape"])
            k = a.key if a.key in cz else "p"
            p = np.unpackbits(cz[k], count=int(np.prod(shp))).reshape(shp).astype(bool)
            ref = nib.load(os.path.join(ROOT, "labelsTr", c + ".nii.gz"))
            g = np.asanyarray(ref.dataobj) > 0
            z = ref.header.get_zooms()[:3]
            gl, pl = largest(g), largest(p)
            vml = float(np.prod(z)) / 1000
            rows.append({"case": c, "d_gt_mm": longest_axial_diameter(gl, z), "d_pred_mm": longest_axial_diameter(pl, z),
                         "v_gt_ml": float(gl.sum() * vml), "v_pred_ml": float(pl.sum() * vml),
                         "lobe_gt": lobe_of(gl, c, z), "lobe_pred": lobe_of(pl, c, z), "dice": r["ours"]["dice"]})
            print(c, {k2: (round(v, 1) if isinstance(v, float) else v) for k2, v in rows[-1].items() if k2 != "case"}, flush=True)
    found = [r for r in rows if r["d_pred_mm"] > 0]
    lobe_rows = [r for r in rows if r["lobe_gt"] and r["lobe_pred"]]
    out = {"n": len(rows), "n_with_prediction": len(found), "prediction_key": a.key,
           "diameter_all": agreement([r["d_gt_mm"] for r in rows], [r["d_pred_mm"] for r in rows]),
           "diameter_detected": agreement([r["d_gt_mm"] for r in found], [r["d_pred_mm"] for r in found]) if len(found) > 2 else None,
           "within_5mm": float(np.mean([abs(r["d_pred_mm"] - r["d_gt_mm"]) <= 5 for r in rows])),
           "lobe_agreement": float(np.mean([r["lobe_gt"] == r["lobe_pred"] for r in lobe_rows])) if lobe_rows else None,
           "lobe_n": len(lobe_rows), "rows": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
