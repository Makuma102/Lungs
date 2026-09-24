"""Preprocess MSD Task06_Lung for 3D training.

Per case: load CT (HU) + expert tumor label, derive a lung mask (HU threshold +
largest air components, `data.lungs_from_hu`), crop to the lung bounding box
(+10 mm), resample to isotropic `spacing` mm (linear for CT, nearest for
labels), window HU, and save as data/prep/<case>.npz with arrays
img (float16, [0,1]), lab (uint8: 0 bg, 1 lung, 2 tumor), plus the crop box and
affine so predictions can be mapped back to the original scan.
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage

from .data import lungs_from_hu, window_hu


def prep_case(args):
    img_path, out_dir, spacing = args
    case = os.path.basename(img_path).replace(".nii.gz", "")
    out = os.path.join(out_dir, case + ".npz")
    if os.path.exists(out):
        return case, "cached"
    ct = nib.as_closest_canonical(nib.load(img_path))  # RAS orientation
    lab_img = nib.as_closest_canonical(nib.load(img_path.replace("imagesTr", "labelsTr")))
    hu = np.asanyarray(ct.dataobj).astype(np.float32)  # x, y, z
    tum = np.asanyarray(lab_img.dataobj) > 0
    zooms = np.array(ct.header.get_zooms()[:3], float)
    # lungs_from_hu expects slices on axis 0
    lung = np.moveaxis(lungs_from_hu(np.moveaxis(hu, 2, 0)), 0, 2) | tum
    nz = np.argwhere(lung)
    m = np.ceil(10 / zooms).astype(int)
    lo = np.maximum(nz.min(0) - m, 0)
    hi = np.minimum(nz.max(0) + m + 1, hu.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    lab = np.zeros(hu.shape, np.uint8)
    lab[lung] = 1
    lab[tum] = 2
    f = zooms / spacing
    img_r = ndimage.zoom(window_hu(hu[sl]), f, order=1)
    lab_r = ndimage.zoom(lab[sl], f, order=0)
    # store as [z, y, x] (axial slices first) to match the rest of the codebase
    np.savez_compressed(out, img=np.transpose(img_r, (2, 1, 0)).astype(np.float16),
                        lab=np.transpose(lab_r, (2, 1, 0)), lo=lo, hi=hi, zooms=zooms,
                        spacing=np.array([spacing] * 3), affine=ct.affine, shape=np.array(hu.shape))
    return case, f"{img_r.shape} tumor_vox={int((lab_r == 2).sum())}"


def load_prepped(prep_dir, cases=None):
    files = sorted(glob.glob(os.path.join(prep_dir, "lung_*.npz")))
    if cases is not None:
        files = [f for f in files if os.path.basename(f)[:-4] in cases]
    vols, meta = [], []
    for f in files:
        d = np.load(f)
        vols.append((d["img"].astype(np.float32), d["lab"]))
        meta.append({"case": os.path.basename(f)[:-4], "spacing": tuple(d["spacing"].tolist())})
    return vols, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Task06_Lung")
    ap.add_argument("--out", default="data/prep")
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(a.root, "imagesTr", "lung_*.nii.gz")))
    with ProcessPoolExecutor(a.workers) as ex:
        for case, msg in ex.map(prep_case, [(p, a.out, a.spacing) for p in imgs]):
            print(case, msg, flush=True)


if __name__ == "__main__":
    main()


def to_original_nifti(prob_zyx, npz_path, orig_path):
    """Map a tumor probability map (or binary mask) on the preprocessed grid
    ([z,y,x], isotropic, lung crop, RAS) back to the original scan grid and
    orientation, via trilinear upsampling + 0.5 threshold. Returns a Nifti1Image."""
    d = np.load(npz_path)
    lo, hi, shape = d["lo"], d["hi"], tuple(d["shape"])
    p = np.transpose(prob_zyx.astype(np.float32), (2, 1, 0))  # -> x, y, z
    tgt = hi - lo
    p = (ndimage.zoom(p, tgt / np.array(p.shape), order=1) >= 0.5).astype(np.uint8)
    p = p[:tgt[0], :tgt[1], :tgt[2]]
    p = np.pad(p, [(0, t - s) for t, s in zip(tgt, p.shape)])
    full = np.zeros(shape, np.uint8)
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = p
    orig = nib.load(orig_path)
    canon = nib.Nifti1Image(full, d["affine"])
    xform = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt("RAS"),
                                            nib.orientations.io_orientation(orig.affine))
    out = canon.as_reoriented(xform)
    assert out.shape == orig.shape and np.allclose(out.affine, orig.affine, atol=1e-3)
    return out
