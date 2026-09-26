"""End-to-end anatomy reconstruction for one MSD case.

  python scripts/reconstruct_case.py lung_004 --ckpt runs/msd3d/best.pt

1. (if missing) TotalSegmentator: lobes + trachea (fast) and airway/vessels
2. 3D U-Net tumor prediction on the preprocessed grid, mapped back to the scan
3. anatomy.build(): meshes (OBJ), airway centreline, report
4. copy the web bundle into viewer/models/<case>.json and update cases.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lungseg import anatomy  # noqa: E402
from lungseg.metrics import dice, hd95  # noqa: E402
from lungseg.model import SmallUNet3D  # noqa: E402
from lungseg.prep_msd import to_original_nifti  # noqa: E402
from lungseg.reconstruct3d import postprocess, predict_volume  # noqa: E402

TS_DEVICE = os.environ.get("TS_DEVICE", "cpu")  # set TS_DEVICE=gpu on a CUDA machine
LOBE_ROI = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
            "lung_middle_lobe_right", "lung_lower_lobe_right", "trachea"]


def _lung_crop(ct, anat, margin_mm=15):
    """Bounding box of the lobes (+ trachea) in voxel indices, with a margin."""
    img = nib.load(ct)
    m = np.zeros(img.shape, bool)
    for r in LOBE_ROI:
        p = os.path.join(anat, "total", r + ".nii.gz")
        if os.path.exists(p):
            m |= np.asanyarray(nib.load(p).dataobj) > 0
    nz = np.argwhere(m)
    pad = np.ceil(margin_mm / np.array(img.header.get_zooms()[:3])).astype(int)
    lo = np.maximum(nz.min(0) - pad, 0)
    hi = np.minimum(nz.max(0) + pad + 1, img.shape)
    return img, lo, hi


def totalseg(ct, anat, vessels=True):
    # all six outputs must exist: an interrupted run can leave a partial set
    if not all(os.path.exists(os.path.join(anat, "total", r + ".nii.gz")) for r in LOBE_ROI):
        subprocess.check_call(["TotalSegmentator", "-i", ct, "-o", os.path.join(anat, "total"), "--fast",
                               "--roi_subset", *LOBE_ROI, "-d", TS_DEVICE,
                               # multi-process saving deadlocked / left partial outputs in this container
                               "--nr_thr_resamp", "1", "--nr_thr_saving", "1"])
    out = os.path.join(anat, "vessels")
    if vessels and not (os.path.exists(os.path.join(out, "lung_airways.nii.gz"))
                                or os.path.exists(os.path.join(out, "lung_vessels.nii.gz"))):
        # The full-resolution airway/vessel model needs >6 GB RAM on a whole
        # 512x512xN scan; crop to the lungs first, then paste results back.
        img, lo, hi = _lung_crop(ct, anat)
        crop_ct = os.path.join(anat, "ct_lungcrop.nii.gz")
        nib.save(img.slicer[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], crop_ct)
        tmp = os.path.join(anat, "vessels_crop")
        subprocess.check_call(["TotalSegmentator", "-i", crop_ct, "-o", tmp, "-ta", "lung_vessels",
                               "-d", TS_DEVICE, "--nr_thr_resamp", "1", "--nr_thr_saving", "1"])
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(tmp):
            c = np.asanyarray(nib.load(os.path.join(tmp, f)).dataobj)
            full = np.zeros(img.shape, np.uint8)
            full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = c > 0
            nib.save(nib.Nifti1Image(full, img.affine), os.path.join(out, f))
        shutil.rmtree(tmp)
        os.remove(crop_ct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--root", default="data/Task06_Lung")
    ap.add_argument("--ckpt", default="runs/msd3d/best.pt")
    ap.add_argument("--no-vessels", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    ct = os.path.join(a.root, "imagesTr", a.case + ".nii.gz")
    gt = ct.replace("imagesTr", "labelsTr")
    anat = os.path.join("data/anat", a.case)
    out = os.path.join("data/recon", a.case)
    totalseg(ct, anat, not a.no_vessels)

    pred_path, metrics = None, None
    if os.path.exists(a.ckpt):
        st = torch.load(a.ckpt, weights_only=False)
        args = st["args"]
        model = SmallUNet3D(base=args["base"], anisotropic=not args.get("isotropic", False))
        model.load_state_dict(st["model"])
        npz = os.path.join(args["root"], a.case + ".npz")  # preprocessing this model was trained on
        d = np.load(npz)
        sel_p = "results/msd/postproc_selection.json"
        rule = json.load(open(sel_p))["selected"]["rule"] if os.path.exists(sel_p) else {}
        lab, prob = predict_volume(model, d["img"].astype(np.float32))
        lab = postprocess(lab, tuple(d["spacing"]), tumor_prob=prob, **rule)
        prob = np.where(lab == 2, np.maximum(prob, 0.5), np.minimum(prob, 0.49))  # respect post-processing
        img = to_original_nifti(prob, npz, ct)
        os.makedirs(out, exist_ok=True)
        pred_path = os.path.join(out, "tumor_pred.nii.gz")
        nib.save(img, pred_path)
        g = np.asanyarray(nib.load(gt).dataobj) > 0
        p = np.asanyarray(img.dataobj) > 0
        zooms = nib.load(ct).header.get_zooms()[:3]
        metrics = {"dice": round(float(dice(p, g)), 4), "hd95_mm": round(float(hd95(p, g, zooms)), 2), "postproc": rule,
                   "split": "test" if a.case in [args["cases"][i] for i in args["split"]["test"]] else "train/val"}

    rep = anatomy.build(ct, anat, out, gt, pred_path)
    web = json.load(open(os.path.join(out, "web.json")))
    if metrics:
        web["report"]["pred_metrics"] = metrics
        rep["pred_metrics"] = metrics
        json.dump(rep, open(os.path.join(out, "anatomy.json"), "w"), indent=1)
    os.makedirs("viewer/models", exist_ok=True)
    json.dump(web, open(f"viewer/models/{a.case}.json", "w"))
    cases_path = "viewer/models/cases.json"
    cases = json.load(open(cases_path)) if os.path.exists(cases_path) else []
    cases = [c for c in cases if c["id"] != a.case] + [{"id": a.case, "note": a.note}]
    json.dump(cases, open(cases_path, "w"), indent=1)
    print(json.dumps({k: rep.get(k) for k in ("airway_tree", "tumor_gt", "tumor_pred", "pred_metrics")}, indent=1)[:2000])


if __name__ == "__main__":
    main()
