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

LOBE_ROI = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
            "lung_middle_lobe_right", "lung_lower_lobe_right", "trachea"]


def totalseg(ct, anat, vessels=True):
    if not os.path.exists(os.path.join(anat, "total", "trachea.nii.gz")):
        subprocess.check_call(["TotalSegmentator", "-i", ct, "-o", os.path.join(anat, "total"), "--fast",
                               "--roi_subset", *LOBE_ROI, "-d", "cpu"])
    if vessels and not os.path.exists(os.path.join(anat, "vessels", "lung_vessels.nii.gz")):
        subprocess.check_call(["TotalSegmentator", "-i", ct, "-o", os.path.join(anat, "vessels"),
                               "-ta", "lung_vessels", "-d", "cpu"])


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
        npz = os.path.join("data/prep", a.case + ".npz")
        d = np.load(npz)
        lab, prob = predict_volume(model, d["img"].astype(np.float32))
        lab = postprocess(lab, tuple(d["spacing"]))
        prob = np.where(lab == 2, np.maximum(prob, 0.5), np.minimum(prob, 0.49))  # respect post-processing
        img = to_original_nifti(prob, npz, ct)
        os.makedirs(out, exist_ok=True)
        pred_path = os.path.join(out, "tumor_pred.nii.gz")
        nib.save(img, pred_path)
        g = np.asanyarray(nib.load(gt).dataobj) > 0
        p = np.asanyarray(img.dataobj) > 0
        zooms = nib.load(ct).header.get_zooms()[:3]
        metrics = {"dice": round(float(dice(p, g)), 4), "hd95_mm": round(float(hd95(p, g, zooms)), 2),
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
