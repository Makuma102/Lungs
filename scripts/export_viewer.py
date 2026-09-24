"""Bundle meshes + reports for the web CAD viewer (viewer/index.html).

Writes results/phantom_viewer/{gt,unet2d,unet3d}.obj and meta.json for
test case 0 of the phantom split, so predictions can be compared to ground truth.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lungseg.model import SmallUNet, SmallUNet3D  # noqa: E402
from lungseg.reconstruct3d import mesh, nodule_report, postprocess, predict_volume, write_obj  # noqa: E402
from lungseg.train import load  # noqa: E402
from lungseg.metrics import dice, hd95  # noqa: E402


def main(out="results/phantom_viewer"):
    os.makedirs(out, exist_ok=True)
    meta = {}
    runs = {"unet2d": ("runs/phantom", SmallUNet), "unet3d": ("runs/phantom3d", SmallUNet3D)}
    gt_done = False
    for name, (run, cls) in runs.items():
        ck = os.path.join(run, "best.pt")
        if not os.path.exists(ck):
            continue
        state = torch.load(ck)
        a = state["args"]
        model = cls(base=a["base"])
        model.load_state_dict(state["model"])
        ns = type("A", (), a)
        _, _, (tev, tes) = load(ns)
        vol, gt = tev[0]
        sp = tes[0]
        if not gt_done:
            write_obj(os.path.join(out, "gt.obj"), [("lungs",) + mesh(gt >= 1, sp), ("tumor",) + mesh(gt == 2, sp)])
            meta["gt"] = nodule_report(gt, sp)
            meta["spacing_mm"] = [float(s) for s in sp]
            meta["shape"] = list(vol.shape)
            gt_done = True
        labels, prob = predict_volume(model, vol)
        labels = postprocess(labels, sp)
        write_obj(os.path.join(out, f"{name}.obj"),
                  [("lungs",) + mesh(labels >= 1, sp), ("tumor",) + mesh(labels == 2, sp)])
        rep = nodule_report(labels, sp, prob)
        rep["case_metrics"] = {
            "lung_dice": round(dice(labels >= 1, gt >= 1), 4),
            "tumor_dice": round(dice(labels == 2, gt == 2), 4),
            "tumor_hd95_mm": round(hd95(labels == 2, gt == 2, sp), 2),
        }
        test = json.load(open(os.path.join(run, "results.json")))
        rep["params"] = test["params"]
        rep["test_summary"] = {k: (v["mean"] if isinstance(v, dict) else v) for k, v in test["test"].items()}
        meta[name] = rep
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v.get("case_metrics") for k, v in meta.items() if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
