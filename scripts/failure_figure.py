"""Failure-case figure: expert tumor vs raw network output on the 1.5 mm grid.

For a given case, shows the axial and coronal slices through the expert tumor
with contours of the expert label (orange), the predicted tumor (cyan) and the
threshold-derived lung label (thin blue), plus the composition of the raw
prediction inside the expert tumor (background / lung / tumor fractions).

  python scripts/failure_figure.py lung_028 --out paper/fig/fig_failure.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lungseg.model import SmallUNet3D  # noqa: E402
from lungseg.reconstruct3d import predict_volume  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--ckpt", default="runs/msd3d/best.pt")
    ap.add_argument("--out", default="paper/fig/fig_failure.png")
    a = ap.parse_args()
    st = torch.load(a.ckpt, weights_only=False)
    m = SmallUNet3D(base=st["args"]["base"], anisotropic=not st["args"].get("isotropic", False))
    m.load_state_dict(st["model"])
    d = np.load(os.path.join("data/prep", a.case + ".npz"))
    img, lab = d["img"].astype(np.float32), d["lab"]
    raw, _ = predict_volume(m, img)
    gt = lab == 2
    comp = np.bincount(raw[gt], minlength=3) / max(1, gt.sum())
    stats = {"case": a.case, "gt_ml": float(gt.sum() * np.prod(d["spacing"]) / 1000),
             "inside_gt_frac": {"background": float(comp[0]), "lung": float(comp[1]), "tumor": float(comp[2])},
             "mean_window_intensity_in_gt": float(img[gt].mean())}
    z = int(np.argmax(gt.sum((1, 2))))
    y = int(np.argmax(gt.sum((0, 2))))
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.3))
    views = [(img[z][::-1], gt[z][::-1], (raw[z] == 2)[::-1], (lab[z] == 1)[::-1], "axial"),
             (img[:, y, :][::-1], gt[:, y, :][::-1], (raw[:, y, :] == 2)[::-1], (lab[:, y, :] == 1)[::-1], "coronal")]
    for axx, (im, g, p, l, t) in zip(ax, views):
        axx.imshow(im, cmap="gray", vmin=0, vmax=1)
        for msk, col, lw in ((l, "#4a7bd1", 0.5), (g, "#ff7a1a", 1.3), (p, "#2fd4ff", 1.3)):
            if msk.any():
                axx.contour(msk.astype(float), [0.5], colors=col, linewidths=lw)
        axx.set_title(t, fontsize=8)
        axx.axis("off")
    fig.suptitle(f"{a.case}: expert tumor {stats['gt_ml']:.0f} mL (orange), predicted tumor (cyan), lung label (blue). "
                 f"Inside the expert tumor the network predicted {100 * comp[0]:.0f}% background, "
                 f"{100 * comp[1]:.0f}% lung, {100 * comp[2]:.0f}% tumor.", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    json.dump(stats, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
