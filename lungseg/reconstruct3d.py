"""3D reconstruction: slice-wise predictions -> volume -> surface meshes + report.

Steps: (1) run the 2D U-Net over every axial slice, (2) stack into a label
volume, (3) 3D post-processing (keep the 2 largest lung components, remove
tumor specks < min_tumor_mm3, restrict tumor to the lung hull), (4) marching
cubes in physical (mm) space, exported as Wavefront OBJ, and (5) per-nodule
volumetry (mL, equivalent diameter) for a structured report.
"""
import json

import numpy as np
import torch
from scipy import ndimage
from skimage import measure


@torch.no_grad()
def predict_volume(model, vol, batch=16, device="cpu"):
    """vol: [D,H,W] in [0,1]. Returns (labels [D,H,W] uint8, tumor prob [D,H,W])."""
    model.eval()
    probs = []
    for i in range(0, vol.shape[0], batch):
        x = torch.from_numpy(vol[i:i + batch, None]).float().to(device)
        p = model(x).softmax(1)
        p = (p + model(x.flip(-1)).softmax(1).flip(-1)) / 2  # flip TTA
        probs.append(p.cpu().numpy())
    probs = np.concatenate(probs)
    return probs.argmax(1).astype(np.uint8), probs[:, 2]


def postprocess(labels, spacing, min_tumor_mm3=20.0):
    out = np.zeros_like(labels)
    lung = labels >= 1
    lab, n = ndimage.label(lung)
    if n:
        sizes = ndimage.sum(lung, lab, range(1, n + 1))
        lung = np.isin(lab, np.argsort(sizes)[::-1][:2] + 1)
    out[lung] = 1
    tum = (labels == 2) & ndimage.binary_closing(lung, iterations=2)
    vox_mm3 = float(np.prod(spacing))
    lab, n = ndimage.label(tum)
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() * vox_mm3 >= min_tumor_mm3:
            out[comp] = 2
    return out


def mesh(mask, spacing, step=1):
    if mask.sum() < 8:
        return None, None
    m = ndimage.gaussian_filter(mask.astype(np.float32), 0.8)
    verts, faces, _, _ = measure.marching_cubes(np.pad(m, 1), 0.5, spacing=spacing, step_size=step)
    return verts - np.asarray(spacing), faces


def write_obj(path, parts):
    """parts: list of (name, verts, faces). Multiple objects in one OBJ."""
    with open(path, "w") as f:
        off = 1
        for name, v, fc in parts:
            if v is None:
                continue
            f.write(f"o {name}\n")
            f.writelines(f"v {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in v[:, ::-1])  # x y z
            f.writelines(f"f {a + off} {b + off} {c + off}\n" for a, b, c in fc)
            off += len(v)


def nodule_report(labels, spacing, tumor_prob=None):
    vox = float(np.prod(spacing))
    lab, n = ndimage.label(labels == 2)
    nods = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        comp = lab == i
        vol = comp.sum() * vox
        c = ndimage.center_of_mass(comp)
        nods.append({
            "id": i,
            "volume_ml": round(vol / 1000, 4),
            "equiv_diameter_mm": round((6 * vol / np.pi) ** (1 / 3), 2),
            "centroid_zyx_mm": [round(a * s, 1) for a, s in zip(c, spacing)],
            "mean_prob": None if tumor_prob is None else round(float(tumor_prob[comp].mean()), 3),
            # Lung-RADS-like size band (solid nodule, mm) - informational only
            "size_band": "<6mm" if (6 * vol / np.pi) ** (1 / 3) < 6 else "6-8mm" if (6 * vol / np.pi) ** (1 / 3) < 8 else ">=8mm",
        })
    return {
        "lung_volume_ml": round((labels >= 1).sum() * vox / 1000, 2),
        "n_nodules": n,
        "suspicious": bool(n),
        "nodules": nods,
        "disclaimer": "Research prototype, not a medical device. Not for diagnosis.",
    }


def reconstruct(model, vol, spacing, out_prefix, device="cpu"):
    labels, prob = predict_volume(model, vol, device=device)
    labels = postprocess(labels, spacing)
    parts = [("lungs",) + mesh(labels >= 1, spacing), ("tumor",) + mesh(labels == 2, spacing)]
    write_obj(out_prefix + ".obj", parts)
    np.save(out_prefix + "_labels.npy", labels)
    rep = nodule_report(labels, spacing, prob)
    with open(out_prefix + "_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    return labels, rep
