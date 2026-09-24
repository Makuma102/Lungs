"""Anatomy-level 3D reconstruction from a real chest CT.

Combines:
  * lobes (5) and trachea          - TotalSegmentator `total` task (Wasserthal et al., Radiol AI 2023)
  * bronchial tree + lung vessels  - TotalSegmentator `lung_vessels` task
  * tumor                          - MSD Task06 expert label and/or our U-Net prediction
and derives:
  * airway centreline (3D skeleton), branch points, terminal endpoints, and
    per-branch generation counting from the carina (Weibel-style)
  * per-structure volumes (mL) and meshes in scanner (RAS) millimetres.

Output bundle (for viewer/ or 3D Slicer): <out>/<structure>.obj, <out>/centerline.json, <out>/anatomy.json
"""
import base64
import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize

LOBES = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
         "lung_upper_lobe_left", "lung_lower_lobe_left"]


def _load_mask(path, ref_shape):
    if not os.path.exists(path):
        return None
    m = np.asanyarray(nib.load(path).dataobj) > 0
    return m if m.shape == ref_shape else None


def load_case(ct_path, anat_dir, tumor_path=None, pred_path=None):
    """Returns dict of boolean masks (voxel space) plus the affine."""
    img = nib.load(ct_path)
    shape, aff = img.shape, img.affine
    masks = {}
    for lobe in LOBES:
        masks[lobe] = _load_mask(os.path.join(anat_dir, "total", lobe + ".nii.gz"), shape)
    masks["trachea"] = _load_mask(os.path.join(anat_dir, "total", "trachea.nii.gz"), shape)
    masks["bronchi"] = _load_mask(os.path.join(anat_dir, "vessels", "lung_trachea_bronchia.nii.gz"), shape)
    masks["vessels"] = _load_mask(os.path.join(anat_dir, "vessels", "lung_vessels.nii.gz"), shape)
    if tumor_path:
        masks["tumor_gt"] = _load_mask(tumor_path, shape)
    if pred_path:
        masks["tumor_pred"] = _load_mask(pred_path, shape)
    return {k: v for k, v in masks.items() if v is not None and v.any()}, aff, img.header.get_zooms()[:3]


def airway_mask(masks):
    """Prefer the full-resolution airway model (its class includes the trachea);
    fall back to the coarse (3 mm) trachea from the fast `total` model."""
    a = np.zeros_like(next(iter(masks.values())))
    for k in (("bronchi",) if "bronchi" in masks else ("trachea",)):
        a |= masks[k]
    if not a.any():
        return a
    lab, n = ndimage.label(a)  # keep the main connected tree
    sizes = ndimage.sum(a, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


_OFFS = [np.array(o) - 1 for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]


def _prune_spurs(sk, min_len_vox, rounds=3):
    """Remove terminal skeleton branches shorter than min_len_vox (surface noise)."""
    sk = sk.copy()
    for _ in range(rounds):
        pts = {tuple(p) for p in np.argwhere(sk)}
        nb = lambda p: [q for o in _OFFS if (q := (p[0] + o[0], p[1] + o[1], p[2] + o[2])) in pts]
        removed = False
        for e in [p for p in pts if len(nb(p)) == 1]:
            path, prev, cur = [e], None, e
            while True:
                nxt = [q for q in nb(cur) if q != prev and q not in path]
                if len(nxt) != 1 or len(nb(nxt[0])) >= 3:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
                if len(path) > min_len_vox:
                    break
            if len(path) <= min_len_vox:
                for q in path:
                    sk[q] = False
                removed = True
        if not removed:
            break
    return sk


def centerline(airway, spacing, affine, min_branch_mm=5.0):
    """3D skeleton of the airway. Returns graph summary in RAS mm:
    points, edges, branch points, endpoints, and generation per endpoint."""
    sk = _prune_spurs(skeletonize(airway), int(np.ceil(min_branch_mm / min(spacing))))
    pts = np.argwhere(sk)
    if len(pts) == 0:
        return {"points": [], "edges": [], "endpoints": [], "branchpoints": []}
    idx = {tuple(p): i for i, p in enumerate(pts)}
    nbrs = [[idx[t] for o in _OFFS if (t := tuple(p + o)) in idx] for p in pts]
    deg = np.array([len(n) for n in nbrs])
    edges = sorted({(min(i, j), max(i, j)) for i, ns in enumerate(nbrs) for j in ns})
    # root = highest (most superior) skeleton point in the trachea
    ras = nib.affines.apply_affine(affine, pts)
    root = int(np.argmax(ras[:, 2]))
    # BFS for generation: +1 each time we pass a branch point (deg >= 3)
    gen = -np.ones(len(pts), int)
    gen[root] = 0
    q = [root]
    while q:
        i = q.pop(0)
        for j in nbrs[i]:
            if gen[j] < 0:
                gen[j] = gen[i] + (1 if deg[i] >= 3 else 0)
                q.append(j)
    ends = [int(i) for i in np.nonzero(deg == 1)[0] if i != root]
    bps = [int(i) for i in np.nonzero(deg >= 3)[0]]
    return {
        "points": np.round(ras, 1).tolist(),
        "edges": [list(e) for e in edges],
        "endpoints": ends,
        "branchpoints": bps,
        "generation": gen.tolist(),
        "max_generation": int(gen[ends].max()) if ends else 0,
        "length_mm": float(sum(np.linalg.norm(ras[a] - ras[b]) for a, b in edges)),
    }


def taubin(verts, faces, iters=10, lam=0.5, mu=-0.53):
    """Taubin lambda|mu smoothing: removes staircase artefacts without the
    shrinkage of plain Laplacian smoothing (Taubin, SIGGRAPH 1995)."""
    from scipy import sparse
    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    W = sparse.diags(1.0 / deg) @ A
    v = verts.astype(np.float64)
    for _ in range(iters):
        v = v + lam * (W @ v - v)
        v = v + mu * (W @ v - v)
    return v


def mesh_ras(mask, affine, step=1, sigma_mm=1.0, smooth_iters=10):
    """Marching cubes on a mask smoothed with a Gaussian of `sigma_mm` (physical
    units, so coarse and fine inputs look alike), mapped voxel->RAS mm, then
    Taubin-smoothed."""
    sl = ndimage.find_objects(mask.astype(np.uint8))[0]
    sub = np.pad(mask[sl], 3)
    origin = np.array([s.start for s in sl]) - 3
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
    f = ndimage.gaussian_filter(sub.astype(np.float32), sigma_mm / zooms)
    verts, faces, _, _ = measure.marching_cubes(f, 0.5, step_size=step)
    verts = nib.affines.apply_affine(affine, verts + origin)
    if smooth_iters:
        verts = taubin(verts, faces, smooth_iters)
    return verts, faces


def write_obj(path, verts, faces, name):
    with open(path, "w") as f:
        f.write(f"o {name}\n")
        f.write("".join(f"v {a:.1f} {b:.1f} {c:.1f}\n" for a, b, c in verts))
        f.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces))


# triangle budget per structure for the web viewer (quadric decimation)
WEB_TRIS = {"airway": 80000, "vessels": 160000, "tumor_gt": 15000, "tumor_pred": 15000}


def decimate(verts, faces, target):
    if len(faces) <= target:
        return verts, faces
    import pyfqmr
    s = pyfqmr.Simplify()
    s.setMesh(verts.astype(np.float64), faces.astype(np.int32))
    s.simplify_mesh(target_count=target, aggressiveness=5, preserve_border=True, verbose=0)
    v, f, _ = s.getMesh()
    return v, f


def web_mesh(verts, faces):
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    return {"v": b64(verts.astype(np.float32)), "f": b64(faces.astype(np.uint32)), "nv": int(len(verts)), "nf": int(len(faces))}


# mesh resolution per structure: fine for thin airways/vessels, coarse for big lobes
STEP = {"trachea": 1, "bronchi": 1, "airway": 1, "vessels": 1, "tumor_gt": 1, "tumor_pred": 1}
# smoothing kernel per structure (mm): lobes come from a 3 mm model, vessels are thin
SIGMA_MM = {"airway": 0.8, "vessels": 0.5, "tumor_gt": 0.7, "tumor_pred": 0.7}


def build(ct_path, anat_dir, out_dir, tumor_path=None, pred_path=None):
    os.makedirs(out_dir, exist_ok=True)
    masks, aff, zooms = load_case(ct_path, anat_dir, tumor_path, pred_path)
    vox_ml = float(np.prod(zooms)) / 1000
    report = {"case": os.path.basename(ct_path), "spacing_mm": [float(z) for z in zooms], "structures": {}}
    aw = airway_mask(masks)
    coarse_airway = "bronchi" not in masks
    if aw.any():
        masks = {k: v for k, v in masks.items() if k not in ("trachea", "bronchi")}
        masks["airway"] = aw
    web = {}
    for name, m in masks.items():
        step = STEP.get(name, 2)
        sig = 2.0 if (name == "airway" and coarse_airway) else SIGMA_MM.get(name, 2.0)
        v, f = mesh_ras(m, aff, step=step, sigma_mm=sig)
        write_obj(os.path.join(out_dir, name + ".obj"), v, f, name)
        report["structures"][name] = {"volume_ml": round(m.sum() * vox_ml, 2), "triangles": int(len(f))}
        web[name] = web_mesh(*decimate(v, f, WEB_TRIS.get(name, 25000)))
    if aw.any():
        cl = centerline(aw, zooms, aff)
        json.dump(cl, open(os.path.join(out_dir, "centerline.json"), "w"))
        report["airway_tree"] = {k: cl[k] for k in ("max_generation", "length_mm")}
        report["airway_tree"].update(n_endpoints=len(cl["endpoints"]), n_branchpoints=len(cl["branchpoints"]))
    for t in ("tumor_gt", "tumor_pred"):
        if t in masks:
            lab, n = ndimage.label(masks[t])
            lobe_hits = {}
            for lobe in LOBES:
                if lobe in masks:
                    ov = int((masks[t] & masks[lobe]).sum())
                    if ov:
                        lobe_hits[lobe] = ov
            nods = []
            for i in range(1, n + 1):
                comp = lab == i
                vol = comp.sum() * vox_ml * 1000
                c = nib.affines.apply_affine(aff, np.array(ndimage.center_of_mass(comp)))
                nods.append({"volume_ml": round(vol / 1000, 3), "equiv_diameter_mm": round((6 * vol / np.pi) ** (1 / 3), 1),
                             "centroid_ras_mm": np.round(c, 1).tolist()})
            report[t] = {"n_lesions": n, "lesions": sorted(nods, key=lambda d: -d["volume_ml"]),
                         "lobe": max(lobe_hits, key=lobe_hits.get) if lobe_hits else "outside lobes"}
    report["disclaimer"] = "Research prototype, not a medical device. Not for diagnosis."
    json.dump(report, open(os.path.join(out_dir, "anatomy.json"), "w"), indent=1)
    cl_web = None
    if aw.any():  # centreline for the viewer: points + edges + endpoints
        cl_web = {k: cl[k] for k in ("points", "edges", "endpoints", "branchpoints")}
    json.dump({"report": report, "meshes": web, "centerline": cl_web},
              open(os.path.join(out_dir, "web.json"), "w"))
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True)
    ap.add_argument("--anat", required=True, help="dir with total/ and vessels/ TotalSegmentator outputs")
    ap.add_argument("--tumor", help="tumor label NIfTI (e.g. MSD labelsTr)")
    ap.add_argument("--pred", help="predicted tumor NIfTI from our U-Net")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(build(a.ct, a.anat, a.out, a.tumor, a.pred), indent=1))
