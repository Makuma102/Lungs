"""Paper evaluation on the held-out MSD Task06 test split, at native resolution.

For every test case the 3D U-Net prediction is mapped back to the original CT
grid and compared with the expert label. We report per-case Dice, HD95 (mm),
voxel sensitivity/precision, lesion detection and volume error, with
bootstrap 95% CIs (2000 resamples over cases), and three controls that expose
false success:
  * EMPTY      - predicts no tumor (lower bound; Dice must be 0)
  * THRESHOLD  - naive intensity baseline: soft tissue (-300..200 HU) inside the
                 lung hull, largest connected component (what a trivial method gets)
  * SHUFFLED   - our predictions scored against a *different* test case's label
                 (a metric that rewards generic blobs would score high here)
Also records the resampling ceiling (expert label -> 1.5 mm grid -> back).

  python scripts/evaluate_msd.py --ckpt runs/msd3d/best.pt --out results/msd
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lungseg.metrics import dice, hd95, lesion_detection, precision, sensitivity  # noqa: E402
from lungseg.model import SmallUNet3D, count_parameters  # noqa: E402
from lungseg.prep_msd import to_original_nifti  # noqa: E402
from lungseg.reconstruct3d import postprocess, predict_volume  # noqa: E402

ROOT = "data/Task06_Lung"


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def case_metrics(p, g, zooms):
    vml = float(np.prod(zooms)) / 1000
    tp, fn, fp = lesion_detection(p, g, spacing=zooms, min_diameter_mm=3.0)
    return {"dice": dice(p, g), "hd95": hd95(p, g, zooms), "sens": sensitivity(p, g), "prec": precision(p, g),
            "tp": tp, "fn": fn, "fp": fp, "vol_gt_ml": g.sum() * vml, "vol_pred_ml": p.sum() * vml,
            "detected": bool((p & g).any())}


def threshold_baseline(hu, zooms):
    from lungseg.data import lungs_from_hu
    lung = np.moveaxis(lungs_from_hu(np.moveaxis(hu, 2, 0)), 0, 2)
    hull = np.stack([ndimage.binary_fill_holes(s) for s in np.moveaxis(ndimage.binary_closing(lung, iterations=3), 2, 0)])
    hull = np.moveaxis(hull, 0, 2)
    cand = hull & ~lung & (hu > -300) & (hu < 200)
    cand = ndimage.binary_opening(cand, iterations=2)
    lab, n = ndimage.label(cand)
    if n == 0:
        return cand
    sizes = ndimage.sum(cand, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def overlay(ax, ct2d, g2d, p2d, title):
    """ct2d/g2d/p2d: the axial slice (x, y) with the largest tumor area."""
    ax.imshow(np.rot90(np.clip(ct2d, -1000, 400)), cmap="gray")
    for m, c in ((g2d, "#ff7a1a"), (p2d, "#2fd4ff")):
        if m.any():
            ax.contour(np.rot90(m), levels=[0.5], colors=c, linewidths=1.2)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/msd3d/best.pt")
    ap.add_argument("--out", default="results/msd")
    ap.add_argument("--rescore", action="store_true", help="reuse cached native-grid predictions")
    ap.add_argument("--max-cases", type=int, default=None, help="smoke-test on the first N test cases")
    ap.add_argument("--postproc", default="results/msd/postproc_selection.json",
                    help="validation-selected rule (scripts/select_postproc.py); evaluated alongside the default")
    a = ap.parse_args()
    sel_rule = json.load(open(a.postproc))["selected"]["rule"] if os.path.exists(a.postproc) else None
    os.makedirs(a.out, exist_ok=True)
    st = torch.load(a.ckpt, weights_only=False)
    args = st["args"]
    model = SmallUNet3D(base=args["base"], anisotropic=not args.get("isotropic", False))
    model.load_state_dict(st["model"])
    test_cases = [args["cases"][i] for i in args["split"]["test"]][:a.max_cases]
    train_cases = [args["cases"][i] for i in args["split"]["train"]]
    assert not set(test_cases) & set(train_cases), "train/test leakage"

    rows, masks = [], {}
    for c in test_cases:
        ct_p = os.path.join(ROOT, "imagesTr", c + ".nii.gz")
        npz = os.path.join("data/prep", c + ".npz")
        d = np.load(npz)
        cache = os.path.join(a.out, "pred_cache", c + ".npz")
        use_cache = a.rescore and os.path.exists(cache)
        raw = prob_raw = None
        if not use_cache:
            raw, prob_raw = predict_volume(model, d["img"].astype(np.float32))

        def native(rule):
            lab = postprocess(raw, tuple(d["spacing"]), tumor_prob=prob_raw, **(rule or {}))
            pr = np.where(lab == 2, np.maximum(prob_raw, 0.5), np.minimum(prob_raw, 0.49))
            return np.asanyarray(to_original_nifti(pr, npz, ct_p).dataobj) > 0
        if use_cache:  # re-scoring without re-running the network (only with --rescore)
            cz = np.load(cache)
            shp = tuple(cz["shape"])
            up = lambda k: np.unpackbits(cz[k], count=int(np.prod(shp))).reshape(shp).astype(bool)
            p, p_sel = up("p"), (up("p_sel") if "p_sel" in cz else None)
        else:
            p = native(None)
            p_sel = native(sel_rule) if sel_rule else None
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez_compressed(cache, shape=np.array(p.shape), p=np.packbits(p),
                                **({"p_sel": np.packbits(p_sel)} if p_sel is not None else {}))
        ct = nib.load(ct_p)
        hu = np.asanyarray(ct.dataobj).astype(np.float32)
        zooms = ct.header.get_zooms()[:3]
        g = np.asanyarray(nib.load(ct_p.replace("imagesTr", "labelsTr")).dataobj) > 0
        ceil = np.asanyarray(to_original_nifti(d["lab"] == 2, npz, ct_p).dataobj) > 0
        r = {"case": c, "ours": case_metrics(p, g, zooms), "empty": case_metrics(np.zeros_like(g), g, zooms),
             **({"ours_sel": case_metrics(p_sel, g, zooms)} if p_sel is not None else {}),
             "threshold": case_metrics(threshold_baseline(hu, zooms), g, zooms),
             "ceiling_dice": dice(ceil, g)}
        rows.append(r)
        # keep memory bounded: bit-packed masks for the shuffled control, one 2D slice for figures
        z = int(np.argmax((g | p).sum((0, 1)))) if (g | p).any() else hu.shape[2] // 2
        masks[c] = {"shape": g.shape, "g": np.packbits(g), "p": np.packbits(p),
                    "slice": (hu[:, :, z].copy(), g[:, :, z].copy(), p[:, :, z].copy())}
        del hu, ceil
        print(c, {k: round(v, 3) for k, v in r["ours"].items() if isinstance(v, float)}, flush=True)

    # shuffled-GT control: score each prediction against the next case's label
    # (only defined when shapes match; otherwise compare in a common 1.5 mm crop)
    shuffled = []
    for i, c in enumerate(test_cases):
        o = test_cases[(i + 1) % len(test_cases)]
        unpack = lambda m, k: np.unpackbits(m[k], count=int(np.prod(m["shape"]))).reshape(m["shape"]).astype(bool)
        p = unpack(masks[c], "p")
        g2 = unpack(masks[o], "g")
        if p.shape == g2.shape:
            shuffled.append(dice(p, g2))
        else:
            s = tuple(slice(0, min(x, y)) for x, y in zip(p.shape, g2.shape))
            shuffled.append(dice(p[s], g2[s]))

    def agg(key):
        out = {}
        for m in ("dice", "hd95", "sens", "prec"):
            out[m] = boot_ci([r[key][m] for r in rows])
        tp = sum(r[key]["tp"] for r in rows); fn = sum(r[key]["fn"] for r in rows); fp = sum(r[key]["fp"] for r in rows)
        out["lesion_sensitivity"] = tp / max(1, tp + fn)
        out["fp_per_scan"] = fp / len(rows)
        out["case_detection_rate"] = float(np.mean([r[key]["detected"] for r in rows]))
        gv = np.array([r[key]["vol_gt_ml"] for r in rows]); pv = np.array([r[key]["vol_pred_ml"] for r in rows])
        out["volume_abs_err_ml"] = boot_ci(np.abs(pv - gv))
        out["volume_pearson_r"] = float(np.corrcoef(gv, pv)[0, 1]) if pv.std() > 0 else float("nan")
        return out

    summary = {
        "n_test": len(rows), "n_train": len(train_cases), "test_cases": test_cases,
        "params": count_parameters(model),
        "ours": agg("ours"), "threshold": agg("threshold"), "empty": agg("empty"),
        **({"ours_sel": agg("ours_sel"), "selected_rule": sel_rule} if sel_rule else {}),
        "shuffled_gt_dice": boot_ci(shuffled),
        "resampling_ceiling_dice": boot_ci([r["ceiling_dice"] for r in rows]),
    }
    json.dump({"summary": summary, "cases": rows}, open(os.path.join(a.out, "test_eval.json"), "w"), indent=1, default=float)

    # ---------------- figures
    order = sorted(rows, key=lambda r: r["ours"]["dice"])
    fig, ax = plt.subplots(figsize=(7, 2.6))
    xs = np.arange(len(order))
    ax.bar(xs - 0.2, [r["ours"]["dice"] for r in order], 0.4, label="3D U-Net (ours)", color="#0f7c8c")
    ax.bar(xs + 0.2, [r["threshold"]["dice"] for r in order], 0.4, label="HU-threshold baseline", color="#b9c4cc")
    ax.plot(xs, [r["ceiling_dice"] for r in order], "k_", ms=10, label="resampling ceiling")
    ax.set_xticks(xs, [r["case"].replace("lung_", "") for r in order], fontsize=7)
    ax.set_ylabel("Tumor Dice"); ax.set_ylim(0, 1); ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_dice_per_case.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    gv = [r["ours"]["vol_gt_ml"] for r in rows]; pv = [r["ours"]["vol_pred_ml"] for r in rows]
    mx = max(max(gv), max(pv)) * 1.1
    ax.plot([0, mx], [0, mx], color="#999", lw=0.8)
    ax.scatter(gv, pv, s=18, color="#0f7c8c")
    ax.set_xlabel("Expert tumor volume (mL)"); ax.set_ylabel("Predicted volume (mL)")
    ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_volume.png"), dpi=200); plt.close(fig)

    picks = [order[-1], order[len(order) // 2], order[0]]
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.6))
    for axx, r, tag in zip(axs, picks, ("best", "median", "worst")):
        overlay(axx, *masks[r["case"]]["slice"], f"{tag}: {r['case']}  Dice {r['ours']['dice']:.2f}")
    fig.suptitle("orange = expert label, cyan = 3D U-Net", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_qualitative.png"), dpi=200); plt.close(fig)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
