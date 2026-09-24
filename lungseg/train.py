"""Train + evaluate the small U-Net.

  python -m lungseg.train --data phantom --epochs 30          # 2D, runs offline
  python -m lungseg.train --data phantom --dim 3 --batch 2    # 3D U-Net
  python -m lungseg.train --data msd --root data/Task06_Lung  # real CT tumors
  python -m lungseg.train --data png --root data/cxr          # CXR lung masks
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from .losses import DiceCELoss
from .metrics import dice, hd95, iou, lesion_detection, precision, sensitivity
from .model import SmallUNet, SmallUNet3D, count_parameters
from .reconstruct3d import postprocess, predict_volume, reconstruct


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def load(args):
    if args.data == "phantom":
        shape = (args.depth_slices, args.size, args.size)
        vols = D.make_phantom_volumes(args.n_train + args.n_val + args.n_test, shape, seed=args.seed)
        spacings = [(2.5, 1.0 * 256 / args.size, 1.0 * 256 / args.size)] * len(vols)
    elif args.data == "msd3d":  # preprocessed by `python -m lungseg.prep_msd`
        from .prep_msd import load_prepped
        vols, meta = load_prepped(args.root)
        spacings = [m["spacing"] for m in meta]
        args.cases = [m["case"] for m in meta]
    elif args.data == "msd":
        vols, spacings = D.load_msd_lung(args.root, args.size, args.max_cases)
    else:
        vols = D.load_png_pairs(os.path.join(args.root, "images"), os.path.join(args.root, "masks"), args.size)
        spacings = [(1.0, 1.0, 1.0)] * len(vols)
    idx = np.random.default_rng(args.seed).permutation(len(vols))  # case-level split: no slice leakage
    n_te = max(1, round(len(vols) * args.n_test / (args.n_train + args.n_val + args.n_test)))
    n_va = max(1, round(len(vols) * args.n_val / (args.n_train + args.n_val + args.n_test)))
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
    args.split = {"train": [int(i) for i in tr], "val": [int(i) for i in va], "test": [int(i) for i in te]}
    pick = lambda ii: ([vols[i] for i in ii], [spacings[i] for i in ii])
    return pick(tr), pick(va), pick(te)


def evaluate(model, vols, spacings, device):
    rows = []
    for (img, gt), sp in zip(vols, spacings):
        pred, _ = predict_volume(model, img, device=device)
        pred = postprocess(pred, sp) if img.shape[0] > 1 else pred
        r = {}
        for name, cls in (("lung", lambda a: a >= 1), ("tumor", lambda a: a == 2)):
            p, g = cls(pred), cls(gt)
            r[f"{name}_dice"] = dice(p, g)
            r[f"{name}_iou"] = iou(p, g)
            r[f"{name}_hd95"] = hd95(p, g, sp)
            r[f"{name}_sens"] = sensitivity(p, g)
            r[f"{name}_prec"] = precision(p, g)
        r["tp"], r["fn"], r["fp"] = lesion_detection(pred == 2, gt == 2)
        r["has_tumor_gt"], r["has_tumor_pred"] = bool((gt == 2).any()), bool((pred == 2).any())
        rows.append(r)
    return rows


def summarize(rows):
    out = {}
    for k in rows[0]:
        if k in ("tp", "fn", "fp", "has_tumor_gt", "has_tumor_pred"):
            continue
        v = np.array([r[k] for r in rows], float)
        if k.startswith("tumor"):  # tumor segmentation metrics over tumor-positive cases only
            v = np.array([r[k] for r in rows if r["has_tumor_gt"]], float)
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v))}
    tp, fn, fp = (sum(r[k] for r in rows) for k in ("tp", "fn", "fp"))
    out["lesion_sensitivity"] = tp / (tp + fn) if tp + fn else float("nan")
    out["fp_per_scan"] = fp / len(rows)
    y, yh = [r["has_tumor_gt"] for r in rows], [r["has_tumor_pred"] for r in rows]
    tpc = sum(a and b for a, b in zip(y, yh)); tnc = sum((not a) and (not b) for a, b in zip(y, yh))
    out["case_accuracy"] = (tpc + tnc) / len(rows)
    out["case_sensitivity"] = tpc / max(1, sum(y))
    out["case_specificity"] = tnc / max(1, len(y) - sum(y))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["phantom", "msd", "msd3d", "png"], default="phantom")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--depth-slices", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, choices=[2, 3], default=2, help="2D slice U-Net or 3D U-Net")
    ap.add_argument("--isotropic", action="store_true", help="3D: pool all axes equally (isotropic data)")
    ap.add_argument("--patch", type=int, nargs=3, default=None, help="3D patch size D H W")
    ap.add_argument("--samples-per-volume", type=int, default=4)
    ap.add_argument("--tumor-oversample", type=float, default=0.5)
    ap.add_argument("--base", type=int, default=None, help="base width (default 16 for 2D, 12 for 3D)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args(argv)

    if args.base is None:
        args.base = 12 if args.dim == 3 else 16
    seed_all(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    (trv, trs), (vav, vas), (tev, tes) = load(args)
    print(f"cases train/val/test = {len(trv)}/{len(vav)}/{len(tev)}")

    if args.dim == 3:
        model = SmallUNet3D(base=args.base, anisotropic=not args.isotropic).to(device)
        patch = tuple(args.patch) if args.patch else (min(32, args.depth_slices), 96, 96)
        ds = D.PatchDataset3D(trv, patch=patch, samples_per_volume=args.samples_per_volume,
                              tumor_oversample=args.tumor_oversample, seed=args.seed)
    else:
        model = SmallUNet(base=args.base).to(device)
        ds = D.SliceDataset(trv, augment=True, seed=args.seed)
    print(f"{type(model).__name__} params: {count_parameters(model):,}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    crit = DiceCELoss(ce_weight=[0.5, 1.0, 3.0]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl))

    best, log = -1.0, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step(); sched.step(); tot += loss.item()
        s = summarize(evaluate(model, vav, vas, device))
        score = (s["lung_dice"]["mean"] + s.get("tumor_dice", {"mean": 0})["mean"]) / 2
        log.append({"epoch": ep, "loss": tot / len(dl), "val_score": score, "sec": time.time() - t0})
        print(f"ep {ep:3d} loss {tot / len(dl):.4f} val lung {s['lung_dice']['mean']:.3f} "
              f"tumor {s.get('tumor_dice', {'mean': float('nan')})['mean']:.3f} ({time.time() - t0:.0f}s)")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"))["model"])
    res = summarize(evaluate(model, tev, tes, device))
    json.dump({"test": res, "log": log, "params": count_parameters(model), "args": vars(args)},
              open(os.path.join(args.out, "results.json"), "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
    if tev[0][0].shape[0] > 1:
        _, rep = reconstruct(model, tev[0][0], tes[0], os.path.join(args.out, "case0"), device)
        print("3D report:", json.dumps(rep)[:400])
    return res


if __name__ == "__main__":
    main()
