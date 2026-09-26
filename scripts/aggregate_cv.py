"""Aggregate 5-fold cross-validation into one out-of-fold result per patient.

Every one of the 63 MSD patients is tested exactly once (by the model that did
not see it). We report pooled per-patient statistics with bootstrap CIs, the
spread across folds, size-stratified results, and all controls.

  python scripts/aggregate_cv.py --out results/cv/cv_summary.json
"""
import argparse
import glob
import json
import os

import numpy as np


def boot(x, n=5000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if not len(x):
        return [float("nan")] * 3
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def pooled(rows, key):
    out = {m: boot([r[key][m] for r in rows]) for m in ("dice", "hd95", "sens", "prec")}
    out["median_dice"] = float(np.median([r[key]["dice"] for r in rows]))
    tp = sum(r[key]["tp"] for r in rows); fn = sum(r[key]["fn"] for r in rows); fp = sum(r[key]["fp"] for r in rows)
    out["lesion_sensitivity"] = tp / max(1, tp + fn)
    out["lesions"] = [tp, tp + fn]
    out["fp_per_scan"] = fp / len(rows)
    out["case_detection_rate"] = float(np.mean([r[key]["detected"] for r in rows]))
    gv = np.array([r[key]["vol_gt_ml"] for r in rows]); pv = np.array([r[key]["vol_pred_ml"] for r in rows])
    out["volume_abs_err_ml"] = boot(np.abs(pv - gv))
    out["volume_pearson_r"] = float(np.corrcoef(gv, pv)[0, 1]) if pv.std() > 0 and gv.std() > 0 else float("nan")
    # Bland-Altman on log volumes is more appropriate for a skewed size range
    d = np.log(pv + 0.1) - np.log(gv + 0.1)
    out["bland_altman_log"] = {"bias": float(d.mean()), "loa": [float(d.mean() - 1.96 * d.std()), float(d.mean() + 1.96 * d.std())]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", default="results/cv")
    ap.add_argument("--out", default="results/cv/cv_summary.json")
    a = ap.parse_args()
    rows, per_fold, sel = [], [], []
    for f in sorted(glob.glob(os.path.join(a.cv_dir, "fold*/test_eval.json"))):
        ev = json.load(open(f))
        k = int(os.path.basename(os.path.dirname(f)).replace("fold", ""))
        for r in ev["cases"]:
            r["fold"] = k
            rows.append(r)
        per_fold.append({"fold": k, "n": len(ev["cases"]), "dice": ev["summary"]["ours"]["dice"][0],
                         "dice_sel": ev["summary"].get("ours_sel", {}).get("dice", [float("nan")])[0]})
        sp = os.path.join(os.path.dirname(f), "postproc_selection.json")
        if os.path.exists(sp):
            sel.append({"fold": k, "rule": json.load(open(sp))["selected"]["rule"]})
    cases = [r["case"] for r in rows]
    assert len(cases) == len(set(cases)), "a patient was tested in more than one fold"
    vols = np.array([r["ours"]["vol_gt_ml"] for r in rows])
    terc = np.percentile(vols, [33.3, 66.7])
    strata = {}
    for name, lo, hi in (("small", 0, terc[0]), ("medium", terc[0], terc[1]), ("large", terc[1], np.inf)):
        sub = [r for r in rows if lo <= r["ours"]["vol_gt_ml"] < hi]
        strata[name] = {"range_ml": [float(lo), float(hi) if np.isfinite(hi) else None], "n": len(sub),
                        "dice": boot([r["ours"]["dice"] for r in sub]),
                        "dice_sel": boot([r["ours_sel"]["dice"] for r in sub]) if sub and "ours_sel" in sub[0] else None,
                        "ceiling": boot([r["ceiling_dice"] for r in sub])}
    summary = {
        "n_patients": len(rows), "n_folds": len(per_fold), "per_fold": per_fold, "selected_rules": sel,
        "ours": pooled(rows, "ours"),
        **({"ours_sel": pooled(rows, "ours_sel")} if rows and "ours_sel" in rows[0] else {}),
        "threshold": pooled(rows, "threshold"), "empty": pooled(rows, "empty"),
        "ceiling": boot([r["ceiling_dice"] for r in rows]),
        "fold_dice_sd": float(np.std([f["dice"] for f in per_fold], ddof=1)) if len(per_fold) > 1 else None,
        "strata": strata,
    }
    json.dump({"summary": summary, "cases": rows}, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("strata",)}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
