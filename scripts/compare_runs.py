"""Paired comparison of two evaluated models on the same test patients.

  python scripts/compare_runs.py results/msd/test_eval.json results/msd_anat/test_eval.json \
      --names v1_threshold_lung v2_anatomical_lung --out results/compare_v1_v2.json

Reports per-patient Dice for both post-processing rules, the paired difference
(B - A) with a bootstrap 95% CI, and a two-sided Wilcoxon signed-rank test.
With ~13 patients the test has little power; the CI is the main summary.
"""
import argparse
import json

import numpy as np
from scipy import stats


def boot(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n, len(d))).mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--out", default="results/compare.json")
    x = ap.parse_args()
    A, B = json.load(open(x.a)), json.load(open(x.b))
    ca = {r["case"]: r for r in A["cases"]}; cb = {r["case"]: r for r in B["cases"]}
    assert set(ca) == set(cb), "different test patients - not a paired comparison"
    cases = sorted(ca)
    out = {"names": x.names, "cases": cases, "n": len(cases)}
    for key in ("ours", "ours_sel"):
        if key not in ca[cases[0]] or key not in cb[cases[0]]:
            continue
        da = np.array([ca[c][key]["dice"] for c in cases]); db = np.array([cb[c][key]["dice"] for c in cases])
        diff = db - da
        w = stats.wilcoxon(db, da) if np.any(diff != 0) else None
        out[key] = {"dice_a": da.tolist(), "dice_b": db.tolist(), "mean_a": float(da.mean()), "mean_b": float(db.mean()),
                    "diff_ci": boot(diff), "wilcoxon_p": float(w.pvalue) if w else 1.0,
                    "n_better": int((diff > 0.01).sum()), "n_worse": int((diff < -0.01).sum()),
                    "fp_a": sum(ca[c][key]["fp"] for c in cases), "fp_b": sum(cb[c][key]["fp"] for c in cases)}
        print(key, json.dumps({k: v for k, v in out[key].items() if not k.startswith("dice_")}))
    json.dump(out, open(x.out, "w"), indent=1)


if __name__ == "__main__":
    main()
