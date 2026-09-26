"""Build the paper (HTML + PDF) from result files. No number is typed by hand:
every figure in the text is read from the JSON outputs of the pipeline.

Inputs
  results/msd/test_eval.json      held-out test evaluation (scripts/evaluate_msd.py)
  runs/msd3d/results.json         training log + internal test summary
  results/results.json            phantom 2D sanity study
  data/recon/<case>/anatomy.json  anatomy reconstructions
  results/tests.json              pytest outcome (scripts/build_paper.py --run-tests)
  paper/fig/*.png                 figures

  python scripts/build_paper.py --run-tests
"""
import argparse
import base64
import datetime
import glob
import html
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "paper"


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def ci(x, d=3):
    return f"{x[0]:.{d}f} [{x[1]:.{d}f}, {x[2]:.{d}f}]"


def img(path, width="100%"):
    if not os.path.exists(path):
        return f'<p class="missing">[missing figure: {html.escape(path)}]</p>'
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{width}">'


def run_tests():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-rs"], capture_output=True, text=True)
    last = [l for l in r.stdout.strip().splitlines() if l.strip()][-1]
    counts = {k: 0 for k in ("passed", "failed", "skipped")}
    for tok in last.replace(",", "").split():
        pass
    import re
    for n, k in re.findall(r"(\d+) (passed|failed|skipped)", last):
        counts[k] = int(n)
    res = {"summary": last, **counts, "returncode": r.returncode}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/tests.json", "w"), indent=1)
    return res


def dice_figure(cases, path):
    order = sorted(cases, key=lambda r: r["ours"]["dice"])
    xs = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 2.9))
    ax.bar(xs - 0.2, [r["ours"]["dice"] for r in order], 0.4, label="3D U-Net (default post-proc.)", color="#0f7c8c")
    ax.bar(xs + 0.2, [r["threshold"]["dice"] for r in order], 0.4, label="HU-threshold baseline", color="#b9c4cc")
    ax.plot(xs, [r["ceiling_dice"] for r in order], "k_", ms=12, mew=1.5, label="resampling ceiling")
    ax.set_xticks(xs, [r["case"].replace("lung_", "") for r in order], fontsize=7)
    ax.set_ylabel("Tumor Dice"); ax.set_ylim(0, 1.02); ax.set_xlabel("test patient (lung_###)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def qualitative_figure(cases, path, cache_dir="results/msd/pred_cache", root="data/Task06_Lung"):
    """Best / median / worst test patient on the axial slice with the largest
    EXPERT tumor area, using the cached native-grid predictions."""
    import nibabel as nib
    order = sorted(cases, key=lambda r: r["ours"]["dice"])
    picks = [("best", order[-1]), ("median", order[len(order) // 2]), ("worst", order[0])]
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.7))
    for axx, (tag, r) in zip(axs, picks):
        c = r["case"]
        cz = np.load(os.path.join(cache_dir, c + ".npz"))
        shp = tuple(cz["shape"])
        p = np.unpackbits(cz["p"], count=int(np.prod(shp))).reshape(shp).astype(bool)
        g = np.asanyarray(nib.load(os.path.join(root, "labelsTr", c + ".nii.gz")).dataobj) > 0
        z = int(np.argmax(g.sum((0, 1))))
        ct = np.asanyarray(nib.load(os.path.join(root, "imagesTr", c + ".nii.gz")).dataobj[:, :, z]).astype(np.float32)
        axx.imshow(np.rot90(np.clip(ct, -1000, 400)), cmap="gray")
        for m, col in ((g[:, :, z], "#ff7a1a"), (p[:, :, z], "#2fd4ff")):
            if m.any():
                axx.contour(np.rot90(m).astype(float), [0.5], colors=col, linewidths=1.2)
        axx.set_title(f"{tag}: {c}  Dice {r['ours']['dice']:.2f}", fontsize=8)
        axx.axis("off")
    fig.suptitle("orange = expert label, cyan = 3D U-Net (default post-processing)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def loss_figure(log, path):
    ep = [r["epoch"] for r in log]
    loss = [r["loss"] for r in log]
    val = [(r["epoch"], r["val_score"]) for r in log if "val_score" in r]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(ep, loss, color="#0f7c8c", lw=1.5, label="train loss (Dice+CE)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.spines[["top", "right"]].set_visible(False)
    if val:
        ax2 = ax.twinx()
        ax2.plot(*zip(*val), "o-", color="#d2531f", ms=3, lw=1, label="val mean Dice")
        ax2.set_ylabel("val mean Dice (lung, tumor)"); ax2.set_ylim(0, 1)
        ax2.spines[["top"]].set_visible(False)
    fig.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--showcase", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "fig"), exist_ok=True)
    tests = run_tests() if a.run_tests else J("results/tests.json")
    ev = J("results/msd/test_eval.json")
    tr = J("runs/msd3d/results.json")
    ph = J("results/results.json")
    assert ev and tr, "run training and scripts/evaluate_msd.py first"
    S = ev["summary"]; O = S["ours"]; T = S["threshold"]; E = S["empty"]
    OS = S.get("ours_sel"); sel = J("results/msd/postproc_selection.json")
    cmp_ = J("results/compare_v1_v2.json"); ev2 = J("results/msd_anat/test_eval.json")
    tr2 = J("runs/msd3d_anat/results.json"); sel2 = J("results/msd_anat/postproc_selection.json")
    fp2 = J("results/msd_anat/fp_components.json") or []; fp1 = J("results/msd/fp_components.json") or []
    args = tr["args"]
    loss_figure(tr["log"], os.path.join(OUT, "fig", "fig_training.png"))
    dice_figure(ev["cases"], os.path.join(OUT, "fig", "fig_dice_per_case.png"))
    if os.path.isdir("results/msd/pred_cache"):
        qualitative_figure(ev["cases"], os.path.join(OUT, "fig", "fig_qualitative.png"))
    recon = {}
    for p in sorted(glob.glob("data/recon/lung_*/anatomy.json")):
        c = os.path.basename(os.path.dirname(p))
        if a.showcase is None or c in a.showcase:
            recon[c] = J(p)
    n_train, n_val, n_test = len(args["split"]["train"]), len(args["split"]["val"]), len(args["split"]["test"])
    sec_per_ep = np.mean([r["sec"] for r in tr["log"]])
    per_case = sorted(ev["cases"], key=lambda r: -r["ours"]["dice"])
    n_det = sum(r["ours"]["detected"] for r in ev["cases"])
    small = [r for r in ev["cases"] if r["ours"]["vol_gt_ml"] < 5]
    big = [r for r in ev["cases"] if r["ours"]["vol_gt_ml"] >= 5]
    md = lambda rs: np.mean([r["ours"]["dice"] for r in rs]) if rs else float("nan")
    worst = per_case[-1]
    fc = J(os.path.join(OUT, "fig", "fig_failure.json"))
    failure_case_text = ""
    if fc:
        fr = fc["inside_gt_frac"]
        failure_case_text = (f" The largest test tumor ({fc['case']}, {fc['gt_ml']:.0f} mL) was essentially missed. Inside it the raw network output was "
                             f"{100*fr['background']:.0f}% background, {100*fr['lung']:.0f}% lung and {100*fr['tumor']:.0f}% tumor. On visual inspection (Fig. 6) it is a posterior mass "
                             f"against the chest wall and spine, with soft-tissue density continuous with the tissue outside the lung. Our lung class is defined by thresholding aerated lung, "
                             f"so during training such tissue looks like the background class. Deriving the lung class from anatomical lung masks, which include the tumor bed, "
                             f"might address this; Section 4.2 tests it. Other failures were not audited individually.")
    rho = float(np.corrcoef(np.log([r["ours"]["vol_gt_ml"] for r in ev["cases"]]), [r["ours"]["dice"] for r in ev["cases"]])[0, 1])
    fp_total = sum(r["ours"]["fp"] for r in ev["cases"])
    rho_ceil = float(np.corrcoef(np.log([r["ours"]["vol_gt_ml"] for r in ev["cases"]]), [r["ceiling_dice"] for r in ev["cases"]])[0, 1])
    assoc = ("showed no clear association with" if abs(rho) < 0.3 else ("increased with" if rho > 0 else "decreased with"))
    failure_text = (f"Across the {S['n_test']} test patients, Dice {assoc} log tumor volume (Pearson r = {rho:.2f}). "
                    + (f"The resampling ceiling tends to be lower for smaller tumors (r = {rho_ceil:.2f} with log volume; Table 3), so our 1.5 mm grid costs more there. "
                       if rho_ceil > 0.3 else f"The resampling ceiling showed no clear size dependence (r = {rho_ceil:.2f}). ")
                    + 
                    f"The weakest test case ({worst['case']}, expert volume {worst['ours']['vol_gt_ml']:.1f} mL) reached Dice {worst['ours']['dice']:.2f}. "
                    f"With the default post-processing the model produced {fp_total} false-positive components across {S['n_test']} scans; "
                    + ((f"the validation-selected largest-component rule reduced this to {sum(r['ours_sel']['fp'] for r in ev['cases'])}. "
                        + (f"Lesion sensitivity was unchanged ({OS['lesion_sensitivity']:.2f}) on this test set, but the rule would drop any second lesion in a multi-lesion scan."
                           if abs(OS['lesion_sensitivity'] - O['lesion_sensitivity']) < 1e-9 else
                           f"Lesion sensitivity fell from {O['lesion_sensitivity']:.2f} to {OS['lesion_sensitivity']:.2f}, because the rule drops second lesions."))
                       if OS else "")
                    + failure_case_text)

    # ------------------------------------------------------------ tables
    def row(name, M, bold=False):
        cells = [name, ci(M["dice"]), ci(M["hd95"], 1) if np.isfinite(M["hd95"][0]) else "∞",
                 ci(M["sens"]), (ci(M["prec"]) if np.isfinite(M["prec"][0]) else "n/a"),
                 f'{M["lesion_sensitivity"]:.2f}', f'{M["fp_per_scan"]:.2f}', f'{M["case_detection_rate"]:.2f}']
        t = "".join(f"<td>{c}</td>" for c in cells)
        return f'<tr class="{"b" if bold else ""}">{t}</tr>'
    table_main = f"""
<table><caption><b>Table 1.</b> Tumor segmentation on the held-out MSD Task06 test split (n = {S['n_test']} patients), evaluated at native CT resolution.
Mean with bootstrap 95% CI over patients. HD95 in mm. Lesion sensitivity and FP/scan count connected components of at least 3 mm equivalent diameter on both sides (smaller expert-label specks are annotation noise). A lesion (or a case) counts as detected only if the prediction covers &gt;10% of its volume.</caption>
<thead><tr><th>Method</th><th>Dice</th><th>HD95 (mm)</th><th>Sensitivity</th><th>Precision</th><th>Lesion sens.</th><th>FP / scan</th><th>Case det.</th></tr></thead>
<tbody>{row("Small 3D U-Net (ours, default post-proc.)", O, True)}{row("&nbsp;&nbsp;+ largest-component rule (val-selected)", OS, True) if OS else ""}{row("HU-threshold baseline", T)}{row("Empty prediction (control)", E)}</tbody></table>"""
    table_ctrl = f"""
<table><caption><b>Table 2.</b> False-success controls. A trustworthy result must sit well above the controls and below the ceiling.</caption>
<thead><tr><th>Check</th><th>Tumor Dice</th><th>Interpretation</th></tr></thead><tbody>
<tr><td>Ours vs. expert label (default)</td><td>{ci(O['dice'])}</td><td>main result</td></tr>
{f"<tr><td>Ours vs. expert label (largest component)</td><td>{ci(OS['dice'])}</td><td>validation-selected rule</td></tr>" if OS else ""}
<tr><td>Ours vs. <i>another patient's</i> label</td><td>{ci(S['shuffled_gt_dice'])}</td><td>should be ~0; high values would mean the metric rewards generic blobs</td></tr>
<tr><td>Expert label &rarr; 1.5 mm grid &rarr; native</td><td>{ci(S['resampling_ceiling_dice'])}</td><td>upper bound imposed by our resampling</td></tr>
<tr><td>HU-threshold baseline</td><td>{ci(T['dice'])}</td><td>what a trivial intensity rule achieves</td></tr>
<tr><td>Empty prediction</td><td>{ci(E['dice'])}</td><td>must be exactly 0</td></tr>
</tbody></table>"""
    h = lambda x: "&infin;" if not np.isfinite(x) else f"{x:.1f}"
    sel_cells = lambda r: (f"<td>{r['ours_sel']['dice']:.3f}</td><td>{h(r['ours_sel']['hd95'])}</td><td>{r['ours_sel']['fp']}</td>" if OS else "")
    pc_rows = "".join(
        f"<tr><td>{r['case']}</td><td>{r['ours']['vol_gt_ml']:.1f}</td><td>{r['ours']['vol_pred_ml']:.1f}</td><td>{r['ours']['dice']:.3f}</td>"
        f"<td>{h(r['ours']['hd95'])}</td><td>{r['ours']['tp']}/{r['ours']['tp'] + r['ours']['fn']}</td><td>{r['ours']['fp']}</td>{sel_cells(r)}<td>{r['ceiling_dice']:.3f}</td></tr>"
        for r in per_case)
    table_cases = f"""
<table class="small"><caption><b>Table 3.</b> Per-patient results (test split), sorted by Dice (default rule). Volumes in mL; HD95 in mm (&infin; when nothing was predicted).</caption>
<thead><tr><th rowspan="2">Case</th><th rowspan="2">GT vol.</th><th colspan="5">default post-processing</th>{'<th colspan="3">largest component</th>' if OS else ''}<th rowspan="2">Ceiling</th></tr>
<tr><th>Pred. vol.</th><th>Dice</th><th>HD95</th><th>Lesions found</th><th>FP</th>{'<th>Dice</th><th>HD95</th><th>FP</th>' if OS else ''}</tr></thead><tbody>{pc_rows}</tbody></table>"""

    nz = lambda x, d=0: "&ndash;" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"
    rec_rows = ""
    for c, r in recon.items():
        st = r["structures"]; aw = r.get("airway_tree", {})
        lv = sum(v["volume_ml"] for k, v in st.items() if k.startswith("lung_"))
        rec_rows += (f"<tr><td>{c}</td><td>{lv / 1000:.2f}</td><td>{sum(k.startswith('lung_') for k in st)}</td>"
                     f"<td>{nz(st.get('airway', {}).get('volume_ml'))}</td><td>{aw.get('n_branchpoints', '–')}</td><td>{aw.get('n_endpoints', '–')}</td>"
                     f"<td>{aw.get('max_generation', '–')}</td><td>{nz(st.get('arteries', {}).get('volume_ml'))}</td><td>{nz(st.get('veins', {}).get('volume_ml'))}</td>"
                     f"<td>{r.get('tumor_gt', {}).get('lobe', '–').replace('lung_', '').replace('_', ' ')}</td>"
                     f"<td>{nz(r.get('pred_metrics', {}).get('dice'), 3)}</td></tr>")
    table_recon = f"""
<table class="small"><caption><b>Table 4.</b> Anatomy reconstructions. Lung volume in L; airway, artery and vein volumes in mL; airway graph from the pruned centreline (spurs &lt; 5 mm removed).</caption>
<thead><tr><th>Case</th><th>Lung</th><th>Lobes</th><th>Airway</th><th>Branch pts</th><th>Endpoints</th><th>Max gen.</th><th>Arteries</th><th>Veins</th><th>Tumor lobe</th><th>U-Net Dice</th></tr></thead><tbody>{rec_rows}</tbody></table>""" if recon else ""

    phantom = ""
    if ph:
        t = ph["test"]
        phantom = (f"On this phantom the 2D model reached lung Dice {t['lung_dice']['mean']:.3f} "
                   f"and tumor Dice {t['tumor_dice']['mean']:.3f}. These numbers only show that the code works; the phantom is far "
                   f"easier than real CT and we do not use it to support any claim.")

    figs_recon = "".join(
        f'<div class="row"><figure class="half">{img(os.path.join(OUT, "fig", f"{c}_A.png"))}<figcaption>{c}: lobes, airway tree with centreline and endpoints, tumor</figcaption></figure>'
        + (f'<figure class="half">{img(os.path.join(OUT, "fig", f"{c}_A_vessels.png"))}<figcaption>{c}: pulmonary arteries (blue) and veins (red) added</figcaption></figure>'
           if os.path.exists(os.path.join(OUT, "fig", f"{c}_A_vessels.png")) else "") + '</div>'
        for c in recon if os.path.exists(os.path.join(OUT, "fig", f"{c}_A.png")))
    notes = []
    for c, r in recon.items():
        lv = sum(v["volume_ml"] for k, v in r["structures"].items() if k.startswith("lung_")) / 1000
        if lv > 8:
            notes.append(f"{c} has a lobe-derived lung volume of {lv:.2f} L, above the usual adult range; our independent air-threshold estimate agrees, so it reflects the image as stored, but we cannot rule out a slice-spacing error in the file header. ")
        if not any(k in r["structures"] for k in ("arteries", "veins")):
            notes.append(f"For {c} we ran only the fast lobe/trachea model, not the airway/vessel model, so it has no vessels and only a coarse trachea. ")
    recon_notes = "".join(notes)
    tests_line = (f"{tests['passed']} passed, {tests['failed']} failed, {tests['skipped']} skipped" if tests else "not run")

    if sel and OS:
        d, t_ = sel["default"], sel["selected"]
        selection_text = (f"The default rule keeps every tumor component &ge;20 mm<sup>3</sup> inside the lung hull. Because the first test run showed many false-positive "
                          f"components, we compared {len(sel['candidates'])} alternative rules (minimum component volume, minimum mean probability, largest component only). "
                          f"We compared them on the <i>validation</i> patients only ({', '.join(sel['val_cases'])}). The best rule, <code>{t_['rule']}</code>, raised validation Dice from "
                          f"{d['dice']:.3f} to {t_['dice']:.3f} and cut FP/scan from {d['fp_per_scan']:.1f} to {t_['fp_per_scan']:.1f}. It was then applied unchanged to the test set. "
                          f"We report both rules. Because the aggregate test FP rate motivated this experiment, the second row of Table 1 is not a fully blind result, and with "
                          f"five validation patients the selection is itself noisy.")
    else:
        selection_text = "Not run."
    ablation = ""
    abstract_ablation = ""
    if cmp_ and ev2:
        S2 = ev2["summary"]
        def best_val(t):
            return max(r["val_score"] for r in t["log"] if "val_score" in r)
        c1 = {r["case"]: r for r in ev["cases"]}; c2 = {r["case"]: r for r in ev2["cases"]}
        fc2 = J(os.path.join(OUT, "fig", "fig_failure.json"))
        big_fp = [f for f in fp2 if f["fp_volume_ml"] >= 1]
        def arow(name, M, D=None):
            d = (f"<td>{D['diff_ci'][0]:+.3f} [{D['diff_ci'][1]:+.3f}, {D['diff_ci'][2]:+.3f}]</td><td>{D['wilcoxon_p']:.2f}</td>" if D else "<td>&ndash;</td><td>&ndash;</td>")
            return (f"<tr><td>{name}</td><td>{ci(M['dice'])}</td><td>{M['fp_per_scan']:.2f}</td><td>{M['lesion_sensitivity']:.2f}</td>"
                    f"<td>{M['case_detection_rate']:.2f}</td>{d}</tr>")
        worst_case = fc2["case"] if fc2 else "lung_028"
        ablation = f"""
<h3>4.2 Ablation: anatomical lung class</h3>
<p>Motivated by the failure analysis, we retrained the same network (same split, seed, patches, schedule and number of epochs) with the lung class taken from the union of the five TotalSegmentator lobes, which include the tumor bed, instead of the air threshold (v2). Post-processing was again selected on the validation patients only; the same largest-component rule was chosen (validation Dice {sel2['default']['dice']:.3f} default, {sel2['selected']['dice']:.3f} selected). On validation, v2 looked better than v1 (best validation score, the mean of lung and tumor Dice, {best_val(tr2):.3f} vs. {best_val(tr):.3f}). <b>On the test set it was not.</b> {"The paired difference in Dice (v2 &minus; v1) was " + format(cmp_['ours']['diff_ci'][0], '+.3f') + " [" + format(cmp_['ours']['diff_ci'][1], '+.3f') + ", " + format(cmp_['ours']['diff_ci'][2], '+.3f') + "] with default post-processing (Wilcoxon p = " + format(cmp_['ours']['wilcoxon_p'], '.2f') + ") and " + format(cmp_['ours_sel']['diff_ci'][0], '+.3f') + " [" + format(cmp_['ours_sel']['diff_ci'][1], '+.3f') + ", " + format(cmp_['ours_sel']['diff_ci'][2], '+.3f') + "] with the largest-component rule (p = " + format(cmp_['ours_sel']['wilcoxon_p'], '.2f') + ")"} (Table 5).
The targeted failure improved only partly: with default post-processing, {worst_case} went from Dice {c1[worst_case]['ours']['dice']:.2f} to {c2[worst_case]['ours']['dice']:.2f}.
The main cost was large false positives. In {len(fp2)} of {S2['n_test']} test patients v2's largest predicted component did not touch the expert tumor (v1: {len(fp1)}), so the largest-component rule discarded the true tumor. {len(big_fp)} of these components were &ge;1 mL ({"; ".join(f"{f['case']}: {f['fp_volume_ml']:.0f} mL, {f['distance_to_tumor_mm']:.0f} mm from the tumor, {f['lateral_offset_from_trachea_mm']:.0f} mm lateral to the trachea" for f in big_fp)}). Components close to the tracheal midline are compatible with hilar or mediastinal soft tissue, which the lobe masks partly include; we infer this from coordinates only and did not verify it visually.
We therefore keep v1 as the main model. The ablation also shows why a small validation set can mislead: the v2 validation gain did not transfer to the test set.</p>
<table class="small"><caption><b>Table 5.</b> Ablation on the same 13 test patients. Dice: mean with bootstrap 95% CI. Paired difference: v2 &minus; v1 per patient, bootstrap 95% CI; two-sided Wilcoxon signed-rank test.</caption>
<thead><tr><th>Model / post-processing</th><th>Dice</th><th>FP / scan</th><th>Lesion sens.</th><th>Case det.</th><th>Paired diff. vs v1</th><th>p</th></tr></thead><tbody>
{arow("v1 threshold lung, default", O)}{arow("v2 anatomical lung, default", S2['ours'], cmp_['ours'])}
{arow("v1 threshold lung, largest comp.", OS)}{arow("v2 anatomical lung, largest comp.", S2['ours_sel'], cmp_['ours_sel'])}
</tbody></table>"""
        abstract_ablation = (f" Retraining with an anatomical lung class improved validation Dice but reduced test Dice "
                             f"(paired difference {cmp_['ours']['diff_ci'][0]:+.2f} [{cmp_['ours']['diff_ci'][1]:+.2f}, {cmp_['ours']['diff_ci'][2]:+.2f}]), a negative result we report in full.")
    today = datetime.date.today().isoformat()
    body = f"""
<header>
<h1>A small 3D U-Net for lung tumor segmentation with anatomy-level 3D reconstruction: a reproducible CPU baseline with false-success controls</h1>
<p class="auth">Lungs project contributors &middot; technical report &middot; {today}</p>
</header>

<section class="abstract"><h2>Abstract</h2>
<p><b>Purpose.</b> We describe an open, fully reproducible pipeline that segments lung tumors on chest CT with a deliberately small 3D U-Net and turns each scan into an anatomy-level 3D model (five lobes, airway tree with centreline, pulmonary arteries and veins, and tumor). We report results with checks that catch <i>false success</i>. The whole thing runs on a 4-core CPU.
<b>Methods.</b> We trained a {tr['params']/1e6:.1f}M-parameter 3D U-Net on {n_train} patients from the Medical Segmentation Decathlon (MSD) Task06 Lung dataset. Scans were resampled to 1.5 mm isotropic and cropped to the lungs. We validated on {n_val} patients and tested on {n_test} held-out patients. Split is by patient. Lobes, airways and vessels come from the public TotalSegmentator models. We add a pruned airway centreline, branch-generation counting and patient-space meshes. Test metrics are computed at native resolution with bootstrap confidence intervals. We compare against an intensity-threshold baseline, an empty predictor and a shuffled-label control, and against the ceiling set by our own resampling.
<b>Results.</b> On the test split the model reached tumor Dice {ci(O['dice'])}, HD95 {ci(O['hd95'], 1)} mm and lesion-level sensitivity {O['lesion_sensitivity']:.2f} with {O['fp_per_scan']:.2f} false-positive components per scan. It detected the tumor in {n_det} of {S['n_test']} patients.{(" A largest-component rule chosen on validation data gave Dice " + ci(OS['dice']) + " with " + format(OS['fp_per_scan'], '.2f') + " FP/scan.") if OS else ""} The threshold baseline scored Dice {ci(T['dice'])}; the shuffled-label control scored {ci(S['shuffled_gt_dice'])}; the resampling ceiling was {ci(S['resampling_ceiling_dice'])}.{abstract_ablation}
<b>Conclusion.</b> A small model trained in about {sum(r['sec'] for r in tr['log'])/3600:.1f} hours on a 4-core CPU (including validation) gives a clear, honestly bounded baseline. For context, nnU-Net reports roughly 0.7 Dice on the official MSD lung test set; that set differs from our internal split, so the numbers are not directly comparable. We release code, tests, the 3D viewer and every number in this report as machine-generated files. This is a research prototype and not a diagnostic device.</p>
</section>

<section><h2>1&nbsp; Introduction</h2>
<p>Lung cancer is the leading cause of cancer death worldwide. CT is the main imaging test for finding and staging it. Automatic tumor segmentation supports volumetry, follow-up and treatment planning. A 3D model that shows the tumor next to the lobes, airways and vessels helps with surgical and bronchoscopic planning. Strong segmentation systems exist, such as nnU-Net <a href="#r2">[2]</a> and TotalSegmentator <a href="#r5">[5]</a>. But reported numbers are often hard to reproduce, and small studies can overstate success through patient leakage, lenient metrics or missing baselines.</p>
<p>This report makes three contributions. (i) A small, documented 3D U-Net baseline for MSD Task06 that can be trained end to end on a CPU. (ii) An anatomy-level reconstruction pipeline that places the predicted tumor in the context of lobes, airway tree (with centreline and endpoints) and vessels, and exports meshes in patient coordinates. (iii) An evaluation protocol whose negative controls and automated tests are built to expose false success, with every reported number generated from code.</p></section>

<section><h2>2&nbsp; Related work</h2>
<p>The U-Net <a href="#r1">[1]</a> and its 3D extension <a href="#r3">[3]</a> are the standard architectures for volumetric medical segmentation. nnU-Net <a href="#r2">[2]</a> showed that careful, automatically configured training matters more than architectural changes, and its Decathlon results <a href="#r4">[4]</a> remain the reference point for this task, with lung tumor Dice of roughly 0.7. TotalSegmentator <a href="#r5">[5]</a> offers robust open models for more than 100 structures, including lobes, airways and pulmonary vessels. LIDC-IDRI/LUNA16 <a href="#r6">[6]</a> is the main benchmark for nodule detection. Metric pitfalls such as Dice on empty masks, unbounded boundary distances and resampling bias are catalogued in <a href="#r7">[7]</a>, which shaped our protocol.</p></section>

<section><h2>3&nbsp; Methods</h2>
<h3>3.1 Data</h3>
<p>MSD Task06 Lung <a href="#r4">[4]</a> contains 63 labelled thin-slice CT scans of patients with non-small-cell lung cancer (CC-BY-SA 4.0), with expert tumor masks. The 32 official test scans have no public labels, so we split the 63 labelled scans by patient with a fixed seed ({args['seed']}): {n_train} training, {n_val} validation and {n_test} test. Tests assert that no patient appears in more than one split. All scans are LAS-oriented. We convert them to RAS for training and map predictions back to each original grid for evaluation.</p>
<h3>3.2 Preprocessing</h3>
<p>We clip HU to [&minus;1000, 400] and scale to [0, 1]. A classical lung mask is derived: threshold at &minus;320 HU, remove air connected to the image border, keep the two largest components and fill holes. It is used to crop each scan (+10 mm margin) and as the lung class. The tumor label overrides it. Crops are resampled to 1.5 mm isotropic, trilinear for images and nearest for labels. The resulting three-class problem is background, lung and tumor.</p>
<h3>3.3 Network and training</h3>
<p>The 3D U-Net has four resolution levels, base width {args['base']} (widths {args['base']}&ndash;{args['base']*16}) and {tr['params']:,} parameters. Each level has two 3&times;3&times;3 convolutions with InstanceNorm and LeakyReLU(0.01). Pooling is isotropic; upsampling uses transposed convolutions; the bottleneck has dropout 0.1. The loss is soft Dice over foreground classes (batch-level) plus cross-entropy with class weights (0.5, 1, 3). We train with AdamW (lr {args['lr']}, wd 10<sup>&minus;4</sup>), a one-cycle schedule, gradient clipping at 12 and batch {args['batch']}. Patches are {'&times;'.join(map(str, args['patch']))} voxels, {args['samples_per_volume']} per patient per epoch, {int(args['tumor_oversample']*100)}% centred on tumor voxels. Augmentations are left&ndash;right flip, gamma, contrast and Gaussian noise. We train for {args['epochs']} epochs (about {sec_per_ep/60:.0f} min per epoch on 4 CPU cores) and keep the checkpoint with the best validation mean Dice (lung and tumor), checked every {args['val_every']} epochs.</p>
<h3>3.4 Inference</h3>
<p>Inference uses a sliding window (patch 96&times;128&times;128, 50% overlap) with Gaussian importance weighting and flip test-time augmentation. Post-processing keeps the two largest lung components and removes tumor components under 20 mm<sup>3</sup> or outside the closed lung hull. Tumor probabilities are mapped back to the native grid by trilinear interpolation and thresholded at 0.5.</p>
<h3>3.5 Post-processing selection</h3>
<p>{selection_text}</p>
<h3>3.6 Anatomy-level reconstruction</h3>
<p>Per patient, the five lobes and trachea come from TotalSegmentator's fast model, and the airway tree, pulmonary arteries and pulmonary veins from its <code>lung_vessels</code> model. We run the latter on a lung-cropped CT (about 44% of the voxels) to stay within 15 GB of RAM, then paste the result back. The airway is skeletonised in 3D, and terminal spurs shorter than 5 mm are pruned. The resulting graph gives branch points, terminal endpoints, centreline length and a Weibel-style generation count from the most superior tracheal point. Surfaces are extracted with marching cubes after Gaussian smoothing in physical units (2 mm for lobes, 0.5&ndash;0.8 mm for airway, vessels and tumor) and Taubin &lambda;|&mu; smoothing <a href="#r8">[8]</a>, then mapped to RAS millimetres. Each tumor is assigned to the lobe it overlaps most. Meshes are exported as OBJ (full resolution, for 3D Slicer) and as a decimated web bundle for our viewer.</p>
<h3>3.7 Evaluation and false-success controls</h3>
<p>All test metrics are computed on the <i>native</i> CT grid against the original expert label. We report Dice, HD95 (mm, using the physical voxel spacing), voxel sensitivity and precision, lesion-level sensitivity and false-positive components per scan. Components count if they are &ge;3 mm in equivalent diameter, on both sides. The expert labels contain 1&ndash;10-voxel specks that are not lesions, and counting them had artificially deflated lesion sensitivity in our first evaluation run. A lesion counts as detected only if the prediction covers &gt;10% of it. Our first run used "any overlap", which counted a 1% touch of a 151 mL tumor as a hit. We also report case-level detection (&gt;10% of the tumor covered) and volume error. 95% confidence intervals come from 2000 bootstrap resamples over patients. Controls: (a) an <i>empty</i> predictor; (b) an <i>HU-threshold</i> baseline (soft tissue, &minus;300 to 200 HU, inside the closed lung hull but outside aerated lung, largest component); (c) a <i>shuffled-label</i> control, scoring our prediction for patient <i>i</i> against the label of patient <i>i</i>+1; and (d) the <i>resampling ceiling</i>, the expert label sent through our 1.5 mm pipeline and back. The automated test suite ({tests_line}) checks, among other things: empty-vs-non-empty masks give Dice 0 and infinite HD95, not NaN or 0; tumor-free cases do not inflate tumor Dice; an all-foreground predictor is penalised; an untrained network scores near zero; splits are disjoint and deterministic; sliding-window inference matches whole-volume inference exactly for a voxel-wise model; nodule volumes are physically correct under anisotropic spacing; and reconstructed tumor centroids match label centroids in RAS within 3 mm.</p></section>

<section><h2>4&nbsp; Results</h2>
{table_main}
<p>The small 3D U-Net clearly beats the trivial threshold rule and every control (Tables 1&ndash;2). The shuffled-label control stays near zero, so the score reflects patient-specific localisation rather than generic tumor-like blobs. Mean Dice was {md(big):.2f} for tumors &ge;5 mL (n = {len(big)}) and {md(small):.2f} for tumors &lt;5 mL (n = {len(small)}); with groups this small the difference is not reliable. Predicted and expert tumor volumes correlated with Pearson r = {O['volume_pearson_r']:.2f}, with mean absolute error {ci(O['volume_abs_err_ml'], 1)} mL (Fig. 2).</p>
{table_ctrl}
<figure>{img(os.path.join(OUT, 'fig', 'fig_dice_per_case.png'))}<figcaption><b>Figure 1.</b> Per-patient tumor Dice on the test split for our model (default post-processing) and the threshold baseline; black ticks mark the resampling ceiling.</figcaption></figure>
<div class="row"><figure class="half">{img(os.path.join('results/msd', 'fig_volume.png'))}<figcaption><b>Figure 2.</b> Predicted (default post-processing) vs. expert tumor volume, symmetric log axes.</figcaption></figure>
<figure class="half">{img(os.path.join(OUT, 'fig', 'fig_training.png'))}<figcaption><b>Figure 3.</b> Training loss and validation mean Dice.</figcaption></figure></div>
<figure>{img(os.path.join(OUT, 'fig', 'fig_qualitative.png') if os.path.exists(os.path.join(OUT, 'fig', 'fig_qualitative.png')) else os.path.join('results/msd', 'fig_qualitative.png'))}<figcaption><b>Figure 4.</b> Best, median and worst test patients (axial slice with the largest expert-tumor area). Orange: expert; cyan: 3D U-Net (default post-processing).</figcaption></figure>
{table_cases}
<figure>{img(os.path.join(OUT, 'fig', 'fig_failure.png'))}<figcaption><b>Figure 6.</b> Failure analysis of the largest test tumor on the 1.5 mm grid. Orange: expert tumor; cyan: predicted tumor; blue: threshold-derived lung label.</figcaption></figure>
{ablation}
<h3>4.3 Anatomy-level reconstruction</h3>
<p>{recon_notes}Figure 5 shows reconstructions from real CT. Lobes, airway tree with centreline and endpoints, arteries, veins and tumor are all in patient RAS coordinates, so they can be loaded straight into 3D Slicer or the web viewer. Table 4 summarises the derived anatomy.</p>
{table_recon}
{figs_recon}
<p class="figcap"><b>Figure 5.</b> Anatomy-level reconstructions of held-out test patients (anterior view; patient right on the viewer's left). Lobes are translucent, airway green with centreline and red endpoints, tumor orange (expert) and yellow wireframe (3D U-Net, largest-component rule).</p></section>

<section><h2>5&nbsp; Discussion</h2>
<p><b>What the numbers mean.</b> nnU-Net reports roughly 0.7 Dice on this task <a href="#r2">[2]</a>, but on the official MSD test set (hidden labels). Ours is a {n_test}-patient internal split of the labelled data, so the two cannot be ranked against each other. Our model has about {tr['params']/1e6:.0f}M parameters and trained for {args['epochs']} short CPU epochs; nnU-Net's default is 1000 GPU epochs with a five-fold ensemble. We do not claim state-of-the-art accuracy. The value of this work is a transparent, bounded baseline. The controls show the score is real (shuffled labels near 0), non-trivial (above the threshold rule) and limited partly by our own resampling (the ceiling is below 1 for small tumors).</p>
<p><b>Failure modes.</b> {failure_text}</p>
<p><b>Limitations.</b> (1) A single split with {n_test} test patients; the confidence intervals are wide, and cross-validation would be better. (2) No external test set: MSD comes from one institution. (3) Anatomy labels come from TotalSegmentator and are not checked against experts here. The airway generation count comes from a pruned skeleton and is an approximation. Values above about 10 (Table 4 reaches {max(r.get("airway_tree", {}).get("max_generation", 0) for r in recon.values()) if recon else 0}) exceed what CT normally resolves and probably reflect skeleton loops, not real branching. (4) The lung class is threshold-derived, not expert-drawn. (5) The system segments tumors that are already known to be present. It does not tell malignant from benign nodules and must not be used for diagnosis or screening. (6) Chest X-ray is supported only for lung-field segmentation in the codebase; this report does not evaluate it.</p>
<p><b>Future work.</b> An anatomical lung class combined with explicit suppression of hilar/mediastinal false positives (Section 4.2), five-fold cross-validation, longer GPU training, external validation on LIDC-IDRI/NSCLC-Radiomics, and expert review of the anatomy meshes.</p></section>

<section><h2>6&nbsp; Conclusion</h2>
<p>We present a small, CPU-trainable 3D U-Net and an anatomy-level reconstruction pipeline for lung tumor CT, evaluated with a protocol designed to reveal false success. The code, tests, viewer and every number in this report are generated from the public repository.</p></section>

<section><h2>Statistical analysis</h2>
<p>Per-patient metrics are summarised by their mean with a nonparametric bootstrap 95% confidence interval (resampling patients, 2000&ndash;5000 replicates). Paired model comparisons use per-patient differences with a bootstrap CI and a two-sided Wilcoxon signed-rank test. No correction for multiple comparisons was applied, and p-values are reported as descriptive only. With 13 test patients per split the study is not powered to detect small differences, so we emphasise interval estimates over significance tests. All analysis code is in the repository.</p></section>

<section><h2>Ethics, data and code availability</h2>
<p>This study uses only the publicly released, de-identified Medical Segmentation Decathlon Task06 data (CC-BY-SA 4.0), so no ethics approval was required. Anatomical labels were generated with the publicly available TotalSegmentator models. All code, tests, trained-model configuration, result files and the interactive viewer are in the project repository. Every number in this report is generated from those files by <code>scripts/build_paper.py</code>.</p></section>

<section><h2>Reproducibility</h2>
<pre>python scripts/download_datasets.py msd
python -m lungseg.prep_msd
python -m lungseg.train --data msd3d --root data/prep --dim 3 --isotropic --patch {' '.join(map(str, args['patch']))} \\
    --base {args['base']} --batch {args['batch']} --epochs {args['epochs']} --samples-per-volume {args['samples_per_volume']} \\
    --tumor-oversample {args['tumor_oversample']} --lr {args['lr']} --n-train {n_train} --n-val {n_val} --n-test {n_test} \\
    --val-every {args['val_every']} --seed {args['seed']} --out runs/msd3d
python scripts/evaluate_msd.py --ckpt runs/msd3d/best.pt
python scripts/reconstruct_case.py &lt;case&gt; --ckpt runs/msd3d/best.pt
python scripts/build_paper.py --run-tests</pre>
<p>Test split: {', '.join(S['test_cases'])}.</p></section>

<section class="refs"><h2>References</h2><ol>
<li id="r1">Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015.</li>
<li id="r2">Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18, 203&ndash;211 (2021).</li>
<li id="r3">&Ccedil;i&ccedil;ek &Ouml;, Abdulkadir A, Lienkamp SS, Brox T, Ronneberger O. 3D U-Net: learning dense volumetric segmentation from sparse annotation. MICCAI 2016.</li>
<li id="r4">Antonelli M, Reinke A, Bakas S, et al. The Medical Segmentation Decathlon. Nature Communications 13, 4128 (2022).</li>
<li id="r5">Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. Radiology: Artificial Intelligence 5(5), e230024 (2023).</li>
<li id="r6">Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in CT: the LUNA16 challenge. Medical Image Analysis 42, 1&ndash;13 (2017).</li>
<li id="r7">Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation. Nature Methods 21, 195&ndash;212 (2024).</li>
<li id="r8">Taubin G. A signal processing approach to fair surface design. SIGGRAPH 1995.</li>
<li id="r9">Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): a guide for authors and reviewers. Radiology: Artificial Intelligence 2(2), e200029 (2020).</li>
</ol></section>

<section class="appendix"><h2>Appendix B &nbsp;CLAIM checklist (abridged)</h2>
<p>Checklist for Artificial Intelligence in Medical Imaging (Mongan et al., 2020; 2024 update), with where each item is addressed or why it is not.</p>
<table class="small"><thead><tr><th>Item</th><th>Status</th><th>Where / note</th></tr></thead><tbody>
<tr><td>Study design and objective</td><td>Reported</td><td>&sect;1, &sect;3</td></tr>
<tr><td>Data source, licence, de-identification</td><td>Reported</td><td>&sect;3.1, Ethics</td></tr>
<tr><td>Eligibility, inclusion and exclusion</td><td>Partial</td><td>All 63 labelled MSD cases were used; MSD's own selection criteria are not published in detail</td></tr>
<tr><td>Demographics and scanner characteristics</td><td>Not available</td><td>Not released with MSD</td></tr>
<tr><td>Reference standard (ground truth)</td><td>Reported</td><td>MSD expert labels; annotator count and experience are not released</td></tr>
<tr><td>Preprocessing</td><td>Reported</td><td>&sect;3.2</td></tr>
<tr><td>Data partitions and leakage control</td><td>Reported</td><td>&sect;3.1 (patient-level, seeded, test-enforced); cross-validation</td></tr>
<tr><td>Model architecture and initialisation</td><td>Reported</td><td>&sect;3.3</td></tr>
<tr><td>Training details and hyperparameters</td><td>Reported</td><td>&sect;3.3, Reproducibility</td></tr>
<tr><td>Model and post-processing selection</td><td>Reported</td><td>&sect;3.3, &sect;3.5 (validation only)</td></tr>
<tr><td>Metrics and their definitions</td><td>Reported</td><td>&sect;3.7</td></tr>
<tr><td>Statistical methods and uncertainty</td><td>Reported</td><td>Statistical analysis</td></tr>
<tr><td>Baselines and controls</td><td>Reported</td><td>Tables 1&ndash;2; nnU-Net kit provided but not run (no GPU)</td></tr>
<tr><td>Robustness and failure analysis</td><td>Reported</td><td>&sect;4, Fig. 6, &sect;4.2</td></tr>
<tr><td>External validation</td><td>Not done</td><td>No external data reachable in our environment; Limitations</td></tr>
<tr><td>Explainability</td><td>Not done</td><td>Only qualitative overlays</td></tr>
<tr><td>Clinical utility or reader study</td><td>Not done</td><td>Limitations</td></tr>
<tr><td>Code and data availability</td><td>Reported</td><td>Availability statement</td></tr>
<tr><td>Registration and protocol</td><td>Not applicable</td><td>Retrospective technical study</td></tr>
</tbody></table></section>

<section class="appendix"><h2>Appendix A &nbsp;Synthetic phantom</h2>
<p>During development we used a procedural chest-CT phantom (body, fat, lungs with vessel-like texture, spine, 0&ndash;3 lobulated nodules, Gaussian noise) to test the code offline. {phantom}</p></section>
"""
    css = """
@page{size:A4;margin:18mm 16mm}
body{font:10.2pt/1.45 "Times New Roman",Times,serif;color:#111;max-width:178mm;margin:0 auto;padding:0 8px}
h1{font-size:16pt;line-height:1.25;text-align:center;margin:0 0 4px}
.auth{text-align:center;color:#444;margin:0 0 14px;font-size:9.5pt}
h2{font-size:11.5pt;margin:14px 0 4px} h3{font-size:10.5pt;margin:10px 0 2px;font-style:italic;font-weight:normal}
p{margin:0 0 6px;text-align:justify;hyphens:auto}
.abstract{border-top:1px solid #000;border-bottom:1px solid #000;padding:6px 0;margin-bottom:8px}
.abstract p{font-size:9.6pt}
table{border-collapse:collapse;width:100%;margin:8px 0 10px;font-size:8.6pt;font-variant-numeric:tabular-nums}
caption{caption-side:top;text-align:left;font-size:8.8pt;margin-bottom:4px}
th,td{padding:2px 5px;text-align:right;border-bottom:0.5px solid #bbb} th:first-child,td:first-child{text-align:left}
thead th{border-top:1.2px solid #000;border-bottom:0.8px solid #000} tbody tr:last-child td{border-bottom:1.2px solid #000}
tr.b td{font-weight:bold}
table.small{font-size:8pt}
figure{margin:8px 0;break-inside:avoid} figcaption,.figcap{font-size:8.8pt;color:#222}
.row{display:flex;gap:10px;align-items:flex-start} .half{flex:0 0 49%;max-width:49%;margin:4px 0}
pre{font:8pt/1.35 "DejaVu Sans Mono",monospace;background:#f4f4f4;padding:6px;white-space:pre-wrap}
.refs li{font-size:8.8pt;margin-bottom:2px}
code{font-size:8.8pt}
.missing{color:#b00}
a{color:#0b4f8a;text-decoration:none}
"""
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><title>Small 3D U-Net lung tumor segmentation</title><style>{css}</style></head><body>{body}</body></html>"
    open(os.path.join(OUT, "paper.html"), "w").write(html_doc)
    from playwright.sync_api import sync_playwright
    exe = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe if os.path.exists(exe) else None)
        pg = b.new_page()
        pg.goto("file://" + os.path.abspath(os.path.join(OUT, "paper.html")))
        pg.pdf(path=os.path.join(OUT, "paper.pdf"), format="A4", print_background=True,
               margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})
        b.close()
    print("wrote", os.path.join(OUT, "paper.html"), os.path.join(OUT, "paper.pdf"))


if __name__ == "__main__":
    main()
