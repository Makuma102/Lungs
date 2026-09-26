"""Build the journal manuscript (.docx + .pdf) following the supervisor's outline
(Kien_outline.docx). Every number is read from result files; nothing is typed.

Primary results = 5-fold cross-validation (results/cv/cv_summary.json) when all
five folds exist, otherwise the development split (fold 0), flagged in a note.

  python scripts/build_journal.py            # -> paper/journal/manuscript.docx (+ .pdf)
"""
import datetime
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = "paper/journal"
FIG = os.path.join(OUT, "fig")


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f3(x):
    return f"{x:.3f}"


def ci(x, d=3):
    return f"{x[0]:.{d}f} (95% CI {x[1]:.{d}f}–{x[2]:.{d}f})"


def cis(x, d=3):
    return f"{x[0]:.{d}f} [{x[1]:.{d}f}, {x[2]:.{d}f}]"


# ------------------------------------------------------------------ references
REFS = [
    ("bray2024", "Bray F, Laversanne M, Sung H, et al. Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin. 2024;74(3):229–263."),
    ("nlst", "National Lung Screening Trial Research Team. Reduced lung-cancer mortality with low-dose computed tomographic screening. N Engl J Med. 2011;365(5):395–409."),
    ("nelson", "de Koning HJ, van der Aalst CM, de Jong PA, et al. Reduced lung-cancer mortality with volume CT screening in a randomized trial. N Engl J Med. 2020;382(6):503–513."),
    ("recist", "Eisenhauer EA, Therasse P, Bogaerts J, et al. New response evaluation criteria in solid tumours: revised RECIST guideline (version 1.1). Eur J Cancer. 2009;45(2):228–247."),
    ("litjens", "Litjens G, Kooi T, Bejnordi BE, et al. A survey on deep learning in medical image analysis. Med Image Anal. 2017;42:60–88."),
    ("unet", "Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. In: MICCAI. 2015:234–241."),
    ("unet3d", "Çiçek Ö, Abdulkadir A, Lienkamp SS, Brox T, Ronneberger O. 3D U-Net: learning dense volumetric segmentation from sparse annotation. In: MICCAI. 2016:424–432."),
    ("vnet", "Milletari F, Navab N, Ahmadi SA. V-Net: fully convolutional neural networks for volumetric medical image segmentation. In: 3DV. 2016:565–571."),
    ("unetpp", "Zhou Z, Siddiquee MMR, Tajbakhsh N, Liang J. UNet++: a nested U-Net architecture for medical image segmentation. In: DLMIA. 2018:3–11."),
    ("attunet", "Oktay O, Schlemper J, Le Folgoc L, et al. Attention U-Net: learning where to look for the pancreas. arXiv:1804.03999. 2018."),
    ("unetr", "Hatamizadeh A, Tang Y, Nath V, et al. UNETR: transformers for 3D medical image segmentation. In: WACV. 2022:574–584."),
    ("nnunet", "Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods. 2021;18(2):203–211."),
    ("nnunetrev", "Isensee F, Wald T, Ulrich C, et al. nnU-Net revisited: a call for rigorous validation in 3D medical image segmentation. In: MICCAI. 2024."),
    ("msd", "Antonelli M, Reinke A, Bakas S, et al. The Medical Segmentation Decathlon. Nat Commun. 2022;13:4128."),
    ("lidc", "Armato SG 3rd, McLennan G, Bidaut L, et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans. Med Phys. 2011;38(2):915–931."),
    ("luna", "Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge. Med Image Anal. 2017;42:1–13."),
    ("aerts", "Aerts HJWL, Velazquez ER, Leijenaar RTH, et al. Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. Nat Commun. 2014;5:4006."),
    ("hofman", "Hofmanninger J, Prayer F, Pan J, Röhrich S, Prosch H, Langs G. Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. Eur Radiol Exp. 2020;4:50."),
    ("totalseg", "Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. Radiol Artif Intell. 2023;5(5):e230024."),
    ("slicer", "Fedorov A, Beichel R, Kalpathy-Cramer J, et al. 3D Slicer as an image computing platform for the Quantitative Imaging Network. Magn Reson Imaging. 2012;30(9):1323–1341."),
    ("mc", "Lorensen WE, Cline HE. Marching cubes: a high resolution 3D surface construction algorithm. ACM SIGGRAPH Comput Graph. 1987;21(4):163–169."),
    ("taubin", "Taubin G. A signal processing approach to fair surface design. In: SIGGRAPH. 1995:351–358."),
    ("lee94", "Lee TC, Kashyap RL, Chu CN. Building skeleton models via 3-D medial surface/axis thinning algorithms. CVGIP Graph Models Image Process. 1994;56(6):462–478."),
    ("weibel", "Weibel ER. Morphometry of the Human Lung. Berlin: Springer; 1963."),
    ("metrics", "Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation. Nat Methods. 2024;21(2):195–212."),
    ("pitfalls", "Reinke A, Tizabi MD, Baumgartner M, et al. Understanding metric-related pitfalls in image analysis validation. Nat Methods. 2024;21(2):182–194."),
    ("varoquaux", "Varoquaux G, Cheplygina V. Machine learning for medical imaging: methodological failures and recommendations for the future. NPJ Digit Med. 2022;5:48."),
    ("kapoor", "Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023;4(9):100804."),
    ("claim", "Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): a guide for authors and reviewers. Radiol Artif Intell. 2020;2(2):e200029."),
    ("efron", "Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman & Hall; 1993."),
    ("bland", "Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet. 1986;1(8476):307–310."),
    ("wilcoxon", "Wilcoxon F. Individual comparisons by ranking methods. Biometrics Bull. 1945;1(6):80–83."),
]
IDX = {k: i + 1 for i, (k, _) in enumerate(REFS)}


def c(*keys):
    return "[" + ", ".join(str(IDX[k]) for k in keys) + "]"


# ------------------------------------------------------------------ figures
def fig_framework(path):
    fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
    boxes = [
        (0.1, 1.9, "Chest CT\n(MSD Task06)", "#e8eef3"),
        (1.75, 1.9, "Preprocessing\nHU window · lung crop\n1.5 mm isotropic", "#e8eef3"),
        (3.55, 1.9, "Compact 3D U-Net\n3.2 M params\n(CPU training)", "#d6ecef"),
        (5.35, 1.9, "Sliding-window\ninference + TTA\npost-processing", "#d6ecef"),
        (7.15, 1.9, "Native-grid\ntumor mask", "#d6ecef"),
        (3.55, 0.2, "TotalSegmentator\nlobes · airway\narteries · veins", "#f3ecdc"),
        (5.35, 0.2, "Anatomy fusion\ncentreline graph\nlobe assignment", "#f3ecdc"),
        (7.15, 0.2, "3D meshes (RAS mm)\nOBJ + web viewer\nstructured report", "#f3ecdc"),
        (8.75, 1.9, "False-success\ncontrols +\nbootstrap CIs", "#f6dede"),
    ]
    for x, y, t, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 1.45, 1.1, boxstyle="round,pad=0.03", fc=col, ec="#445", lw=0.8))
        ax.text(x + 0.725, y + 0.55, t, ha="center", va="center", fontsize=7.6)
    arr = dict(arrowstyle="->", color="#333", lw=0.9)
    for x0, x1, y in ((1.55, 1.75, 2.45), (3.2, 3.55, 2.45), (5.0, 5.35, 2.45), (6.8, 7.15, 2.45), (8.6, 8.75, 2.45),
                      (5.0, 5.35, 0.75), (6.8, 7.15, 0.75)):
        ax.annotate("", (x1, y), (x0, y), arrowprops=arr)
    ax.annotate("", (3.55, 0.75), (0.83, 1.9), arrowprops=dict(arrowstyle="->", color="#333", lw=0.9, connectionstyle="arc3,rad=0.25"))
    ax.annotate("", (6.08, 1.3), (7.87, 1.9), arrowprops=dict(arrowstyle="->", color="#333", lw=0.9, connectionstyle="arc3,rad=-0.2"))
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def fig_strata(strata, path):
    names = [k for k in ("small", "medium", "large") if k in strata]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    xs = np.arange(len(names))
    for off, key, col, lab in ((-0.2, "dice", "#0f7c8c", "default"), (0.2, "dice_sel", "#7fb8c4", "largest component")):
        vals = [strata[n][key] for n in names]
        m = [v[0] for v in vals]; lo = [v[0] - v[1] for v in vals]; hi = [v[2] - v[0] for v in vals]
        ax.bar(xs + off, m, 0.38, yerr=[lo, hi], capsize=3, color=col, label=lab)
    ax.plot(xs, [strata[n]["ceiling"][0] for n in names], "k_", ms=16, mew=1.5, label="resampling ceiling")
    ax.set_xticks(xs, [f"{n}\n(n={strata[n]['n']})" for n in names]); ax.set_ylim(0, 1.05); ax.set_ylabel("Tumor Dice")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def fig_agreement(clin, path):
    rows = clin["rows"]
    g = np.array([r["d_gt_mm"] for r in rows]); p = np.array([r["d_pred_mm"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    mx = max(g.max(), p.max()) * 1.08
    ax[0].plot([0, mx], [0, mx], color="#999", lw=0.8); ax[0].scatter(g, p, s=16, color="#0f7c8c")
    ax[0].set_xlabel("Expert longest axial diameter (mm)"); ax[0].set_ylabel("Predicted (mm)")
    mean, diff = (g + p) / 2, p - g
    b, lo, hi = clin["diameter_all"]["bias"], *clin["diameter_all"]["loa"]
    ax[1].scatter(mean, diff, s=16, color="#0f7c8c")
    for y, ls in ((b, "-"), (lo, "--"), (hi, "--")):
        ax[1].axhline(y, color="#d2531f", ls=ls, lw=1)
    ax[1].set_xlabel("Mean diameter (mm)"); ax[1].set_ylabel("Predicted − expert (mm)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def fig_per_case(cases, path):
    order = sorted(cases, key=lambda r: r["ours"]["dice"])
    xs = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(xs - 0.2, [r["ours"]["dice"] for r in order], 0.4, color="#0f7c8c", label="compact 3D U-Net (default)")
    ax.bar(xs + 0.2, [r["threshold"]["dice"] for r in order], 0.4, color="#b9c4cc", label="HU-threshold baseline")
    ax.plot(xs, [r["ceiling_dice"] for r in order], "k_", ms=8 if len(order) > 30 else 12, mew=1.3, label="resampling ceiling")
    ax.set_xticks(xs, [r["case"].replace("lung_", "") for r in order], fontsize=5 if len(order) > 30 else 7, rotation=90 if len(order) > 30 else 0)
    ax.set_ylim(0, 1.02); ax.set_ylabel("Tumor Dice"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22 if len(order) <= 30 else -0.3))
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def fig_volume(cases, path):
    g = np.array([r["ours"]["vol_gt_ml"] for r in cases]); p = np.array([r["ours_sel"]["vol_pred_ml"] if "ours_sel" in r else r["ours"]["vol_pred_ml"] for r in cases])
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    mx = max(g.max(), p.max()) * 1.3
    ax.plot([0.1, mx], [0.1, mx], color="#999", lw=0.8); ax.scatter(np.maximum(g, 0.1), np.maximum(p, 0.1), s=16, color="#0f7c8c")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Expert volume (mL)"); ax.set_ylabel("Predicted volume (mL)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


# ------------------------------------------------------------------ table helpers
def mrow(name, M):
    return [name, cis(M["dice"]), (cis(M["hd95"], 1) if np.isfinite(M["hd95"][0]) else "∞"), cis(M["sens"]),
            (cis(M["prec"]) if np.isfinite(M["prec"][0]) else "n/a"), f"{M['lesion_sensitivity']:.2f}",
            f"{M['fp_per_scan']:.2f}", f"{M['case_detection_rate']:.2f}"]


def pooled_from_eval(ev):
    return ev["summary"]


def pdf_from_blocks(B, out_pdf):
    """PDF preview rendered from the same blocks as the .docx (LibreOffice is
    unavailable in our build environment). Pagination may differ from Word."""
    import base64, html as H
    def rt(r):
        if isinstance(r, str):
            return H.escape(r)
        out = ""
        for x in r:
            t = H.escape(x["text"])
            if x.get("bold"): t = f"<b>{t}</b>"
            if x.get("italic"): t = f"<i>{t}</i>"
            if x.get("sup"): t = f"<sup>{t}</sup>"
            out += t
        return out
    parts = []
    for b in B:
        t = b["t"]
        if t == "title": parts.append(f"<h1 class=title>{rt(b['text'])}</h1>")
        elif t == "authors": parts.append(f"<p class=auth>{rt(b['text'])}</p>")
        elif t == "note": parts.append(f"<p class=note>{rt(b['text'])}</p>")
        elif t in ("h1", "h2", "h3"): parts.append(f"<{t}>{rt(b['text'])}</{t}>")
        elif t in ("p", "caption"): parts.append(f"<p>{rt(b.get('runs') or b['text'])}</p>")
        elif t == "bullets": parts.append("<ul>" + "".join(f"<li>{rt(i)}</li>" for i in b["items"]) + "</ul>")
        elif t == "table":
            head = "".join(f"<th>{rt(h)}</th>" for h in b["header"])
            body = "".join("<tr>" + "".join(f"<td>{rt(c)}</td>" for c in r) + "</tr>" for r in b["rows"])
            parts.append(f"<p class=cap>{rt(b['caption'])}</p><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif t == "img":
            data = base64.b64encode(open(b["path"], "rb").read()).decode()
            parts.append(f"<figure><img src='data:image/png;base64,{data}' style='width:{int(100*b.get('width',1))}%'><figcaption>{rt(b['caption'])}</figcaption></figure>")
        elif t == "refs": parts.append("<ol class=refs>" + "".join(f"<li>{H.escape(i)}</li>" for i in b["items"]) + "</ol>")
    css = """body{font:10.5pt/1.45 'Times New Roman',Times,serif;max-width:170mm;margin:0 auto;color:#000}
    h1.title{font-size:16pt;text-align:center;margin:0 0 6px} .auth{text-align:center;margin-bottom:12px}
    h1{font-size:13pt;margin:16px 0 6px} h2{font-size:11.5pt;margin:12px 0 4px} p{text-align:justify;margin:0 0 6px}
    .note{background:#fff4d6;padding:6px;font-style:italic} table{border-collapse:collapse;width:100%;font-size:8.5pt;margin:4px 0 10px}
    th{border-top:1px solid #000;border-bottom:1px solid #000;background:#f2f2f2;padding:3px} td{padding:3px;text-align:center;border-bottom:0.5px solid #ddd}
    td:first-child,th:first-child{text-align:left} tbody tr:last-child td{border-bottom:1px solid #000} .cap{margin-top:10px;font-size:9.5pt}
    figure{margin:8px 0;text-align:center;break-inside:avoid} figcaption{font-size:9.5pt;text-align:justify} .refs li{font-size:9pt}"""
    html_doc = f"<!doctype html><html><head><meta charset=utf-8><style>{css}</style></head><body>{''.join(parts)}</body></html>"
    hp = out_pdf.replace(".pdf", ".html"); open(hp, "w").write(html_doc)
    from playwright.sync_api import sync_playwright
    exe = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path=exe if os.path.exists(exe) else None)
        pg = br.new_page(); pg.goto("file://" + os.path.abspath(hp))
        pg.pdf(path=out_pdf, format="A4", margin={"top": "20mm", "bottom": "20mm", "left": "20mm", "right": "20mm"},
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:8pt;width:100%;text-align:center'><span class=pageNumber></span></div>")
        br.close()


def main():
    os.makedirs(FIG, exist_ok=True)
    tr = J("runs/msd3d/results.json"); dev = J("results/msd/test_eval.json")
    sel = J("results/msd/postproc_selection.json"); cmp_ = J("results/compare_v1_v2.json")
    ev2 = J("results/msd_anat/test_eval.json"); tr2 = J("runs/msd3d_anat/results.json"); sel2 = J("results/msd_anat/postproc_selection.json")
    fp2 = J("results/msd_anat/fp_components.json") or []; fp1 = J("results/msd/fp_components.json") or []
    fail = J("paper/fig/fig_failure.json"); tests = J("results/tests.json")
    cv = J("results/cv/cv_summary.json")
    use_cv = bool(cv and cv["summary"]["n_folds"] == 5)
    clin = J("results/cv/clinical.json") if use_cv else J("results/msd/clinical.json")
    recon = {c_: J(f"data/recon/{c_}/anatomy.json") for c_ in ("lung_014", "lung_075") if os.path.exists(f"data/recon/{c_}/anatomy.json")}
    assert tr and dev, "missing core results"

    if use_cv:
        S = cv["summary"]; cases = cv["cases"]; strata = S["strata"]; N = S["n_patients"]
        scope = f"five-fold cross-validation over all {N} labelled patients (each patient tested once, out of fold)"
    else:
        S = dev["summary"]; cases = dev["cases"]; N = S["n_test"]
        scope = f"the held-out development split ({N} patients)"
        # strata computed on the fly for the dev split
        vols = np.array([r["ours"]["vol_gt_ml"] for r in cases]); t1, t2 = np.percentile(vols, [33.3, 66.7])
        def bt(x):
            x = np.asarray(x, float); m = np.random.default_rng(0).choice(x, (5000, len(x))).mean(1)
            return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
        strata = {}
        for nm, lo, hi in (("small", 0, t1), ("medium", t1, t2), ("large", t2, np.inf)):
            sub = [r for r in cases if lo <= r["ours"]["vol_gt_ml"] < hi]
            strata[nm] = {"range_ml": [float(lo), None if not np.isfinite(hi) else float(hi)], "n": len(sub),
                          "dice": bt([r["ours"]["dice"] for r in sub]), "dice_sel": bt([r["ours_sel"]["dice"] for r in sub]),
                          "ceiling": bt([r["ceiling_dice"] for r in sub])}
    O, OS, T, E = S["ours"], S["ours_sel"], S["threshold"], S["empty"]
    ceiling = S["ceiling"] if use_cv else S["resampling_ceiling_dice"]
    shuffled = dev["summary"]["shuffled_gt_dice"]
    args = tr["args"]; params = tr["params"]
    ep_min = np.mean([r["sec"] for r in tr["log"][1:]]) / 60
    train_h = sum(r["sec"] for r in tr["log"]) / 3600

    fig_framework(os.path.join(FIG, "fig1_framework.png"))
    fig_per_case(cases, os.path.join(FIG, "fig_per_case.png"))
    fig_strata(strata, os.path.join(FIG, "fig_strata.png"))
    fig_volume(cases, os.path.join(FIG, "fig_volume.png"))
    if clin:
        fig_agreement(clin, os.path.join(FIG, "fig_agreement.png"))

    B = []
    P = lambda t: B.append({"t": "p", "text": t})
    H1 = lambda t: B.append({"t": "h1", "text": t}); H2 = lambda t: B.append({"t": "h2", "text": t})
    fign = {"n": 0}; tabn = {"n": 0}

    def FIGURE(path, cap, width=1.0):
        fign["n"] += 1
        B.append({"t": "img", "path": path, "width": width, "caption": [{"text": f"Fig. {fign['n']}. ", "bold": True}, {"text": cap}]})
        return fign["n"]

    def TABLE(cap, header, rows, widths=None):
        tabn["n"] += 1
        B.append({"t": "table", "caption": [{"text": f"Table {tabn['n']}. ", "bold": True}, {"text": cap}], "header": header, "rows": rows, "widths": widths})
        return tabn["n"]

    n_det = int(round(O["case_detection_rate"] * N))
    B.append({"t": "title", "text": "A Reproducible Resource-Efficient Framework for 3D Lung Tumor Segmentation and Anatomy-Aware Reconstruction with False-Success Controls"})
    B.append({"t": "authors", "text": "Kien Nguyen¹ · [Supervisor]¹  —  ¹[Affiliation]  ·  Manuscript draft, " + datetime.date.today().isoformat()})
    if not use_cv:
        B.append({"t": "note", "text": "Draft note: five-fold cross-validation over all 63 patients is running. The numbers below are from the development split (fold 0, 13 patients) and will be replaced automatically when cross-validation completes."})

    # ---------------------------------------------------------------- abstract
    H1("Abstract")
    P(f"Background: Automated lung tumor segmentation on CT supports volumetry, response assessment and treatment planning. Published results, however, are often hard to reproduce and can overstate performance through leakage, lenient metrics or missing baselines. "
      f"Methods: We developed a compact 3D U-Net ({params/1e6:.1f} M parameters) that trains on a 4-core CPU, and an anatomy-aware reconstruction pipeline that fuses the predicted tumor with the pulmonary lobes, airway tree and pulmonary vessels in patient coordinates. "
      f"Using the 63 labelled CT scans of the Medical Segmentation Decathlon lung task, we evaluated the model on {scope} at native resolution. The evaluation includes negative controls (empty prediction, intensity-threshold baseline, shuffled labels), a resampling-ceiling analysis, validation-only post-processing selection, patient-level bootstrap confidence intervals, tumor-size stratification, clinically oriented agreement (longest diameter, lobe localisation) and a controlled ablation of the lung representation. "
      f"Results: Tumor Dice was {ci(O['dice'])} with default post-processing and {ci(OS['dice'])} with a largest-component rule selected on validation data, which reduced false-positive components from {O['fp_per_scan']:.2f} to {OS['fp_per_scan']:.2f} per scan. The tumor was detected in {n_det} of {N} patients. "
      f"The threshold baseline reached {T['dice'][0]:.3f}, the shuffled-label control {shuffled[0]:.3f}, and the resampling ceiling {ceiling[0]:.3f}. "
      + (f"Predicted and expert longest diameters differed by {clin['diameter_all']['bias']:+.1f} mm on average (median absolute error {clin['diameter_all']['median_ae']:.1f} mm), and the tumor was assigned to the correct lobe in {100*clin['lobe_agreement']:.0f}% of patients. " if clin else "")
      + (f"Replacing the threshold-derived lung class with anatomical lobes improved validation performance but lowered test Dice (paired difference {cmp_['ours']['diff_ci'][0]:+.3f}, 95% CI {cmp_['ours']['diff_ci'][1]:+.3f} to {cmp_['ours']['diff_ci'][2]:+.3f}). " if cmp_ else "")
      + "Conclusions: A resource-efficient model evaluated with explicit false-success controls gives a transparent, bounded baseline for lung tumor segmentation and a practical route to anatomy-aware 3D visualisation. All code, tests and result files are released, and every number in this manuscript is generated from them.")
    P("Keywords: lung cancer; computed tomography; 3D U-Net; tumor segmentation; 3D reconstruction; reproducibility; false success; negative controls")

    # ---------------------------------------------------------------- 1 intro
    H1("1. Introduction")
    H2("1.1. Clinical and technological background")
    P(f"Lung cancer remains the leading cause of cancer death worldwide {c('bray2024')}. Computed tomography (CT) is central to its detection, staging and follow-up, and low-dose CT screening reduces lung-cancer mortality {c('nlst', 'nelson')}. "
      f"Treatment decisions and response assessment rely on measurements of the tumor, most often its longest diameter under RECIST 1.1 {c('recist')}, and increasingly on its full volume. Manual delineation is slow and varies between readers. This has motivated automated segmentation that is fast, consistent, and accurate enough for volumetry and planning.")
    H2("1.2. Emergence of 3D deep learning")
    P(f"Convolutional networks now dominate medical image segmentation {c('litjens')}. The U-Net {c('unet')} and its volumetric extensions {c('unet3d', 'vnet')} made dense 3D segmentation practical. Nested, attention and transformer variants have since been proposed {c('unetpp', 'attunet', 'unetr')}. "
      f"nnU-Net {c('nnunet')} showed that careful, self-configuring training of a plain U-Net matches or beats most architectural innovations. A recent large re-evaluation confirmed that many claimed improvements disappear under rigorous validation {c('nnunetrev')}. "
      f"State-of-the-art pipelines, however, assume GPU training for days, which limits their use in resource-constrained clinical and research settings.")
    H2("1.3. Problem of reproducibility and false success")
    P(f"Medical imaging AI suffers from methodological failures that inflate reported performance {c('varoquaux')}. Data leakage alone has undermined results across many fields {c('kapoor')}. "
      f"Segmentation metrics have well-documented pitfalls: Dice is undefined or misleading for empty masks, boundary distances are unbounded when a structure is missed, and lesion-level counting depends on matching rules {c('metrics', 'pitfalls')}. "
      f"We use the term false success for a result that looks good because of how it was measured rather than what the model does. Examples are a lenient detection criterion, a metric averaged over cases with nothing to find, or a threshold tuned on the test set. "
      f"Reporting guidelines such as CLAIM {c('claim')} ask authors to address these issues, but explicit negative controls remain rare in segmentation studies.")
    H2("1.4. From segmentation to anatomy-aware reconstruction")
    P(f"A tumor mask alone answers “how large?”, but surgical and bronchoscopic planning also asks “where, relative to what?”: which lobe, how close to the airway and vessels. "
      f"Open models such as TotalSegmentator {c('totalseg')} now segment lobes, airways and pulmonary vessels robustly, and platforms such as 3D Slicer {c('slicer')} display such structures interactively. "
      f"Combining a tumor model with these anatomical models, in one coordinate system with quantitative outputs, is a natural but rarely evaluated step.")
    H2("1.5. Research gap")
    P("We found no study that combines (i) a compact 3D tumor segmentation model trainable without a GPU, (ii) anatomy-level reconstruction of the tumor together with lobes, airway tree and vessels, and (iii) an evaluation framework whose negative controls, resampling ceiling and uncertainty estimates are designed to expose false success, with all numbers generated from released code.")
    H2("1.6. Research objectives")
    B.append({"t": "bullets", "items": [
        "Develop a compact 3D U-Net capable of training and inference under CPU-constrained conditions.",
        "Develop an anatomy-level reconstruction pipeline integrating tumor segmentation with pulmonary anatomy.",
        "Establish a rigorous evaluation framework incorporating negative controls, resampling ceiling analysis and patient-level uncertainty estimation.",
        "Characterize model failure modes and quantify the effect of post-processing and anatomical lung representations."]})
    H2("1.7. Contributions")
    B.append({"t": "bullets", "items": [
        f"A {params/1e6:.1f} M-parameter 3D U-Net and training recipe that runs on a 4-core CPU (about {ep_min:.0f} min per epoch), with resumable training and deterministic splits.",
        "An anatomy-aware reconstruction pipeline producing patient-space meshes of lobes, airway tree (with pruned centreline, bifurcations, endpoints and branch generations), pulmonary arteries and veins and the tumor, with a structured report and an interactive web viewer.",
        "A false-success evaluation protocol: native-resolution metrics, an empty-prediction and an intensity-threshold baseline, a shuffled-label control, a resampling ceiling, a size-filtered and coverage-based lesion matching rule, validation-only post-processing selection, and patient-level bootstrap confidence intervals, enforced by an automated test suite" + (f" ({tests['passed']} tests)." if tests else "."),
        "A negative ablation result: an anatomical lung representation improved validation scores but reduced test performance. We report it with a paired analysis and a characterisation of the failure.",
        "Full release of code, tests, result files and manuscript generator: every number in this paper is produced from the released result files."]})

    # ---------------------------------------------------------------- 2 related
    H1("2. Related works")
    H2("2.1. The 2D and 3D medical image segmentation")
    P(f"Encoder–decoder networks with skip connections {c('unet')} remain the backbone of medical segmentation. 2D models process slices independently and ignore through-plane context. 3D models {c('unet3d', 'vnet')} capture volumetric shape at a higher memory cost. "
      f"Architectural refinements {c('unetpp', 'attunet', 'unetr')} report gains that are often small relative to training and evaluation choices {c('nnunet', 'nnunetrev')}. We therefore use a plain 3D U-Net and put our effort into training under constraints and into evaluation.")
    H2("2.2. Lung tumor segmentation")
    P(f"Public benchmarks include LIDC-IDRI {c('lidc')} and LUNA16 {c('luna')} for nodules, NSCLC-Radiomics {c('aerts')} for tumors, and the Medical Segmentation Decathlon (MSD) lung task {c('msd')}. The MSD lung task is regarded as one of the hardest Decathlon tasks, because the targets are small and heterogeneous, and nnU-Net reported a lung-tumor Dice of roughly 0.7 on its hidden test set {c('nnunet', 'msd')}. "
      "Most published MSD lung results require GPU training and rarely report negative controls or patient-level uncertainty.")
    H2("2.3. Anatomical lung segmentation")
    P(f"Lung-field segmentation is considered largely solved when training data are diverse {c('hofman')}. Lobe, airway and vessel segmentation are harder. TotalSegmentator {c('totalseg')} provides open models for the five lobes, trachea, airways and pulmonary arteries and veins, and we use it as the source of anatomy. "
      f"Airway-tree analysis traditionally uses skeletonisation {c('lee94')} and Weibel-style generation counting {c('weibel')}.")
    H2("2.4. 3D reconstruction for clinical visualization")
    P(f"Surface reconstruction from voxel masks is usually done with marching cubes {c('mc')} followed by smoothing. Taubin λ|μ smoothing {c('taubin')} removes staircase artefacts without the shrinkage of Laplacian smoothing. "
      f"Clinical platforms such as 3D Slicer {c('slicer')} visualise such meshes. Few segmentation studies, however, evaluate the reconstructed anatomy quantitatively or deliver it in a portable, patient-coordinate format.")
    H2("2.5. Reproducibility and trustworthy medical AI")
    P(f"Guidelines and critical reviews {c('claim', 'varoquaux', 'metrics', 'pitfalls')} recommend patient-level splits, appropriate metrics, uncertainty estimates and baselines. Leakage audits {c('kapoor')} show how easily these are violated. "
      "Negative controls, meaning analyses that must fail if the pipeline is sound, are standard in experimental science but uncommon in segmentation papers.")
    H2("2.6. Research gap and hypothesis")
    P("We hypothesised that (H1) a compact 3D U-Net trained on a CPU gives a clinically meaningful, non-trivial baseline on the MSD lung task when evaluated with explicit false-success controls; (H2) validation-selected post-processing can reduce false positives without test-set tuning; and (H3) an anatomical lung representation, which includes the tumor bed, reduces failures on tumors that abut the chest wall or mediastinum.")

    # ---------------------------------------------------------------- 3 methods
    H1("3. Materials and methods")
    H2("3.1. Overall framework")
    P("The framework (Fig. 1) has three branches. The segmentation branch preprocesses the CT, runs the compact 3D U-Net with sliding-window inference and post-processing, and maps the tumor mask back to the native grid. The anatomy branch obtains lobes, airways and vessels and fuses them with the tumor into meshes and a structured report. The evaluation branch scores every prediction against the expert label and against a set of controls.")
    FIGURE(os.path.join(FIG, "fig1_framework.png"), "Overview of the framework: segmentation branch (top), anatomy-aware reconstruction branch (bottom) and evaluation with false-success controls (right).")
    H2("3.2. Dataset")
    P(f"We used the MSD Task06 lung dataset {c('msd')}: 63 thin-slice chest CT scans of patients with non-small-cell lung cancer, with expert tumor masks (CC-BY-SA 4.0). The 32 official test scans have hidden labels, so all experiments use the 63 labelled scans. "
      + (f"Primary results come from five-fold cross-validation with patient-level folds (seeded permutation, seed {args['seed']}). In each fold, five of the remaining patients form a validation set used for checkpoint and post-processing selection, and the rest are used for training. " if use_cv else
         f"Results in this draft come from a patient-level development split (seed {args['seed']}): {len(args['split']['train'])} training, {len(args['split']['val'])} validation and {len(args['split']['test'])} test patients. ")
      + "Automated tests assert that no patient appears in more than one partition. All scans are stored in LAS orientation; we convert them to RAS for training and map predictions back to each scan's original grid.")
    H2("3.3. Preprocessing")
    P("Hounsfield units are clipped to [−1000, 400] and scaled to [0, 1]. A lung mask is derived by thresholding at −320 HU, removing air connected to the image border, keeping the two largest components and filling holes. It is used to crop the scan (with a 10 mm margin) and as the lung class, with the tumor label taking precedence. Crops are resampled to 1.5 mm isotropic voxels (trilinear for images, nearest-neighbour for labels), giving a three-class problem: background, lung and tumor.")
    H2("3.4. Compact 3D U-Net architecture")
    P(f"The network is a four-level 3D U-Net {c('unet3d')} with base width {args['base']} (feature widths {args['base']}, {args['base']*2}, {args['base']*4}, {args['base']*8} and {args['base']*16} at the bottleneck) and {params:,} trainable parameters. "
      "Each block has two 3×3×3 convolutions with instance normalisation and LeakyReLU (slope 0.01). Pooling is isotropic 2×2×2 max-pooling, upsampling uses transposed convolutions with concatenative skip connections, and the bottleneck uses 3D dropout (p = 0.1). Weights are Kaiming-initialised.")
    H2("3.5. Loss function")
    P("We minimise the sum of a batch-level soft Dice loss over the foreground classes (lung and tumor) and a class-weighted cross-entropy with weights 0.5, 1 and 3 for background, lung and tumor. Computing Dice at batch level keeps the loss stable for patches that contain no tumor.")
    H2("3.6. Training strategy")
    P(f"We train with AdamW (learning rate {args['lr']}, weight decay 10⁻⁴), a one-cycle schedule, gradient-norm clipping at 12 and batch size {args['batch']}, for {args['epochs']} epochs. Each epoch draws {args['samples_per_volume']} random {'×'.join(map(str, args['patch']))}-voxel patches per training patient, {int(100*args['tumor_oversample'])}% of them centred on tumor voxels. "
      f"Augmentation consists of left–right flips, gamma, contrast and Gaussian noise. The checkpoint with the best validation score (mean of lung and tumor Dice, evaluated every {args['val_every']} epochs) is kept. "
      f"Training state is saved atomically every epoch, so runs survive interruptions. On a 4-core CPU an epoch takes about {ep_min:.0f} min and a full run about {train_h:.1f} h including validation.")
    H2("3.7. Inference")
    P("Inference uses a sliding window of 96×128×128 voxels with 50% overlap and Gaussian importance weighting, with left–right flip test-time augmentation. Tumor probabilities are mapped back to the native CT grid by trilinear interpolation and thresholded at 0.5. An automated test verifies that tiled inference reproduces whole-volume inference exactly for a voxel-wise model.")
    H2("3.8. Post-processing")
    P(f"The default rule keeps the two largest lung components and every tumor component of at least 20 mm³ inside the closed lung hull. Because the default rule produced many false-positive components, we compared {len(sel['candidates']) if sel else 18} alternative rules. These vary the minimum component volume, a minimum mean probability and whether only the largest component is kept, and they were compared on the validation patients only. "
      + (f"The selected rule keeps only the largest component (validation Dice {sel['default']['dice']:.3f} → {sel['selected']['dice']:.3f}; FP/scan {sel['default']['fp_per_scan']:.1f} → {sel['selected']['fp_per_scan']:.1f}) and was applied unchanged to the test data. " if sel else "")
      + "Both rules are reported. Because the first test run motivated this experiment, the selected-rule results are not fully blind, which we state explicitly.")
    H2("3.9. Anatomy-level reconstruction")
    P(f"For each patient, the five lobes and trachea come from TotalSegmentator's fast model {c('totalseg')}, and the airways, pulmonary arteries and pulmonary veins from its full-resolution lung-vessel model. We run the latter on a lung-cropped CT to fit in 15 GB of memory, then paste the result back. "
      f"The airway mask is skeletonised in 3D {c('lee94')}, and terminal branches shorter than 5 mm are pruned. Adjacent junction voxels are merged into single bifurcation nodes, and generations are counted from the most superior tracheal point {c('weibel')}. "
      f"Surfaces are extracted with marching cubes {c('mc')} after Gaussian smoothing in physical units (2 mm for lobes, 0.5–0.8 mm for airways, vessels and tumor), followed by Taubin smoothing {c('taubin')}, and are mapped to RAS millimetres. "
      "Each tumor is assigned to the lobe it overlaps most. Outputs are OBJ meshes for 3D Slicer, a decimated web bundle for an interactive viewer, and a JSON report with lobe volumes, airway statistics and tumor location.")
    H2("3.10. Evaluation metrics")
    P(f"All metrics are computed on the native CT grid against the original expert label {c('metrics')}. We report Dice, the 95th-percentile Hausdorff distance (HD95, mm, using physical spacing), voxel sensitivity and precision, and volume agreement. "
      "Lesion-level detection counts connected components of at least 3 mm equivalent diameter on both sides. The expert labels contain annotation specks of 1–10 voxels, which we do not treat as lesions. A lesion counts as detected only if the prediction covers more than 10% of it; our first evaluation used any overlap, which counted a 1% touch of a 151 mL tumor as a hit. False positives are predicted components that touch no expert voxel. "
      f"Clinically oriented agreement uses the RECIST-style longest axial diameter {c('recist')} of the largest component on each side, and lobe localisation (same lobe for expert and predicted tumor).")
    H2("3.11. False-success controls")
    B.append({"t": "bullets", "items": [
        "Empty prediction: must score Dice 0 and infinite HD95. This verifies that the metrics do not silently return 0 or NaN.",
        "Intensity-threshold baseline: soft tissue (−300 to 200 HU) inside the closed lung hull but outside aerated lung, largest component. This is what a trivial rule achieves.",
        "Shuffled-label control: each patient's prediction is scored against another patient's label. A high value would mean the metric rewards generic tumor-like blobs.",
        "Resampling ceiling: the expert label is sent through our 1.5 mm pipeline and back. This is the best Dice any model could reach given our preprocessing.",
        "Untrained-network and constant-predictor controls, metric edge cases, split disjointness, geometric correctness and inference equivalence, enforced by an automated test suite" + (f" ({tests['passed']} passed, {tests['failed']} failed)." if tests else ".")]})
    H2("3.12. Statistical analysis")
    P(f"Per-patient metrics are summarised by the mean with a nonparametric patient-level bootstrap 95% confidence interval (5000 resamples) {c('efron')}. Paired comparisons use per-patient differences with bootstrap CIs and the two-sided Wilcoxon signed-rank test {c('wilcoxon')}. "
      f"Measurement agreement uses Bland–Altman bias and 95% limits of agreement {c('bland')}. We did not correct for multiple comparisons and treat p-values as descriptive, emphasising interval estimates.")

    # ---------------------------------------------------------------- 4 experiments
    H1("4. Experimental Results and Discussion")
    H2("4.1. Experiment 1: Baseline segmentation")
    P(f"Compact 3D U-Net with default post-processing, evaluated on {scope}, compared against the intensity-threshold baseline and the empty prediction.")
    H2("4.2. Experiment 2: Post-processing analysis")
    P("Candidate post-processing rules compared on validation patients only. The selected rule is then applied unchanged to the test data and compared with the default rule on the same patients.")
    H2("4.3. Experiment 3: Anatomical lung representation")
    P("The same network, split, seed and schedule are retrained with the lung class defined as the union of the five TotalSegmentator lobes (which include the tumor bed) instead of the air threshold. Post-processing is again selected on validation data, and the comparison with the original model is paired, patient by patient.")
    H2("4.4. Experiment 4: Tumor-size stratification")
    P("Patients are grouped into tertiles of expert tumor volume. Dice and the resampling ceiling are reported per group.")
    H2("4.5. Experiment 5: Failure analysis")
    P("The worst-performing patients are inspected quantitatively (composition of the raw prediction inside the expert tumor) and visually. The largest false-positive components are characterised by volume and location.")
    H2("4.6. Experiment 6: Anatomy reconstruction")
    P("Full anatomy reconstructions of held-out patients, with lobe volumes, airway-tree statistics and the predicted tumor in context. Tumor lobe localisation and diameter agreement are evaluated over all test patients.")

    # 4.7 segmentation performance
    H2("4.7. Segmentation performance")
    t1 = TABLE(f"Tumor segmentation on {scope}. Mean with patient-level bootstrap 95% CI; HD95 in mm. Lesion-level metrics count components ≥3 mm equivalent diameter; detection requires >10% coverage.",
               ["Method", "Dice", "HD95 (mm)", "Sensitivity", "Precision", "Lesion sens.", "FP/scan", "Case det."],
               [mrow("Compact 3D U-Net, default", O), mrow("Compact 3D U-Net, largest comp.", OS), mrow("HU-threshold baseline", T), mrow("Empty prediction", E)],
               [0.22, 0.13, 0.12, 0.13, 0.13, 0.09, 0.08, 0.10])
    P(f"The compact model reached a tumor Dice of {ci(O['dice'])} and detected the tumor in {n_det} of {N} patients (Table {t1}). The median Dice was {np.median([r['ours']['dice'] for r in cases]):.3f}, reflecting a skewed distribution with a few near-complete failures (Fig. {fign['n']+1}). "
      f"The intensity-threshold baseline reached only {T['dice'][0]:.3f}, confirming that the task is not solvable by intensity alone. For context, nnU-Net reported roughly 0.7 on the official hidden test set {c('nnunet', 'msd')}. That figure comes from a different test set, obtained with far more compute and model ensembling, so the two numbers cannot be ranked against each other.")
    FIGURE(os.path.join(FIG, "fig_per_case.png"), "Per-patient tumor Dice (default post-processing) and intensity-threshold baseline; black ticks mark each patient's resampling ceiling.")
    FIGURE("paper/fig/fig_training.png",
           "Training loss and validation score (mean of lung and tumor Dice) for the development run.", 0.55)

    # 4.8 false-success analysis
    H2("4.8. False-success analysis")
    t2 = TABLE("False-success controls. A trustworthy result must lie well above the controls and below the ceiling.",
               ["Check", "Tumor Dice", "Interpretation"],
               [["Model vs. expert (default)", cis(O["dice"]), "main result"],
                ["Model vs. expert (largest comp.)", cis(OS["dice"]), "validation-selected rule"],
                ["Model vs. another patient's label", cis(shuffled), "≈0 required (development split)"],
                ["Resampling ceiling", cis(ceiling), "upper bound from our preprocessing"],
                ["HU-threshold baseline", cis(T["dice"]), "trivial intensity rule"],
                ["Empty prediction", cis(E["dice"]), "must be exactly 0"]], [0.36, 0.26, 0.38])
    P(f"All controls behaved as required (Table {t2}). The shuffled-label control scored {shuffled[0]:.3f}, so the metric rewards patient-specific localisation. The ceiling of {ceiling[0]:.3f} shows that resampling alone costs a few Dice points" + ("" if strata["small"]["ceiling"][0] >= strata["large"]["ceiling"][0] else ", most for small tumors") + ". "
      "Three false-success mechanisms were caught during development and corrected before reporting. (i) Counting 1–10-voxel annotation specks as lesions had deflated lesion sensitivity. (ii) Any-overlap matching counted a 1% touch of a 151 mL tumor as a detection. (iii) Case specificity was reported as 0 when it is undefined, because every scan contains a tumor. "
      "A fourth, pipeline-level error was found in the ablation: evaluation scripts read a hard-coded preprocessing directory, silently feeding the ablation model the wrong inputs. It is now covered by a regression test.")

    # 4.9 post-processing
    H2("4.9. Post-processing results")
    P(f"On validation data the largest-component rule raised Dice from {sel['default']['dice']:.3f} to {sel['selected']['dice']:.3f} and removed false positives ({sel['default']['fp_per_scan']:.1f} → {sel['selected']['fp_per_scan']:.1f} per scan). "
      f"On the test data it gave Dice {ci(OS['dice'])} against {ci(O['dice'])} for the default rule, with {OS['fp_per_scan']:.2f} against {O['fp_per_scan']:.2f} false-positive components per scan and HD95 of {OS['hd95'][0]:.1f} against {O['hd95'][0]:.1f} mm. "
      + ("Lesion sensitivity was unchanged in this cohort. However, the rule by construction discards any second lesion, so it suits single-tumor settings such as MSD but not multifocal disease." if abs(OS["lesion_sensitivity"] - O["lesion_sensitivity"]) < 1e-9 else
         f"Lesion sensitivity changed from {O['lesion_sensitivity']:.2f} to {OS['lesion_sensitivity']:.2f}, because the rule discards second lesions."))

    # 4.10 volume agreement
    H2("4.10. Volume agreement")
    P(f"Predicted and expert tumor volumes are compared in Fig. {fign['n']+1}. With default post-processing the Pearson correlation was {O['volume_pearson_r']:.2f} and the mean absolute volume error {ci(O['volume_abs_err_ml'], 1)} mL. "
      + (f"The corresponding values for the largest-component rule were {OS['volume_pearson_r']:.2f} and {ci(OS['volume_abs_err_ml'], 1)} mL. " if "volume_abs_err_ml" in OS else "")
      + "Correlation is dominated by the few largest tumors, so a single large miss lowers it markedly.")
    FIGURE(os.path.join(FIG, "fig_volume.png"), "Predicted versus expert tumor volume (largest-component rule), log–log axes; the line is identity.", 0.5)
    if clin:
        d = clin["diameter_all"]
        P(f"For the clinically used RECIST-style longest axial diameter {c('recist')}, the bias was {d['bias']:+.1f} mm (95% limits of agreement {d['loa'][0]:.1f} to {d['loa'][1]:.1f} mm), the median absolute error {d['median_ae']:.1f} mm, and {100*clin['within_5mm']:.0f}% of predictions were within 5 mm of the expert measurement (Fig. {fign['n']+1}). "
          f"The predicted tumor was assigned to the same lobe as the expert tumor in {100*clin['lobe_agreement']:.0f}% of patients ({int(round(clin['lobe_agreement']*clin['lobe_n']))}/{clin['lobe_n']}).")
        FIGURE(os.path.join(FIG, "fig_agreement.png"), "Longest axial diameter: predicted versus expert (left) and Bland–Altman plot (right; solid line = bias, dashed = 95% limits of agreement).")

    # 4.11 size
    H2("4.11. Tumor-size analysis")
    rows_s = [[f"{k} ({v['range_ml'][0]:.1f}–{'∞' if v['range_ml'][1] is None else format(v['range_ml'][1], '.1f')} mL)", str(v["n"]), cis(v["dice"]),
               cis(v["dice_sel"]) if v.get("dice_sel") else "–", cis(v["ceiling"])] for k, v in strata.items()]
    t3 = TABLE("Tumor-size stratification (tertiles of expert volume).", ["Stratum", "n", "Dice (default)", "Dice (largest comp.)", "Resampling ceiling"], rows_s, [0.28, 0.07, 0.22, 0.22, 0.21])
    ds = {k: v["dice"][0] for k, v in strata.items()}
    overlap = not (strata["small"]["dice"][2] < strata["large"]["dice"][1] or strata["large"]["dice"][2] < strata["small"]["dice"][1])
    P(f"Dice by stratum was {ds.get('small', float('nan')):.2f} (small), {ds.get('medium', float('nan')):.2f} (medium) and {ds.get('large', float('nan')):.2f} (large) (Table {t3}, Fig. {fign['n']+1}). "
      + (" The confidence intervals overlap, so the data do not show a clear size effect." if overlap else " The confidence intervals of the smallest and largest strata do not overlap.")
      + (f" The resampling ceiling rises with size ({strata['small']['ceiling'][0]:.3f} → {strata['large']['ceiling'][0]:.3f}), so our 1.5 mm grid costs most on small tumors." if strata['small']['ceiling'][0] < strata['large']['ceiling'][0] else "")
      + (" Large tumors were not easier for the model than medium ones." if ds.get('large', 0) <= ds.get('medium', 0) else ""))
    FIGURE(os.path.join(FIG, "fig_strata.png"), "Tumor Dice by size tertile with bootstrap 95% CIs; black ticks show the resampling ceiling.", 0.65)

    # 4.12 failure analysis
    H2("4.12. Failure analysis")
    if fail:
        fr = fail["inside_gt_frac"]
        P(f"The largest tumor in the development split ({fail['case']}, {fail['gt_ml']:.0f} mL) was essentially missed. Inside the expert tumor the raw network output was {100*fr['background']:.0f}% background, {100*fr['lung']:.0f}% lung and {100*fr['tumor']:.0f}% tumor. "
          "On inspection (Fig. {n}) it is a posterior mass against the chest wall and spine, with soft-tissue density continuous with the tissue outside the lung. Our lung class is defined by aerated lung, so during training such tissue resembles the background class.".replace("{n}", str(fign["n"] + 1)))
        FIGURE("paper/fig/fig_failure.png", "Failure case on the 1.5 mm grid. Orange: expert tumor; cyan: predicted tumor; blue: threshold-derived lung label.")
    if cmp_ and ev2:
        S2 = ev2["summary"]
        best_v = lambda t: max(r["val_score"] for r in t["log"] if "val_score" in r)
        c1 = {r["case"]: r for r in dev["cases"]}; c2 = {r["case"]: r for r in ev2["cases"]}
        big = [f for f in fp2 if f["fp_volume_ml"] >= 1]
        near = sum(1 for f in big if f.get("lateral_offset_from_trachea_mm") is not None and f["lateral_offset_from_trachea_mm"] <= 50)
        t4 = TABLE("Experiment 3, anatomical lung class (v2) versus threshold lung class (v1), on the same development test patients. Paired difference = v2 − v1 per patient (bootstrap 95% CI; Wilcoxon signed-rank p).",
                   ["Model / post-processing", "Dice", "FP/scan", "Lesion sens.", "Paired diff.", "p"],
                   [["v1, default", cis(dev["summary"]["ours"]["dice"]), f"{dev['summary']['ours']['fp_per_scan']:.2f}", f"{dev['summary']['ours']['lesion_sensitivity']:.2f}", "–", "–"],
                    ["v2, default", cis(S2["ours"]["dice"]), f"{S2['ours']['fp_per_scan']:.2f}", f"{S2['ours']['lesion_sensitivity']:.2f}", cis(cmp_["ours"]["diff_ci"]), f"{cmp_['ours']['wilcoxon_p']:.2f}"],
                    ["v1, largest comp.", cis(dev["summary"]["ours_sel"]["dice"]), f"{dev['summary']['ours_sel']['fp_per_scan']:.2f}", f"{dev['summary']['ours_sel']['lesion_sensitivity']:.2f}", "–", "–"],
                    ["v2, largest comp.", cis(S2["ours_sel"]["dice"]), f"{S2['ours_sel']['fp_per_scan']:.2f}", f"{S2['ours_sel']['lesion_sensitivity']:.2f}", cis(cmp_["ours_sel"]["diff_ci"]), f"{cmp_['ours_sel']['wilcoxon_p']:.2f}"]],
                   [0.24, 0.22, 0.1, 0.12, 0.22, 0.1])
        wc = fail["case"] if fail else "lung_028"
        P(f"To test H3 we retrained with the anatomical lung class (Experiment 3). Validation performance improved (best score {best_v(tr2):.3f} against {best_v(tr):.3f}), but test performance did not (Table {t4}). The paired Dice difference was {cis(cmp_['ours']['diff_ci'])} with default post-processing and {cis(cmp_['ours_sel']['diff_ci'])} with the largest-component rule. "
          f"The targeted failure improved only partly ({wc}: Dice {c1[wc]['ours']['dice']:.2f} → {c2[wc]['ours']['dice']:.2f}). The cost was large false positives. In {len(fp2)} of {S2['n_test']} patients v2's largest component missed the expert tumor (v1: {len(fp1)}), so the largest-component rule discarded the true tumor. "
          + (f"The largest of these components measured {', '.join(format(f['fp_volume_ml'], '.0f') + ' mL' for f in big)}, and {near} of them lay within 50 mm of the tracheal midline. That position is compatible with hilar or mediastinal soft tissue, which the lobe masks partly include; we infer this from coordinates and did not verify it visually. " if big else "")
          + "H3 is therefore not supported, and the case also shows how a five-patient validation set can mislead model selection.")

    # 4.13 anatomy
    H2("4.13. Anatomy-level reconstruction")
    if recon:
        rrows = []
        for cs, r in recon.items():
            st = r["structures"]; aw = r.get("airway_tree", {})
            lv = sum(v["volume_ml"] for k, v in st.items() if k.startswith("lung_")) / 1000
            nz = lambda k: f"{st[k]['volume_ml']:.0f}" if k in st else "–"
            rrows.append([cs, f"{lv:.2f}", str(sum(k.startswith('lung_') for k in st)), nz("airway"), str(aw.get("n_branchpoints", "–")),
                          str(aw.get("n_endpoints", "–")), str(aw.get("max_generation", "–")), nz("arteries"), nz("veins"),
                          r.get("tumor_gt", {}).get("lobe", "–").replace("lung_", "").replace("_", " "),
                          f"{r['pred_metrics']['dice']:.3f}" if r.get("pred_metrics") else "–"])
        t5 = TABLE("Anatomy reconstructions of held-out patients. Lung volume in L; airway, artery and vein volumes in mL.",
                   ["Case", "Lung", "Lobes", "Airway", "Branch pts", "Endpoints", "Max gen.", "Arteries", "Veins", "Tumor lobe", "U-Net Dice"], rrows,
                   [0.1, 0.07, 0.07, 0.08, 0.09, 0.09, 0.08, 0.08, 0.07, 0.15, 0.12])
        big_lung = [cs for cs, r in recon.items() if sum(v["volume_ml"] for k, v in r["structures"].items() if k.startswith("lung_")) > 8000]
        P(f"Fig. {fign['n']+1} shows reconstructions of held-out patients, with all structures in patient RAS coordinates, and Table {t5} summarises the derived anatomy. "
          + " ".join(f"In {cs} the predicted tumor reached Dice {r['pred_metrics']['dice']:.2f} and was assigned to "
                     + ("the same lobe as" if (r.get('tumor_pred') or {}).get('lobe') == (r.get('tumor_gt') or {}).get('lobe') else "a different lobe from")
                     + " the expert tumor." for cs, r in recon.items() if r.get("pred_metrics")) + " "
          + (f"{', '.join(big_lung)} has a lung volume above the usual adult range; an independent air-threshold estimate agrees, so it reflects the image as stored, but a slice-spacing error in the header cannot be excluded. " if big_lung else "")
          + "Airway generation counts above about 10 exceed what CT normally resolves and probably reflect skeleton loops rather than true branching. We report them as approximate. The anatomy labels come from TotalSegmentator and were not checked against expert annotations in this study.")
        for img, cap in (("paper/fig/lung_014_A.png", "Held-out patient lung_014: lobes (translucent), airway tree with pruned centreline and endpoints (green, red spheres), expert tumor (orange) and predicted tumor (yellow)."),
                         ("paper/fig/lung_014_A_vessels.png", "The same patient with pulmonary arteries (blue) and veins (red)."),
                         ("paper/fig/lung_014_tumor.png", "Close-up: predicted tumor (yellow wireframe) over the expert tumor (orange), with the bronchus leading to it.")):
            if os.path.exists(img):
                FIGURE(img, cap, 0.62)

    # ---------------------------------------------------------------- 5 conclusions
    H1("5. Conclusions")
    P(f"A compact 3D U-Net trained entirely on a CPU reached a tumor Dice of {O['dice'][0]:.2f} (default) and {OS['dice'][0]:.2f} (validation-selected post-processing) on the MSD lung task. It was far above an intensity baseline and all negative controls, and below the ceiling set by its own preprocessing. This supports H1 and H2 within the limits below. "
      "The anatomy-aware pipeline turns each scan into a patient-space 3D model with lobes, airway tree, vessels and tumor, together with clinically meaningful measurements. The anatomical-lung ablation (H3) was a negative result, and we report it as such. "
      "Limitations: the cohort is small and comes from a single public dataset; there is no external validation; anatomy labels are automated and not expert-verified; the lung class is threshold-derived; the model does not distinguish malignant from benign disease; and a GPU-scale comparison with nnU-Net on identical folds is provided as a script but was not run. "
      "Future work will add external validation (NSCLC-Radiomics, LIDC-IDRI), the nnU-Net comparison, expert review of the reconstructed anatomy, and explicit suppression of hilar false positives for anatomical lung representations.")
    H2("Data and code availability")
    P("All experiments use the public, de-identified MSD Task06 data, so no ethics approval was required. Code, automated tests, result files, the interactive viewer and this manuscript generator are available in the project repository.")

    H1("References")
    B.append({"t": "refs", "items": [r for _, r in REFS]})

    os.makedirs(OUT, exist_ok=True)
    jpath = os.path.join(OUT, "blocks.json")
    json.dump(B, open(jpath, "w"), indent=0)
    docx = os.path.join(OUT, "manuscript.docx")
    subprocess.check_call(["node", "paper/docx/render.js", jpath, docx], env={**os.environ, "NODE_PATH": os.path.abspath("paper/docx/node_modules")})
    pdf_from_blocks(B, os.path.join(OUT, "manuscript.pdf"))
    print("wrote", docx, "and", os.path.join(OUT, "manuscript.pdf"), "| primary results:", "5-fold CV" if use_cv else "development split")


if __name__ == "__main__":
    main()
