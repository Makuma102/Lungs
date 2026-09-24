"""Validity tests: guard against *false success*.

A high score is only meaningful if (a) the metric cannot be gamed by trivial
predictors, (b) there is no train/test leakage, (c) geometry/orientation is
right, and (d) a model that learned nothing scores near zero. These tests
check each of those. Tests that need the real MSD data skip when it is absent.
"""
import os

import numpy as np
import pytest
import torch

from lungseg import data as D
from lungseg.metrics import dice, hd95, iou, lesion_detection, precision, sensitivity
from lungseg.model import SmallUNet, SmallUNet3D
from lungseg.reconstruct3d import nodule_report, postprocess, predict_volume
from lungseg.train import evaluate, load, summarize

MSD = "data/Task06_Lung"
PREP = "data/prep"
need_msd = pytest.mark.skipif(not os.path.exists(os.path.join(PREP, "lung_004.npz")), reason="MSD data not present")


# ------------------------------------------------------------------ metric edge cases
def _ball(shape, c, r):
    z, y, x = np.ogrid[tuple(slice(0, s) for s in shape)]
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r


def test_empty_prediction_scores_zero_not_nan():
    gt = _ball((20, 20, 20), (10, 10, 10), 4)
    empty = np.zeros_like(gt)
    assert dice(empty, gt) == 0.0 and iou(empty, gt) == 0.0
    assert sensitivity(empty, gt) == 0.0
    assert hd95(empty, gt) == float("inf")  # never silently 0


def test_both_empty_is_perfect_but_excluded_from_tumor_mean():
    """Dice(empty, empty)=1 is correct per-case, but tumor-free cases must not
    inflate the tumor Dice mean (summarize() averages tumor-positive cases only)."""
    assert dice(np.zeros((5, 5)), np.zeros((5, 5))) == 1.0
    rows = [
        {"lung_dice": 1, "lung_iou": 1, "lung_hd95": 0, "lung_sens": 1, "lung_prec": 1,
         "tumor_dice": 1.0, "tumor_iou": 1, "tumor_hd95": 0, "tumor_sens": np.nan, "tumor_prec": np.nan,
         "tp": 0, "fn": 0, "fp": 0, "has_tumor_gt": False, "has_tumor_pred": False},
        {"lung_dice": 1, "lung_iou": 1, "lung_hd95": 0, "lung_sens": 1, "lung_prec": 1,
         "tumor_dice": 0.2, "tumor_iou": .1, "tumor_hd95": 9, "tumor_sens": .2, "tumor_prec": .2,
         "tp": 1, "fn": 0, "fp": 0, "has_tumor_gt": True, "has_tumor_pred": True},
    ]
    s = summarize(rows)
    assert s["tumor_dice"]["mean"] == pytest.approx(0.2)  # not 0.6


def test_specificity_undefined_without_negatives():
    """With no tumor-free scans, specificity is undefined - reporting 0 (or 1)
    would be a fabricated number."""
    base = {"lung_dice": 1, "lung_iou": 1, "lung_hd95": 0, "lung_sens": 1, "lung_prec": 1,
            "tumor_dice": .5, "tumor_iou": .3, "tumor_hd95": 5, "tumor_sens": .5, "tumor_prec": .5,
            "tp": 1, "fn": 0, "fp": 0, "has_tumor_gt": True, "has_tumor_pred": True}
    s = summarize([base, dict(base)])
    assert np.isnan(s["case_specificity"]) and s["case_sensitivity"] == 1.0


def test_all_foreground_predictor_is_penalised():
    gt = _ball((32, 32, 32), (16, 16, 16), 3)
    everything = np.ones_like(gt)
    assert sensitivity(everything, gt) == 1.0
    assert precision(everything, gt) < 0.01 and dice(everything, gt) < 0.02


def test_hd95_is_physical_and_detects_shift():
    a = _ball((40, 40, 40), (20, 20, 20), 5)
    b = np.roll(a, 4, axis=2)
    assert hd95(a, b, (1, 1, 1)) == pytest.approx(4, abs=1.01)
    assert hd95(a, b, (1, 1, 2.5)) == pytest.approx(10, abs=2.6)  # scales with spacing


def test_lesion_detection_counts_fp_and_fn():
    gt = _ball((40, 40, 40), (10, 10, 10), 3) | _ball((40, 40, 40), (30, 30, 30), 3)
    pred = _ball((40, 40, 40), (10, 10, 10), 3) | _ball((40, 40, 40), (30, 10, 10), 3)
    assert lesion_detection(pred, gt) == (1, 1, 1)  # one hit, one missed, one false alarm


# ------------------------------------------------------------------ negative controls
def _phantom_args(**kw):
    a = dict(data="phantom", size=64, depth_slices=16, n_train=4, n_val=1, n_test=4, seed=7,
             root="", max_cases=None)
    a.update(kw)
    return type("A", (), a)


def test_untrained_model_scores_near_zero_on_tumor():
    """If an untrained network scored well, the metric/pipeline would be broken."""
    torch.manual_seed(0)
    _, _, (tev, tes) = load(_phantom_args())
    s = summarize(evaluate(SmallUNet(base=8), tev, tes, "cpu"))
    assert s.get("tumor_dice", {"mean": 0})["mean"] < 0.2


def test_constant_background_predictor_gets_zero_tumor_dice():
    class Zero(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], 3, *x.shape[2:])
            out[:, 0] = 10
            return out
    _, _, (tev, tes) = load(_phantom_args())
    rows = evaluate(Zero(), tev, tes, "cpu")
    s = summarize(rows)
    if any(r["has_tumor_gt"] for r in rows):
        assert s["tumor_dice"]["mean"] == 0.0
        assert s["lesion_sensitivity"] == 0.0
        assert s["case_sensitivity"] == 0.0


def test_mismatched_ground_truth_gives_low_dice():
    """Evaluating a case's prediction against another case's label must fail:
    guards against a metric that rewards generic 'tumor-like' blobs."""
    rng = np.random.default_rng(3)
    a = b = None
    while a is None or not (a == 2).any() or not (b == 2).any():
        _, a = D.synth_ct_volume((16, 64, 64), rng=rng, n_nodules=1)
        _, b = D.synth_ct_volume((16, 64, 64), rng=rng, n_nodules=1)
    assert dice(a == 2, a == 2) == 1.0
    assert dice(a == 2, b == 2) < 0.3


# ------------------------------------------------------------------ leakage / splits
def test_case_level_split_is_disjoint_and_deterministic():
    args = _phantom_args(n_train=10, n_val=2, n_test=3)
    load(args)
    sp = args.split
    tr, va, te = map(set, (sp["train"], sp["val"], sp["test"]))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == 15
    args2 = _phantom_args(n_train=10, n_val=2, n_test=3)
    load(args2)
    assert args2.split == sp


def test_slice_dataset_never_mixes_volumes():
    vols = D.make_phantom_volumes(2, (4, 32, 32), seed=0)
    ds = D.SliceDataset(vols)
    assert len(ds) == 8 and {v for v, _ in ds.index} == {0, 1}


def test_patch_oversampling_returns_tumor():
    vols = D.make_phantom_volumes(3, (16, 64, 64), seed=5)
    vols = [v for v in vols if (v[1] == 2).any()]
    ds = D.PatchDataset3D(vols, patch=(8, 32, 32), tumor_oversample=1.0, augment=False)
    hits = sum(bool((ds[i][1] == 2).any()) for i in range(len(ds)))
    assert hits == len(ds)


# ------------------------------------------------------------------ geometry
def test_nodule_volume_and_diameter_physical():
    lab = np.zeros((40, 80, 80), np.uint8)
    lab[_ball(lab.shape, (20, 40, 40), 10)] = 2  # r=10 vox
    r = nodule_report(lab, (1.0, 0.5, 0.5))  # anisotropic spacing
    true_ml = 4 / 3 * np.pi * 10 ** 3 * (1.0 * 0.5 * 0.5) / 1000
    assert r["nodules"][0]["volume_ml"] == pytest.approx(true_ml, rel=0.05)


def test_postprocess_removes_specks_and_extrapulmonary_tumor():
    lab = np.zeros((20, 60, 60), np.uint8)
    lab[5:15, 10:30, 10:25] = 1
    lab[5:15, 10:30, 35:50] = 1
    lab[8:12, 15:20, 15:20] = 2           # real tumor inside lung
    lab[1, 55, 55] = 2                     # 1-voxel speck outside lungs
    out = postprocess(lab, (1, 1, 1), min_tumor_mm3=5)
    assert out[1, 55, 55] == 0 and (out[8:12, 15:20, 15:20] == 2).all()


class _LocalModel(torch.nn.Module):
    """Voxel-wise model (no spatial context, no normalisation): tiled and whole
    inference must then agree exactly, so any mismatch is a tiling bug
    (offsets, seams, weighting). A real U-Net differs slightly because
    InstanceNorm uses per-patch statistics - that is expected, not a bug."""
    pools = [(2, 2, 2), (2, 2, 2)]

    def forward(self, x):
        return torch.cat([4 * (0.5 - x), 4 * (x - 0.5), 8 * (x - 0.8)], 1)


def test_sliding_window_equals_whole_volume():
    from lungseg.reconstruct3d import _predict_3d
    vol = D.make_phantom_volumes(1, (30, 70, 66), seed=1)[0][0]  # awkward, non-divisible shape
    whole, pw = _predict_3d(_LocalModel(), vol, "cpu", patch=(64, 128, 128))
    tiled, pt = _predict_3d(_LocalModel(), vol, "cpu", patch=(16, 32, 32), overlap=0.5)
    assert whole.shape == tiled.shape == vol.shape
    assert np.allclose(pw, pt, atol=1e-5) and (whole == tiled).all()


# ------------------------------------------------------------------ real data (skipped if absent)
@need_msd
def test_msd_prep_roundtrip_preserves_tumor():
    import nibabel as nib
    from lungseg.prep_msd import to_original_nifti
    d = np.load(os.path.join(PREP, "lung_004.npz"))
    img = to_original_nifti(d["lab"] == 2, os.path.join(PREP, "lung_004.npz"),
                            os.path.join(MSD, "imagesTr", "lung_004.nii.gz"))
    gt = np.asanyarray(nib.load(os.path.join(MSD, "labelsTr", "lung_004.nii.gz")).dataobj) > 0
    assert dice(np.asanyarray(img.dataobj) > 0, gt) > 0.85


@need_msd
def test_msd_prep_label_semantics():
    d = np.load(os.path.join(PREP, "lung_004.npz"))
    img, lab = d["img"].astype(np.float32), d["lab"]
    assert set(np.unique(lab)) <= {0, 1, 2} and (lab == 2).any()
    assert 0 <= img.min() and img.max() <= 1
    # lung parenchyma is dark (air), tumor is soft tissue: windowed values must differ strongly
    assert img[lab == 1].mean() < 0.25 and img[lab == 2].mean() > 0.5
    assert np.allclose(d["spacing"], 1.5)


@need_msd
def test_mesh_is_in_patient_space_and_tumor_is_in_lungs():
    """Orientation check: the tumor mesh centroid must lie inside the lobe
    meshes' bounding box and match the label centroid in RAS mm."""
    import json
    rec = "data/recon/lung_004/anatomy.json"
    if not os.path.exists(rec):
        pytest.skip("reconstruction not built")
    import nibabel as nib
    from scipy import ndimage
    r = json.load(open(rec))
    img = nib.load(os.path.join(MSD, "labelsTr", "lung_004.nii.gz"))
    c = nib.affines.apply_affine(img.affine, ndimage.center_of_mass(np.asanyarray(img.dataobj) > 0))
    reported = np.array(r["tumor_gt"]["lesions"][0]["centroid_ras_mm"])
    assert np.linalg.norm(c - reported) < 3.0
    assert r["tumor_gt"]["lobe"].startswith("lung_")
