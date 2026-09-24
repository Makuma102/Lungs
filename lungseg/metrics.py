"""Segmentation and detection metrics reported in the paper-style evaluation."""
import numpy as np
from scipy import ndimage


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(pred, gt).sum() / s


def iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    u = np.logical_or(pred, gt).sum()
    return 1.0 if u == 0 else np.logical_and(pred, gt).sum() / u


def sensitivity(pred, gt):
    gt = gt.astype(bool)
    return float("nan") if gt.sum() == 0 else np.logical_and(pred.astype(bool), gt).sum() / gt.sum()


def precision(pred, gt):
    pred = pred.astype(bool)
    return float("nan") if pred.sum() == 0 else np.logical_and(pred, gt.astype(bool)).sum() / pred.sum()


def _surface(mask):
    return mask ^ ndimage.binary_erosion(mask)


def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """95th-percentile symmetric Hausdorff distance in physical units (mm)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("inf")
    spacing = spacing[-pred.ndim:]
    dt_gt = ndimage.distance_transform_edt(~gt, sampling=spacing)
    dt_pr = ndimage.distance_transform_edt(~pred, sampling=spacing)
    d = np.concatenate([dt_gt[_surface(pred)], dt_pr[_surface(gt)]])
    return float(np.percentile(d, 95))


def lesion_detection(pred, gt, min_voxels=10, spacing=None, min_diameter_mm=None, min_overlap_frac=0.0):
    """Lesion-wise detection. Connected components smaller than the size
    threshold are ignored on BOTH sides (expert labels contain annotation specks
    of a few voxels that are not lesions). The threshold is `min_voxels`, or, if
    `spacing` and `min_diameter_mm` are given, the volume of a sphere with that
    diameter (3 mm is the LIDC/LUNA nodule convention).
    A GT lesion is detected if the prediction covers more than `min_overlap_frac`
    of its volume (0 = any overlap, which is lenient: a 1% touch would count);
    a predicted component is a false positive if it overlaps no GT voxel.
    Returns (TP, FN, FP) in lesion counts."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if spacing is not None and min_diameter_mm is not None:
        min_vox = (np.pi / 6 * min_diameter_mm ** 3) / float(np.prod(spacing[-gt.ndim:]))
    else:
        min_vox = min_voxels
    gl, ng = ndimage.label(gt)
    pl, npred = ndimage.label(pred)
    gsize = ndimage.sum(gt, gl, range(1, ng + 1)) if ng else []
    psize = ndimage.sum(pred, pl, range(1, npred + 1)) if npred else []
    tp = fn = fp = 0
    for i in range(1, ng + 1):
        if gsize[i - 1] < min_vox:
            continue
        covered = pred[gl == i].sum() / gsize[i - 1]
        if covered > min_overlap_frac:
            tp += 1
        else:
            fn += 1
    for j in range(1, npred + 1):
        if psize[j - 1] >= min_vox and not gt[pl == j].any():
            fp += 1
    return tp, fn, fp
