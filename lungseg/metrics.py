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


def lesion_detection(pred, gt, min_voxels=10):
    """Lesion-wise detection: a GT connected component counts as detected if any
    predicted voxel overlaps it. Returns (TP, FN, FP) in lesion counts."""
    gl, ng = ndimage.label(gt.astype(bool))
    pl, npred = ndimage.label(pred.astype(bool))
    tp = sum(1 for i in range(1, ng + 1) if (pred.astype(bool)[gl == i]).any())
    fp = 0
    for j in range(1, npred + 1):
        comp = pl == j
        if comp.sum() >= min_voxels and not gt.astype(bool)[comp].any():
            fp += 1
    return tp, ng - tp, fp
