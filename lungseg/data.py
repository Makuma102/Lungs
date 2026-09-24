"""Data: HU windowing, slice datasets, real-dataset readers and a CT phantom.

Label convention everywhere: 0 background, 1 lung, 2 tumor/nodule.

Real datasets supported (see scripts/download_datasets.py):
  * MSD Task06_Lung  (CT NIfTI, tumor labels; lungs derived by HU thresholding)
  * Kaggle kmader/finding-lungs-in-ct-data (CT slices + lung masks)
  * Kaggle nikhilpandey360/chest-xray-masks-and-labels (Montgomery/Shenzhen CXR lung masks)
"""
import glob
import os

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

HU_WINDOW = (-1000.0, 400.0)  # lung-inclusive window


def window_hu(vol, lo=HU_WINDOW[0], hi=HU_WINDOW[1]):
    return ((np.clip(vol, lo, hi) - lo) / (hi - lo)).astype(np.float32)


# --------------------------------------------------------------------------- phantom
def synth_ct_volume(shape=(32, 128, 128), rng=None, n_nodules=None, spacing=(2.5, 1.0, 1.0)):
    """Anatomically-inspired chest CT phantom in Hounsfield units.

    Body ellipse (soft tissue ~40 HU) with fat rim, two lung ellipsoids
    (~-850 HU, with vessel-like texture), spine (~700 HU), and 0-3 spherical /
    lobulated solid nodules (~30 HU, 3-15 mm radius) inside the lungs.
    Gaussian noise (sigma 20 HU) emulates quantum noise. Returns (hu, label).
    """
    rng = rng or np.random.default_rng()
    D, H, W = shape
    z, y, x = np.meshgrid(*(np.arange(s, dtype=np.float32) for s in shape), indexing="ij")
    zc, yc, xc = D / 2, H / 2, W / 2
    zn = (z - zc) / (D / 2)

    hu = np.full(shape, -1000.0, np.float32)  # air
    ry, rx = H * rng.uniform(0.36, 0.42), W * rng.uniform(0.44, 0.48)
    body = ((y - yc) / ry) ** 2 + ((x - xc) / rx) ** 2 <= 1
    fat = ((y - yc) / (ry * 0.92)) ** 2 + ((x - xc) / (rx * 0.92)) ** 2 > 1
    hu[body] = 40.0
    hu[body & fat] = -100.0

    label = np.zeros(shape, np.uint8)
    for side in (-1, 1):
        cx = xc + side * W * rng.uniform(0.19, 0.23)
        cy = yc - H * rng.uniform(0.0, 0.05)
        a = W * rng.uniform(0.13, 0.16)
        b = H * rng.uniform(0.24, 0.29)
        taper = np.clip(1 - 0.5 * zn ** 2, 0.2, 1)  # apex/base narrowing
        lung = ((x - cx) / (a * taper)) ** 2 + ((y - cy) / (b * taper)) ** 2 <= 1
        label[lung & body] = 1
    tex = ndimage.gaussian_filter(rng.normal(0, 1, shape).astype(np.float32), 1.2)
    lung_m = label == 1
    hu[lung_m] = -850.0 + 60.0 * tex[lung_m]
    vessels = (tex > 1.1 * tex[lung_m].std()) & lung_m
    hu[vessels] = -300.0  # bright vessel cross-sections (hard negatives for tumor)

    spine = ((y - (yc + ry * 0.7)) / (H * 0.07)) ** 2 + ((x - xc) / (W * 0.06)) ** 2 <= 1
    hu[spine] = 700.0

    if n_nodules is None:
        n_nodules = rng.integers(0, 4)
    lz, ly, lx = np.nonzero(lung_m)
    sz, sy, sx = (1.0 / s for s in spacing)  # voxels per mm
    for _ in range(n_nodules):
        if len(lz) == 0:
            break
        i = rng.integers(len(lz))
        r_mm = rng.uniform(3, 15)
        lob = 1 + 0.25 * np.sin(rng.uniform(2, 5) * np.arctan2(y - ly[i], x - lx[i]))  # spiculation
        d = np.sqrt(((z - lz[i]) / (r_mm * sz)) ** 2 + ((y - ly[i]) / (r_mm * sy * lob)) ** 2
                    + ((x - lx[i]) / (r_mm * sx * lob)) ** 2)
        nod = (d <= 1) & lung_m
        hu[nod] = 30.0 + 25.0 * tex[nod]
        label[nod] = 2

    hu += rng.normal(0, 20, shape).astype(np.float32)
    return hu, label


def synth_cxr(size=256, rng=None):
    """2D chest radiograph phantom (projection of a CT phantom along AP axis)."""
    rng = rng or np.random.default_rng()
    hu, lab = synth_ct_volume((size, size, 48), rng=rng, spacing=(1.0, 1.0, 2.5))
    att = np.clip(hu + 1000, 0, None).sum(-1)
    img = (att - att.min()) / (np.ptp(att) + 1e-6)
    label = np.zeros((size, size), np.uint8)
    label[(lab == 1).any(-1)] = 1
    label[(lab == 2).any(-1)] = 2
    return img.astype(np.float32), label


# --------------------------------------------------------------------------- datasets
class SliceDataset(Dataset):
    """Axial 2D slices from a list of (image [D,H,W] in [0,1], label [D,H,W]) volumes."""

    def __init__(self, volumes, augment=False, tumor_oversample=0.5, seed=0):
        self.volumes = volumes
        self.index = [(v, s) for v, (img, _) in enumerate(volumes) for s in range(img.shape[0])]
        self.tumor_idx = [(v, s) for v, s in self.index if (volumes[v][1][s] == 2).any()]
        self.augment = augment
        self.tumor_oversample = tumor_oversample if self.tumor_idx else 0.0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if self.augment and self.rng.random() < self.tumor_oversample:
            v, s = self.tumor_idx[self.rng.integers(len(self.tumor_idx))]
        else:
            v, s = self.index[i]
        img, lab = self.volumes[v][0][s].copy(), self.volumes[v][1][s].copy()
        if self.augment:
            img, lab = augment2d(img, lab, self.rng)
        return torch.from_numpy(img[None]).float(), torch.from_numpy(lab.astype(np.int64))


def augment2d(img, lab, rng):
    if rng.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    if rng.random() < 0.5:
        ang = rng.uniform(-15, 15)
        img = ndimage.rotate(img, ang, reshape=False, order=1, mode="nearest")
        lab = ndimage.rotate(lab, ang, reshape=False, order=0, mode="nearest")
    if rng.random() < 0.5:  # intensity: gamma + contrast + noise
        img = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.5)
        img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)
        img = img + rng.normal(0, 0.02, img.shape)
    return np.ascontiguousarray(img, np.float32), np.ascontiguousarray(lab)


def make_phantom_volumes(n, shape=(32, 128, 128), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        hu, lab = synth_ct_volume(shape, rng=rng)
        out.append((window_hu(hu), lab))
    return out


# --------------------------------------------------------------------------- real data
def lungs_from_hu(hu):
    """Classical lung mask (threshold -320 HU, drop outside-body air, fill holes)."""
    air = hu < -320
    lab, _ = ndimage.label(air)
    border = np.unique(np.concatenate([lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    air[np.isin(lab, border)] = False
    lab, n = ndimage.label(air)
    if n == 0:
        return air
    sizes = ndimage.sum(air, lab, range(1, n + 1))
    keep = np.argsort(sizes)[::-1][:2] + 1
    lungs = np.isin(lab, keep)
    return np.stack([ndimage.binary_fill_holes(s) for s in lungs])


def _resize_stack(arr, size, order):
    d, h, w = arr.shape
    return ndimage.zoom(arr, (1, size / h, size / w), order=order)


def load_msd_lung(root, size=256, max_cases=None):
    """Medical Segmentation Decathlon Task06_Lung. Returns volumes + spacings."""
    import nibabel as nib  # optional dependency

    vols, spacings = [], []
    images = sorted(glob.glob(os.path.join(root, "imagesTr", "lung_*.nii.gz")))[:max_cases]
    for p in images:
        img = nib.load(p)
        hu = np.transpose(img.get_fdata().astype(np.float32), (2, 1, 0))
        tum = np.transpose(nib.load(p.replace("imagesTr", "labelsTr")).get_fdata() > 0, (2, 1, 0))
        lab = np.zeros(hu.shape, np.uint8)
        lab[lungs_from_hu(hu)] = 1
        lab[tum] = 2
        z = img.header.get_zooms()
        spacings.append((z[2], z[1] * hu.shape[1] / size, z[0] * hu.shape[2] / size))
        vols.append((_resize_stack(window_hu(hu), size, 1), _resize_stack(lab, size, 0)))
    return vols, spacings


def load_png_pairs(image_dir, mask_dir, size=256, mask_suffix=""):
    """2D pairs (CXR or CT slices), e.g. Montgomery/Shenzhen or kmader. Lung label=1."""
    from PIL import Image

    out = []
    for p in sorted(glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.tif"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        cands = glob.glob(os.path.join(mask_dir, stem + mask_suffix + ".*"))
        if not cands:
            continue
        img = np.asarray(Image.open(p).convert("L").resize((size, size)), np.float32) / 255.0
        m = np.asarray(Image.open(cands[0]).convert("L").resize((size, size), Image.NEAREST)) > 127
        out.append((img[None], m[None].astype(np.uint8)))
    return out
