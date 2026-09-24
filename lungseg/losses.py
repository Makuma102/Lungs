"""Compound Dice + cross-entropy loss (standard in nnU-Net / MSD baselines)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=3, ce_weight=None, smooth=1e-5, include_background=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background
        self.register_buffer("ce_weight", None if ce_weight is None else torch.as_tensor(ce_weight, dtype=torch.float32))

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.ce_weight)
        probs = logits.softmax(1)
        onehot = F.one_hot(target, self.num_classes).movedim(-1, 1).float()
        start = 0 if self.include_background else 1
        p, g = probs[:, start:], onehot[:, start:]
        # batch-level soft Dice: stable when a class is absent in some slices
        dims = (0,) + tuple(range(2, logits.ndim))  # works for 2D and 3D
        inter = (p * g).sum(dims)
        denom = p.sum(dims) + g.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return ce + (1 - dice.mean())
