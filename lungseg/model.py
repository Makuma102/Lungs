"""Small 2D U-Net (Ronneberger et al., 2015) for lung / tumor segmentation.

Design choices (kept deliberately small for CPU training and reproducibility):
  * 4 resolution levels, base width 16 -> ~1.9M parameters.
  * Conv-InstanceNorm-LeakyReLU blocks (nnU-Net style; InstanceNorm is robust
    to the small batch sizes typical of medical imaging).
  * Transposed-conv upsampling and additive-free concatenative skip connections.
  * Optional deep supervision is omitted to keep the reference model minimal.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    """U-Net with `depth` downsamplings. Output: raw logits (N, num_classes, H, W).

    Classes: 0 = background, 1 = lung parenchyma, 2 = tumor / nodule.
    Input spatial size must be divisible by 2**depth.
    """

    def __init__(self, in_channels=1, num_classes=3, base=16, depth=4, dropout=0.1):
        super().__init__()
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_channels
        for w in widths[:-1]:
            self.enc.append(ConvBlock(c, w))
            c = w
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(widths[-2], widths[-1], dropout=dropout)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for w_in, w_out in zip(widths[::-1][:-1], widths[::-1][1:]):
            self.up.append(nn.ConvTranspose2d(w_in, w_out, 2, stride=2))
            self.dec.append(ConvBlock(w_out * 2, w_out))
        self.head = nn.Conv2d(widths[0], num_classes, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
