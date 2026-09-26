"""
medforget/models/resnet18.py
----------------------------
ResNet-18 adapted for small (28×28) medical images.

The standard torchvision ResNet-18 is designed for 224×224 ImageNet input.
Two architectural modifications make it suitable for 28×28 images:

  1. The first conv layer (7×7, stride 2) is replaced with a 3×3, stride 1
     conv — this prevents feature maps from becoming too small too quickly.
  2. The initial max-pool is removed (replaced with nn.Identity) — with
     only 28×28 pixels there is not enough spatial resolution to pool here.

The final FC layer is also replaced to output `num_classes` logits instead
of the default 1000 (ImageNet).

No pretrained weights are used — the model trains from scratch, keeping
experiments clean and reproducible across different hardware environments.

Reference:
    He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
"""

import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes: int) -> nn.Module:
    """Build a ResNet-18 adapted for 28×28 RGB input.

    Args:
        num_classes: Number of output classes.  Pass 9 for PathMNIST.

    Returns:
        An untrained nn.Module ready for optimizer.step() loops.
    """
    model = resnet18(weights=None)

    # ── Modification 1: smaller first conv for small images ───────────────────
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Replaced: Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # ── Modification 2: remove aggressive early max-pool ─────────────────────
    # At 28×28, the max-pool would immediately shrink to 14×14 then 7×7,
    # losing too much spatial information.  Replace with a no-op.
    model.maxpool = nn.Identity()

    # ── Modification 3: replace classification head ───────────────────────────
    # Original: Linear(512, 1000)  →  Replaced: Linear(512, num_classes)
    model.fc = nn.Linear(512, num_classes)

    return model
