# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolutional primitives required by FressDet.yaml."""

import torch
import torch.nn as nn

__all__ = ("Conv", "Concat")


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BatchNorm and SiLU activation."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Concat(nn.Module):
    """Concatenate tensors along a dimension."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # thop 等工具在做 FLOPs 统计时会把单个 Tensor 直接传进来，
        # 这里兼容两种情况：正常训练/推理时传 list/tuple，profiling 时传 Tensor。
        if isinstance(x, (list, tuple)):
            return torch.cat(x, self.d)
        return x
