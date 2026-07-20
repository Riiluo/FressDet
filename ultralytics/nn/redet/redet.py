# Use package-relative import to work under ultralytics.nn.redet
from .regconv import *
import torch
import torch.nn as nn

Act = nn.SiLU

"""
-REGCBA -> -REGConv + BN + Act
"""


class RELiftGCBA(nn.Module):
    def __init__(self, c1, c2, k=4, s=4, p=0):
        super(RELiftGCBA, self).__init__()
        self.gcv = RELift(c1, c2, k, s, p)
        self.bn3d = nn.BatchNorm3d(c2)
        self.act = Act()

    def forward(self, x):
        return self.act(self.bn3d(self.gcv(x)))


class EREGCBA(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super(EREGCBA, self).__init__()
        self.gcv = EREGConv(c1, c2, k, s, p)
        self.bn3d = nn.BatchNorm3d(c2)
        self.act = Act()

    def forward(self, x):
        return self.act(self.bn3d(self.gcv(x)))


class DREGCBA(nn.Module):
    """
    Depth-wise-only rotation-equivariant conv wrapper for (B,C,G,H,W).

    DREGCBA = DREGConv + BN3d + Act, and requires c1 == c2.
    """

    def __init__(self, c1, c2, k, s, p):
        super(DREGCBA, self).__init__()
        assert c1 == c2, f"DREGCBA requires c1 == c2 (depth-wise), got c1={c1}, c2={c2}"
        self.gcv = DREGConv(c1, c1, k, s, p)
        self.bn3d = nn.BatchNorm3d(c2)
        self.act = Act()

    def forward(self, x):
        return self.act(self.bn3d(self.gcv(x)))


class DS2GCBA(nn.Module):
    def __init__(self, c1, c2, k=2, s=2, p=0):
        super(DS2GCBA, self).__init__()
        self.gcv = DS2GConv(c1, c2, k, s, p)
        self.bn3d = nn.BatchNorm3d(c2)
        self.act = Act()

    def forward(self, x):
        return self.act(self.bn3d(self.gcv(x)))

class PREGCBA(nn.Module):
    def __init__(self, c1, c2):
        super(PREGCBA, self).__init__()
        self.gcv = PREGConv(c1, c2)
        self.bn3d = nn.BatchNorm3d(c2)
        self.act = Act()

    def forward(self, x):
        return self.act(self.bn3d(self.gcv(x)))


class MaxPooling(nn.Module):
    def __init__(self, k, s, p, g=g_order):
        super(MaxPooling, self).__init__()
        self.g = g
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.m(x)
        return x.view(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


class RESPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super(RESPPF, self).__init__()
        self.block = MaxPooling(k, 1, k // 2)
        self.gcv = PREGCBA(c1, c2)

    def forward(self, x):
        # channel split
        x = torch.chunk(x, chunks=4, dim=1)
        ys = []
        # RESPPF: only split the input channel for 4 parts
        for i in range(4):
            if i == 0:
                ys.append(x[i])
            elif i == 1:
                ys.append(self.block(x[i]))
            else:
                ys.append(self.block(ys[-1] + x[i]))
        # channel concat
        x = torch.cat(ys, dim=1)
        return self.gcv(x)
        # return x


#--------------------------------------------------------------------
class GroupLayerNorm3d(nn.Module):
    """
    LayerNorm over channel dimension for 5D tensors (B, C, G, H, W).
    实现参考 AAA-TiaoShi/20251109/20251109_re_swinv2.py。
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,G,H,W) -> (B,G,H,W,C) 做 LN 后再 permute 回来
        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
        y = self.ln(x_perm)
        return y.permute(0, 4, 1, 2, 3).contiguous()


class REMLP(nn.Module):
    """
    Equivariant MLP for (B,C,G,H,W), mirroring EfficientFormerV2 MLP with mid depthwise always on.
    Pipeline:
      PREGConv(C,4C) -> BN3d -> GELU ->
      DREGConv(4C,4C,3,1,1) -> BN3d -> GELU -> Drop(0) ->
      PREGConv(4C,C) -> BN3d -> GELU
    """
    def __init__(self, c: int, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden = 4 * c
        self.pw1 = PREGConv(c, hidden)
        self.bn1 = nn.BatchNorm3d(hidden)
        self.act1 = act_layer()

        self.dw = DREGConv(hidden, hidden, k=3, s=1, p=1)
        self.bn2 = nn.BatchNorm3d(hidden)
        self.act2 = act_layer()
        self.drop = nn.Dropout(p=drop) if drop and drop > 0 else nn.Identity()

        self.pw2 = PREGConv(hidden, c)
        self.bn3 = nn.BatchNorm3d(c)
        self.act3 = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,G,H,W)
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.pw2(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x


class PatchMerging(nn.Module):
    """
    Equivariant downsampling (V7DownSampling-style) for (B,C,G,H,W).
    Parallel branches with 2x2 avg pooling and PREGConv, then concat on channels.
    Output shape: (B, ouc, G, H/2, W/2).
    实现拷贝自 20251109_re_swinv2.py 的 PatchMerging。
    """
    def __init__(self, inc: int, ouc: int):
        super().__init__()
        half = ouc // 2
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # branch A: pool -> pw -> bn -> act
        self.a_pw = PREGConv(inc, half)
        self.a_bn = nn.BatchNorm3d(half)
        self.a_act = nn.SiLU()
        # branch B: pw -> bn -> act -> pool
        self.b_pw = PREGConv(inc, half)
        self.b_bn = nn.BatchNorm3d(half)
        self.b_act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,G,H,W)
        a = self.a_act(self.a_bn(self.a_pw(self.pool(x))))
        b = self.pool(self.b_act(self.b_bn(self.b_pw(x))))
        return torch.cat([a, b], dim=1)
