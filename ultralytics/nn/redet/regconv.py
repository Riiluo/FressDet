import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")  # remove warning


def strictly_rot(x, i, g):
    if len(x.shape) >= 5:
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
    theta = 2 * np.pi / g * i
    rot_mat = torch.Tensor([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0]]).to(dtype=x.dtype, device=x.device)
    rot_mat = rot_mat.repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size())
    x = F.grid_sample(x, grid)
    return x.view(x.shape[0], -1, g, x.shape[-2], x.shape[-1]) if len(x.shape) >= 5 else x


g_order = 4


class RELift(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=g_order):
        super(RELift, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        self.w = nn.Parameter(torch.empty(c2, c1, k, k))
        nn.init.kaiming_uniform_(self.w, a=(5 ** 0.5))

    def build_filters(self):
        rotated_filters = [strictly_rot(self.w, i, self.g) for i in range(self.g)]
        rotated_filters = torch.stack(rotated_filters, dim=1)  # 16,4,8,3,3
        return rotated_filters.view(self.c2 * self.g, self.c1, self.k, self.k)

    def forward(self, x):
        x = torch.conv2d(
            x,
            self.build_filters(),
            stride=self.s,
            padding=self.p,
            bias=None
        )
        return x.view(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


# Point-wise operator for REGConv
class PREGConv(nn.Module):
    def __init__(self, c1, c2, g=g_order):
        super(PREGConv, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = 1
        self.s = 1
        self.p = 0
        self.g = g
        self.pw = nn.Parameter(torch.empty(c2, c1, g, 1, 1))
        nn.init.kaiming_uniform_(self.pw, a=(5 ** 0.5))

    def build_pw_filters(self):
        rotated_filters = []
        for i in range(self.g):
            rotated_filters.append(torch.roll(self.pw, i, dims=-3))
            # rotated_filters.append(self.pw)
        rotated_filters = torch.stack(rotated_filters, dim=1)
        return rotated_filters.view(self.c2 * self.g, self.c1 * self.g, 1, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = torch.conv2d(
            x,
            self.build_pw_filters(),
            bias=None
        )
        return x.reshape(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


# Depth-wise operator for REGConv
class DREGConv(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=g_order):
        super(DREGConv, self).__init__()
        assert c1 == c2
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        self.groups = c2 * g
        self.dw = nn.Parameter(torch.empty(c2, 1, k, k))
        nn.init.kaiming_uniform_(self.dw, a=(5 ** 0.5))

    def build_dw_filters(self):
        rotated_filters = [strictly_rot(self.dw, i, self.g) for i in range(self.g)]
        rotated_filters = torch.stack(rotated_filters, dim=1)
        return rotated_filters.view(self.c2 * self.g, 1, self.k, self.k)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = torch.conv2d(
            x,
            self.build_dw_filters(),
            stride=self.s,
            padding=self.p,
            groups=self.groups,
            bias=None
        )
        return x.reshape(x.shape[0], -1, self.g, x.shape[-2], x.shape[-1])


# EREGConv (Efficient REGConv) = PREGConv + DREGConv, i.e., Point-wise + Depth-wise operators for REGConv
class EREGConv(torch.nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super(EREGConv, self).__init__()
        self.conv = nn.Sequential(
            PREGConv(c1, c2),  # 
            DREGConv(c2, c2, k, s, p)  # 
        )

    def forward(self, x):
        return self.conv(x)


class DS2GConv(torch.nn.Module):  # c1要等于c2
    def __init__(self, c1, c2, k, s, p):
        super(DS2GConv, self).__init__()
        self.conv = nn.Sequential(
            DREGConv(c1, c1, k, s, p),
            PREGConv(c1, c2),
        )

    def forward(self, x):
        return self.conv(x)


# Single transposed group conv with rotated kernels (Rg -> Rg upsample).
class AREGUp(nn.Module):
    def __init__(self, c1, c2, k=2, s=2, p=0, g=g_order, use_bn=True, use_act=True):
        super(AREGUp, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        self.groups = g
        # self.use_bn = use_bn
        # self.use_act = use_act
        # self.bn = nn.BatchNorm3d(c2) if use_bn else nn.Identity()
        # self.act = nn.SiLU() if use_act else nn.Identity()
        # base kernel (c2, c1, k, k), rotated per group then used in conv_transpose2d
        self.w = nn.Parameter(torch.empty(c2, c1, k, k))
        nn.init.kaiming_uniform_(self.w, a=(5 ** 0.5))

    def build_filters(self):
        rotated = [strictly_rot(self.w, i, self.g) for i in range(self.g)]
        rotated = torch.stack(rotated, dim=0)  # (g, c2, c1, k, k)
        rotated = rotated.permute(0, 2, 1, 3, 4).contiguous()  # (g, c1, c2, k, k)
        return rotated.view(self.g * self.c1, self.c2, self.k, self.k)

    def forward(self, x):
        assert x.dim() == 5, f"AREGUp expects 5D input (B,C,G,H,W), got {x.shape}"
        assert x.shape[1] == self.c1 and x.shape[2] == self.g, (
            f"AREGUp got C={x.shape[1]}, G={x.shape[2]} but expected C={self.c1}, G={self.g}"
        )

        B, _, _, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B, self.g * self.c1, H, W)
        z = torch.conv_transpose2d(
            x_flat,
            self.build_filters(),
            stride=self.s,
            padding=self.p,
            groups=self.groups,
            bias=None,
        )
        B, OC, H2, W2 = z.shape
        y = z.view(B, self.g, self.c2, H2, W2).transpose(1, 2).contiguous()
        # y = self.act(self.bn(y))  # removed BN+SiLU after AREGUp
        return y

class TransferFlatten(nn.Module):
    def __init__(self):
        super(TransferFlatten, self).__init__()
    
    def forward(self, x):
        # x: (B, C, G, H, W) -> (B, C, H, W)
        assert x.dim() == 5, f"rans expects 5D input (B,C,G,H,W), got {x.shape}"
        return x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
