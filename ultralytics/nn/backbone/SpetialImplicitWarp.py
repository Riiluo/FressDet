import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.redet.regconv import PREGConv, g_order
from ultralytics.nn.redet.redet import GroupLayerNorm3d, REMLP


class PointOffsetMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2, depth=2):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        layers = []
        for i in range(depth - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, out_dim))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return self.mlp(x)


class SpectralCoordEncoder(nn.Module):
    def __init__(
        self,
        num_freqs=6,
        include_input=True,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        # Fixed NeRF-style Fourier feature mapping:
        # psi(lambda) = [sin(2^k * pi * lambda), cos(2^k * pi * lambda)].
        self.scale = math.pi
        if num_freqs <= 0:
            self.out_dim = 1 if include_input else 0
        else:
            self.out_dim = (1 if include_input else 0) + 2 * num_freqs

    def forward(self, coord):
        # coord: (..., 1)
        if self.num_freqs <= 0:
            if self.include_input:
                return coord
            return coord.new_zeros(*coord.shape[:-1], 0)
        k = torch.arange(self.num_freqs, device=coord.device, dtype=coord.dtype)
        freqs = (2.0**k) * self.scale
        coord_1d = coord.squeeze(-1)
        angles = coord_1d[..., None] * freqs
        sin, cos = angles.sin(), angles.cos()
        out = torch.cat([sin, cos], dim=-1)
        if self.include_input:
            out = torch.cat([coord_1d.unsqueeze(-1), out], dim=-1)
        return out


class SpectralSimplicitWarp(nn.Module):
    def __init__(
        self,
        channels,
        hidden=64,
        g=g_order,
        num_freqs=6,
        include_input=True,
        mlp_depth=3,
    ):
        super().__init__()
        self.g = g
        self.coord_encoder = SpectralCoordEncoder(
            num_freqs=num_freqs,
            include_input=include_input,
        )
        self.rank = max(4, hidden // 8)
        a_in_dim = hidden
        b_in_dim = self.coord_encoder.out_dim
        self.a_proj = nn.Linear(a_in_dim, self.rank)
        self.b_mlp = PointOffsetMLP(b_in_dim, hidden=hidden, out_dim=self.rank, depth=mlp_depth)
        # Original full MLP (no coord/cell) used for FLOPs test:
        # in_dim = hidden
        # self.offset_mlp = PointOffsetMLP(in_dim, hidden=hidden, out_dim=1, depth=mlp_depth)
        self.feat_proj = PREGConv(channels, hidden, g=g)
        self.proj = PREGConv(channels, channels, g=g)

    def forward(self, x):
        bs, c, g, h, w = x.shape
        if g != self.g:
            raise ValueError(f"group mismatch: got {g}, expected {self.g}")

        feat = self.feat_proj(x)
        feat_hw = feat.permute(0, 2, 3, 4, 1)  # (B, G, H, W, hidden)
        a_inp = feat_hw
        a = self.a_proj(a_inp.reshape(-1, a_inp.shape[-1])).view(bs, g, h, w, self.rank)

        spe_coord = torch.linspace(-1.0, 1.0, steps=c, device=x.device, dtype=x.dtype).view(c, 1)
        b_inp = self.coord_encoder(spe_coord)
        b_basis = self.b_mlp(b_inp)  # (C, rank)

        # Monotonic (no-folding) spectral warp by construction.
        #
        # Instead of predicting an unconstrained offset u(s), we predict positive increments and integrate
        # them along the channel axis to obtain a strictly increasing mapping t(c) in index space.
        #
        # This gives a clean "continuous + invertible (1D)" story without needing extra regularizer losses.
        raw = torch.einsum("bghwr,cr->bcghw", a, b_basis)  # (B,C,G,H,W)
        step = F.softplus(raw) + 1e-6  # strictly positive => monotonic t(c)
        t = torch.cumsum(step, dim=1)
        t = t - t[:, :1]  # start at 0
        denom = t[:, -1:] + 1e-6
        t = t / denom * float(c - 1)  # end at (C-1)
        t = torch.nan_to_num(t, nan=0.0, posinf=float(c - 1), neginf=0.0).clamp(0.0, float(c - 1))

        t0 = t.floor()
        t1 = (t0 + 1).clamp(0, c - 1)
        w1 = t - t0
        w0 = 1.0 - w1
        t0 = t0.long()
        t1 = t1.long()

        x0 = torch.gather(x, 1, t0)
        x1 = torch.gather(x, 1, t1)
        feat_warp = w0 * x0 + w1 * x1

        return self.proj(feat_warp)


class SpeIWMetaformerStage(nn.Module):
    def __init__(
        self,
        channels,
        hidden=64,
        g=g_order,
        num_freqs=16,
        mlp_depth=5,
    ):
        super().__init__()
        self.norm1 = GroupLayerNorm3d(channels)
        self.warp = SpectralSimplicitWarp(
            channels=channels,
            hidden=hidden,
            g=g,
            num_freqs=num_freqs,
            include_input=True,
            mlp_depth=mlp_depth,
        )
        self.norm2 = GroupLayerNorm3d(channels)
        self.mlp = REMLP(c=channels)

    def forward(self, x):
        y = self.warp(self.norm1(x))
        x = x + y
        z = self.mlp(self.norm2(x))
        return x + z
