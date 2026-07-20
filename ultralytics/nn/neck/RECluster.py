from __future__ import annotations

"""ReCoW neck block for rotation-equivariant group features.

The module expects features in (B, C, G, H, W) format and retains the
spectral/spatial routing implementation used by the reported FressDet model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ultralytics.nn.redet.regconv import g_order

try:
    from ultralytics.nn.redet.redet import DREGCBA, PREGCBA
except Exception:  # pragma: no cover
    DREGCBA = None
    PREGCBA = None


def _ceil_to_multiple(value: int, base: int) -> int:
    return ((value + base - 1) // base) * base


class ReCoWSpectralBranch(nn.Module):
    """Soft spectral routing over local prototypes."""

    def __init__(
        self,
        win_size: int = 4,
        proposal_w: int = 2,
        proposal_h: int = 2,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.proposal_w = proposal_w
        self.proposal_h = proposal_h
        self.num_centers = proposal_w * proposal_h
        self.use_residual = use_residual
        self.logit_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = F.normalize(x1, dim=-1, eps=1e-12)
        x2 = F.normalize(x2, dim=-1, eps=1e-12)
        return torch.matmul(x1, x2.transpose(-2, -1))

    def _cluster_window(self, x_win_all_groups: torch.Tensor) -> torch.Tensor:
        b, c, g, wh, ww = x_win_all_groups.shape
        x_pooled = x_win_all_groups.mean(dim=2)

        centers_grid = F.adaptive_avg_pool2d(x_pooled, (self.proposal_h, self.proposal_w))
        centers_flat = rearrange(centers_grid, "b c h w -> b (h w) c")
        pix_pooled = rearrange(x_pooled, "b c h w -> b (h w) c")

        cos = self._pairwise_cos_sim(centers_flat, pix_pooled)
        scale = self.logit_scale.exp().clamp(max=100.0)
        prob = F.softmax(scale * cos, dim=1)

        agg_flat = torch.matmul(prob.transpose(1, 2), centers_flat)

        conf_assign = (prob * cos).sum(dim=1).clamp(-1.0, 1.0)
        conf_assign = (conf_assign + 1.0) * 0.5
        conf_agree = F.cosine_similarity(agg_flat, pix_pooled, dim=-1).clamp(-1.0, 1.0)
        conf_agree = (conf_agree + 1.0) * 0.5
        agg_flat = agg_flat * (conf_assign * conf_agree).unsqueeze(-1)

        agg = rearrange(agg_flat, "b (h w) c -> b c h w", h=wh, w=ww)
        return agg.unsqueeze(2).expand(b, c, g, wh, ww)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, f"ReCoWSpectralBranch expects (B,C,G,H,W), got {x.shape}"
        b, _, g, h, w = x.shape
        assert g == g_order, f"Expected group order {g_order}, got {g}"

        k = self.win_size
        h_pad = _ceil_to_multiple(h, k)
        w_pad = _ceil_to_multiple(w, k)
        pad_h = h_pad - h
        pad_w = w_pad - w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        pad_bottom = pad_h - pad_top
        pad_right = pad_w - pad_left

        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom)) if pad_h or pad_w else x

        hn = h_pad // k
        wn = w_pad // k
        x_windows = rearrange(
            x_pad,
            "b c g (hn wh) (wn ww) -> (b hn wn) c g wh ww",
            hn=hn,
            wn=wn,
            wh=k,
            ww=k,
        )
        agg_windows = self._cluster_window(x_windows)
        y_windows = x_windows + agg_windows if self.use_residual else agg_windows
        y_pad = rearrange(
            y_windows,
            "(b hn wn) c g wh ww -> b c g (hn wh) (wn ww)",
            b=b,
            hn=hn,
            wn=wn,
            wh=k,
            ww=k,
        )
        if pad_h or pad_w:
            return y_pad[:, :, :, pad_top : pad_top + h, pad_left : pad_left + w]
        return y_pad


class ReCoWSpatialBranch(nn.Module):
    """Hard spatial routing with margin confidence."""

    def __init__(
        self,
        win_size: int = 4,
        proposal_w: int = 2,
        proposal_h: int = 2,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.proposal_w = proposal_w
        self.proposal_h = proposal_h
        self.num_centers = proposal_w * proposal_h
        self.use_residual = use_residual
        self.kernel_log_gamma = nn.Parameter(torch.zeros(1))
        self.conf_log_scale = nn.Parameter(torch.zeros(1))

    def _pairwise_euclidean_sim(self, centers: torch.Tensor, pix: torch.Tensor) -> torch.Tensor:
        centers_n = F.normalize(centers, dim=-1, eps=1e-12)
        pix_n = F.normalize(pix, dim=-1, eps=1e-12)
        dist2 = ((centers_n.unsqueeze(-2) - pix_n.unsqueeze(-3)) ** 2).sum(dim=-1)
        gamma = self.kernel_log_gamma.exp().clamp(max=100.0)
        return torch.exp(-gamma * dist2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, f"ReCoWSpatialBranch expects (B,C,G,H,W), got {x.shape}"
        b, c, g, h, w = x.shape
        assert g == g_order, f"Expected group order {g_order}, got {g}"

        k = self.win_size
        h_pad = _ceil_to_multiple(h, k)
        w_pad = _ceil_to_multiple(w, k)
        pad_h = h_pad - h
        pad_w = w_pad - w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        pad_bottom = pad_h - pad_top
        pad_right = pad_w - pad_left

        if pad_h or pad_w:
            x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
            valid = x.new_ones((b, 1, 1, h, w))
            valid_pad = F.pad(valid, (pad_left, pad_right, pad_top, pad_bottom))
        else:
            x_pad = x
            valid_pad = x.new_ones((b, 1, 1, h, w))

        hn = h_pad // k
        wn = w_pad // k
        spatial_score = x_pad.mean(dim=1).mean(dim=1)
        s_windows = rearrange(
            spatial_score,
            "b (hn wh) (wn ww) -> (b hn wn) 1 wh ww",
            hn=hn,
            wn=wn,
            wh=k,
            ww=k,
        )
        valid_windows = rearrange(
            valid_pad,
            "b 1 1 (hn wh) (wn ww) -> (b hn wn) 1 wh ww",
            hn=hn,
            wn=wn,
            wh=k,
            ww=k,
        )

        bw = s_windows.shape[0]
        num_centers = self.num_centers
        num_tokens = k * k
        num = F.adaptive_avg_pool2d(s_windows * valid_windows, (self.proposal_h, self.proposal_w))
        den_raw = F.adaptive_avg_pool2d(valid_windows, (self.proposal_h, self.proposal_w))
        centers_flat = (num / den_raw.clamp(min=1e-12)).view(bw, num_centers, 1)
        pix_flat = s_windows.view(bw, num_tokens, 1)

        sim = self._pairwise_euclidean_sim(centers_flat, pix_flat)
        center_valid = den_raw.view(bw, num_centers) > 0.0
        pix_valid = valid_windows.view(bw, num_tokens) > 0.0
        sim = sim.masked_fill(~center_valid.unsqueeze(-1), -1e9)
        sim = sim.masked_fill(~pix_valid.unsqueeze(1), -1e9)
        indices = sim.argmax(dim=1)

        top2 = sim.transpose(1, 2).topk(k=min(2, num_centers), dim=-1).values
        if num_centers >= 2:
            margin = (top2[..., 0] - top2[..., 1]).clamp(min=0.0)
        else:
            margin = top2[..., 0].clamp(min=0.0)
        conf_scale = self.conf_log_scale.exp().clamp(max=100.0)
        conf = torch.sigmoid(conf_scale * margin) * pix_valid.to(dtype=sim.dtype)

        x_windows = rearrange(
            x_pad,
            "b c g (hn wh) (wn ww) -> (b hn wn) c g wh ww",
            hn=hn,
            wn=wn,
            wh=k,
            ww=k,
        )
        pix_all = rearrange(x_windows, "bw c g h w -> bw g (h w) c")
        assign = F.one_hot(indices, num_classes=num_centers).to(dtype=pix_all.dtype)
        assign = assign * pix_valid.to(dtype=pix_all.dtype).unsqueeze(-1)

        sum_pix = torch.einsum("bnk,bgnc->bgkc", assign, pix_all)
        count = assign.sum(dim=1).clamp(min=1.0)
        cluster_mean = sum_pix / count.unsqueeze(1).unsqueeze(-1)
        idx = indices.unsqueeze(1).unsqueeze(-1).expand(bw, g, num_tokens, c)
        cluster_pix = cluster_mean.gather(2, idx)
        cluster_map = rearrange(cluster_pix, "bw g (h w) c -> bw c g h w", h=k, w=k)
        cluster_map = cluster_map * conf.view(bw, 1, 1, k, k)

        y_windows = x_windows + cluster_map if self.use_residual else cluster_map
        y_pad = rearrange(
            y_windows,
            "(b hn wn) c g wh ww -> b c g (hn wh) (wn ww)",
            b=b,
            hn=hn,
            wn=wn,
            wh=k,
            ww=k,
        )
        if pad_h or pad_w:
            return y_pad[:, :, :, pad_top : pad_top + h, pad_left : pad_left + w]
        return y_pad


class ReCoW(nn.Module):
    """Rotation-Equivariant Consistency Weighting."""

    def __init__(
        self,
        channels: int,
        win_size: int = 4,
        proposal_w: int = 2,
        proposal_h: int = 2,
        small_residual: bool = False,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if DREGCBA is None or PREGCBA is None:
            raise ImportError("DREGCBA/PREGCBA is required for ReCoW but could not be imported.")
        self.use_residual = use_residual
        self.spectral = ReCoWSpectralBranch(
            win_size=win_size,
            proposal_w=proposal_w,
            proposal_h=proposal_h,
            use_residual=small_residual,
        )
        self.spatial = ReCoWSpatialBranch(
            win_size=win_size,
            proposal_w=proposal_w,
            proposal_h=proposal_h,
            use_residual=small_residual,
        )
        self.spec_post = PREGCBA(channels, channels)
        self.spat_post = DREGCBA(channels, channels, k=3, s=1, p=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.spec_post(self.spectral(x))
        spat = self.spat_post(self.spatial(x))
        fused = spec + spat
        spec_inv = spec.mean(dim=2)
        spat_inv = spat.mean(dim=2)
        agree = F.cosine_similarity(spec_inv, spat_inv, dim=1, eps=1e-12).clamp(-1.0, 1.0)
        agree = ((agree + 1.0) * 0.5).unsqueeze(1).unsqueeze(2)
        y_gate = x * (self.gate(fused) * agree)
        return x + y_gate if self.use_residual else y_gate


# Compatibility aliases for checkpoints and external configs created before
# the public paper-facing name was standardized.
ReCoWAAA = ReCoW
ReCoWAAASpectralBranch = ReCoWSpectralBranch
ReCoWAAASpatialBranch = ReCoWSpatialBranch

__all__ = [
    "ReCoW",
    "ReCoWSpectralBranch",
    "ReCoWSpatialBranch",
]
