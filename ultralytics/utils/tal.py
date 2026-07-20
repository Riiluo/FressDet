import torch
import torch.nn as nn

from .metrics import probiou
from .ops import xywhr2xyxyxyxy


class RotatedTaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, stride=None, topk2=None):
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long, device=gt_labels.device)
        ind[0] = torch.arange(end=self.bs, device=gt_labels.device).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = probiou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)
        return bbox_scores.pow(self.alpha) * overlaps.pow(self.beta), overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        target_labels.clamp_(0)

        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        return target_labels, target_bboxes, torch.where(fg_scores_mask > 0, target_scores, 0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt=None):
        corners = xywhr2xyxyxyxy(gt_bboxes)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)

        return mask_pos.argmax(-2), fg_mask, mask_pos


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    lt, rb = pred_dist.split(2, dim=dim)  # [2,23142,2] [2,23142,2]
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)  # [2,23142,1] [2,23142,1] 
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)  # [2,23142,1] [2,23142,1]
    x, y = xf * cos - yf * sin, xf * sin + yf * cos  # [2,23142,1] [2,23142,1]
    xy = torch.cat([x, y], dim=dim) + anchor_points  # [2,23142,2]
    return torch.cat([xy, lt + rb], dim=dim)  # [2,23142,4]
